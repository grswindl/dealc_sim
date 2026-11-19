"""
Core modelling framework: Units, signal references, Flowsheet, schedule and a
streaming stiff solver with accuracy monitoring.

gPROMS analogy
--------------
* ``Unit``      ~ MODEL: differential states, algebraic ``outputs``, ``derivatives``
* ``Ref``       ~ connection/port: live reference to another unit's output
* ``Flowsheet`` ~ composite MODEL + PROCESS (SCHEDULE of RESET/SWITCH tasks)
* ``Flowsheet.run`` ~ integrator with live reporting (generator), structural
  Jacobian-sparsity detection and per-step local error estimates.

Causality rule (keeps the plant an explicit ODE, no algebraic loops): a unit's
``outputs`` may use only its own states and outputs of units added earlier.
``derivatives`` may use anything. Flows leaving holdups are functions of states.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.integrate import BDF
from scipy.sparse import lil_matrix, csc_matrix


class Ref:
    """Live reference to ``unit.out[key]`` (optionally indexed / scaled)."""

    def __init__(self, unit, key, index=None, scale=1.0):
        self.unit, self.key, self.index, self.scale = unit, key, index, scale

    def __call__(self):
        v = self.unit.out[self.key]
        if self.index is not None:
            v = v[self.index]
        return v * self.scale if self.scale != 1.0 else v

    def __repr__(self):
        return f"Ref({self.unit.name}.{self.key})"


def as_signal(v):
    return v if callable(v) else (lambda: v)


class Unit:
    n_states = 0
    steady_exclude = False      # e.g. integrators of boundary flows (never steady)
    io_keys = None              # outputs that are interface (port) variables; None = all
    kind = "unit"

    def __init__(self, name):
        self.name = name
        self.out = {}
        self.sl = slice(0, 0)

    def x0(self):
        return np.zeros(self.n_states)

    def state_names(self):
        return [f"{self.name}.s{i}" for i in range(self.n_states)]

    def atol(self):
        return np.full(self.n_states, 1e-6)

    def outputs(self, t, s):
        pass

    def derivatives(self, t, s):
        return np.zeros(self.n_states)


@dataclass(order=True)
class Event:
    t: float
    label: str = field(compare=False)
    action: Callable = field(compare=False)
    touches: tuple = field(compare=False, default=())   # (unit, attr) restored after a run


@dataclass
class Update:
    t: float
    rows: list
    steps: list
    events: list
    X: np.ndarray
    done: bool = False


class Results:
    def __init__(self, df, solver_df, X_final, events):
        self.df, self.solver, self.X, self.events = df, solver_df, X_final, events

    @property
    def hours(self):
        return self.df.index.values / 3600.0


class Flowsheet:
    def __init__(self, name="flowsheet"):
        self.name = name
        self.units: list[Unit] = []
        self._by = {}
        self.events: list[Event] = []
        self.n = 0
        self.sparsity = None
        self.meta = {}

    # ----------------------------------------------------------- construction
    def add(self, *units):
        for u in units:
            if u.name in self._by:
                raise ValueError(f"duplicate unit {u.name}")
            u.sl = slice(self.n, self.n + u.n_states)
            self.n += u.n_states
            self.units.append(u)
            self._by[u.name] = u
        return units[0] if len(units) == 1 else units

    def __getitem__(self, name):
        return self._by[name]

    def __contains__(self, name):
        return name in self._by

    def x0(self):
        return np.concatenate([u.x0() for u in self.units])

    def atol(self):
        return np.concatenate([u.atol() for u in self.units])

    def state_names(self):
        return sum((u.state_names() for u in self.units), [])

    # ---------------------------------------------------------------- equations
    def evaluate(self, t, X):
        for u in self.units:
            u.outputs(t, X[u.sl])

    def rhs(self, t, X):
        self.evaluate(t, X)
        dX = np.empty_like(X)
        for u in self.units:
            dX[u.sl] = u.derivatives(t, X[u.sl])
        return dX

    def snapshot(self, t, X, profiles=True, states=True):
        """
        All scalar outputs (after a full RHS pass, so derivative-phase values are fresh).
        1-D array outputs (column profiles) are expanded as ``unit.key[i]`` (1-based) and
        differential states are added as ``unit.state.<name>``.
        """
        self.rhs(t, X)
        row = {}
        for u in self.units:
            for k, v in u.out.items():
                if isinstance(v, (float, int, np.floating, np.integer)) and not isinstance(v, bool):
                    row[f"{u.name}.{k}"] = float(v)
                elif profiles and isinstance(v, np.ndarray) and v.ndim == 1 and 1 < v.size <= 200:
                    for i, vi in enumerate(v):
                        row[f"{u.name}.{k}[{i+1}]"] = float(vi)
            if states and u.n_states:
                for nm, xv in zip(u.state_names(), X[u.sl]):
                    row[u.name + ".state." + nm[len(u.name) + 1:]] = float(xv)
        return row

    # ----------------------------------------------------------------- schedule
    def at(self, t, action, label="", touches=()):
        self.events.append(Event(t, label, action, tuple(touches)))
        self.events.sort()

    def reset(self, t, path, value, label=None):
        unit, attr = path.split(".", 1)

        def act(fs, X):
            setattr(fs[unit], attr, value)
        self.at(t, act, label or f"{path} = {value}", touches=[(unit, attr)])

    def clear_schedule(self):
        self.events = []

    # ------------------------------------------------------- structural analysis
    def detect_sparsity(self, X, n_points=3, seed=0):
        """Jacobian sparsity by perturbation at X and at randomly perturbed copies."""
        rng = np.random.default_rng(seed)
        n = len(X)
        S = lil_matrix((n, n), dtype=bool)
        atol = self.atol()
        pts = [X] + [X * (1.0 + 0.03 * rng.standard_normal(n)) for _ in range(n_points - 1)]
        for Xp in pts:
            f0 = self.rhs(0.0, Xp)
            for j in range(n):
                dx = 1e-6 * max(abs(Xp[j]), atol[j] * 1e2, 1e-9)
                X2 = Xp.copy()
                X2[j] += dx
                nz = np.nonzero(self.rhs(0.0, X2) != f0)[0]
                for i in nz:
                    S[i, j] = True
        for i in range(n):
            S[i, i] = True
        self.sparsity = csc_matrix(S)
        self.evaluate(0.0, X)
        return self.sparsity

    # ---------------------------------------------------------------- solution
    def run(self, t_end, X0=None, t0=0.0, report_dt=15.0, rtol=1e-5, atol_scale=1.0,
            jac="sparse", yield_every=0.4, record=True):
        """
        Generator: integrates with a variable-order BDF, restarting at scheduled events.
        Yields ``Update`` objects every ``yield_every`` s of wall time with new report
        rows (process outputs at ``report_dt`` spacing) and per-step solver statistics:
        step size, order, local error estimate (weighted RMS, <=1 means within tolerance),
        dominant error state, cumulative RHS/Jacobian/LU counts.
        """
        X = self.x0() if X0 is None else np.array(X0, dtype=float)
        saved = {}
        for e in self.events:
            for (u, a) in e.touches:
                saved.setdefault((u, a), getattr(self[u], a))
        try:
            yield from self._run(t_end, X, t0, report_dt, rtol, atol_scale, jac, yield_every, record)
        finally:                      # every run starts from the base-case parameters
            for (u, a), v in saved.items():
                setattr(self[u], a, v)

    def _run(self, t_end, X, t0, report_dt, rtol, atol_scale, jac, yield_every, record):
        atol = self.atol() * atol_scale
        names = self.state_names()
        self.evaluate(t0, X)
        for u in self.units:
            if hasattr(u, "on_start"):
                X = u.on_start(X)
        ev = [e for e in self.events if t0 <= e.t < t_end]
        bounds = sorted(set([t0] + [e.t for e in ev] + [t_end]))
        wall0 = time.time()
        last_yield = wall0
        rows, steps, fired = [], [], []
        next_rep = t0
        tot = dict(nfev=0, njev=0, nlu=0, nsteps=0)
        for a, b in zip(bounds[:-1], bounds[1:]):
            for e in ev:
                if e.t == a:
                    r = e.action(self, X)
                    X = X if r is None else r
                    fired.append((a, e.label))
            use_sp = self.sparsity if (jac == "sparse" and self.sparsity is not None) else None
            solver = BDF(self.rhs, a, X, b, rtol=rtol, atol=atol, jac_sparsity=use_sp)
            prev_dense = None
            base = dict(tot)
            while solver.status == "running":
                k_used = solver.order
                t_old = solver.t
                msg = solver.step()
                if solver.status == "failed":
                    if use_sp is not None:          # retry segment with a dense Jacobian
                        use_sp = None
                        solver = BDF(self.rhs, t_old, X, b, rtol=rtol, atol=atol)
                        continue
                    raise RuntimeError(f"solver failed at t={solver.t:.2f}s: {msg}")
                dense = solver.dense_output()
                y = solver.y
                err, lead = np.nan, ""
                if prev_dense is not None:
                    scale = atol + rtol * np.abs(y)
                    e = solver.error_const[k_used] * (y - prev_dense(solver.t)) / scale
                    err = float(np.sqrt(np.mean(e ** 2)))
                    lead = names[int(np.argmax(np.abs(e)))]
                prev_dense = dense
                tot.update(nfev=base["nfev"] + solver.nfev, njev=base["njev"] + solver.njev,
                           nlu=base["nlu"] + solver.nlu, nsteps=tot["nsteps"] + 1)
                steps.append(dict(t=solver.t, h=solver.t - t_old, order=k_used, err=err,
                                  err_state=lead, wall=time.time() - wall0, **tot))
                if record:
                    while next_rep <= solver.t + 1e-9:
                        Xr = dense(next_rep) if next_rep > t_old else X
                        row = self.snapshot(next_rep, Xr)
                        row["t"] = next_rep
                        rows.append(row)
                        next_rep += report_dt
                X = solver.y.copy()
                if time.time() - last_yield > yield_every:
                    yield Update(solver.t, rows, steps, fired, X)
                    rows, steps, fired = [], [], []
                    last_yield = time.time()
        if record and (not rows or rows[-1]["t"] < t_end - 1e-6):
            row = self.snapshot(t_end, X)
            row["t"] = t_end
            rows.append(row)
        self.evaluate(t_end, X)
        yield Update(t_end, rows, steps, fired, X, done=True)

    def simulate(self, t_end, X0=None, t0=0.0, verbose=True, **kw):
        rows, steps, fired = [], [], []
        X = X0
        for upd in self.run(t_end, X0=X0, t0=t0, **kw):
            rows += upd.rows
            steps += upd.steps
            fired += upd.events
            X = upd.X
            if verbose:
                for t, lab in upd.events:
                    print(f"  t = {t/3600:6.3f} h : {lab}")
        df = pd.DataFrame(rows).set_index("t") if rows else pd.DataFrame()
        sdf = pd.DataFrame(steps)
        if verbose and len(sdf):
            print(f"  {len(sdf)} steps, {int(sdf.nfev.iloc[-1])} RHS evals, "
                  f"{int(sdf.njev.iloc[-1])} Jacobians, wall {sdf.wall.iloc[-1]:.1f} s")
        return Results(df, sdf, X, fired)

    def initialize(self, X0=None, horizon=3 * 3600.0, tol=0.05, max_rounds=3, verbose=True,
                   progress=None, newton=True, **kw):
        """
        Steady state by pseudo-transient continuation (closed loop, schedule disabled).
        Converged when every state's relative rate of change is below ``tol`` per hour.
        """
        saved, self.events = self.events, []
        X = self.x0() if X0 is None else X0
        floor = self.atol() * 1e3
        mask = np.ones(self.n, dtype=bool)
        for u in self.units:
            if u.steady_exclude:
                mask[u.sl] = False
        r = np.inf
        for i in range(max_rounds):
            for upd in self.run(horizon, X0=X, record=False, **kw):
                X = upd.X
                if progress:
                    progress(i, upd.t / horizon)
            f = self.rhs(0.0, X)
            r = float(np.max((np.abs(f) / np.maximum(np.abs(X), floor))[mask]) * 3600)
            if verbose:
                print(f"  initialisation round {i+1}: max relative rate = {r:.2e} 1/h")
            if r < tol:
                break
            if newton:
                Xn, ok = self.newton_polish(X)
                if ok:
                    X = Xn
                    f = self.rhs(0.0, X)
                    r = float(np.max((np.abs(f) / np.maximum(np.abs(X), floor))[mask]) * 3600)
                    if verbose:
                        print(f"  Newton polish: max relative rate = {r:.2e} 1/h")
                    if r < tol:
                        break
        self.events = saved
        self.evaluate(0.0, X)
        return X, r

    def newton_polish(self, X, xtol=1e-10):
        """Solve f(X)=0 (excluding steady_exclude states) by a trust-region Newton method."""
        from scipy.optimize import root
        mask = np.ones(self.n, dtype=bool)
        for u in self.units:
            if u.steady_exclude:
                mask[u.sl] = False
        f0 = self.rhs(0.0, X)
        mask &= ~((np.abs(X) < 1e-9) & (np.abs(f0) < 1e-12))     # structurally inactive (e.g. no extract)
        nonneg = X >= 0.0
        sc = np.maximum(np.abs(X), self.atol() * 1e3)[mask]
        Xb = X.copy()

        def F(z):
            Xz = Xb.copy()
            Xz[mask] = z * sc
            return self.rhs(0.0, Xz)[mask] / sc * 3600.0
        try:
            sol = root(F, X[mask] / sc, method="hybr", options={"xtol": xtol, "maxfev": 40 * mask.sum()})
        except Exception:
            return X, False
        Xn = X.copy()
        Xn[mask] = sol.x * sc
        Xn[nonneg] = np.maximum(Xn[nonneg], 0.0)
        ok = bool(np.all(np.isfinite(Xn)) and np.max(np.abs(F(sol.x))) < np.max(np.abs(F(X[mask] / sc))))
        return (Xn, ok)
