"""Instrumentation, control and conservation monitoring."""
import numpy as np

from .core import Unit


class Analyzer(Unit):
    """Measurement with n first-order lags (e.g. online density/NIR ABV analyser)."""
    kind = "analyser"
    io_keys = {"value"}

    def __init__(self, name, signal, tau=30.0, n_lags=2, init=0.0):
        super().__init__(name)
        self.signal, self.tau, self.n_states, self.init = signal, tau, n_lags, init

    def x0(self):
        return np.full(self.n_states, float(self.init))

    def outputs(self, t, s):
        self.out.update(value=float(s[-1]), raw=float(self.signal()))

    def derivatives(self, t, s):
        u = np.concatenate([[self.signal()], s[:-1]])
        return (u - s) / (self.tau / self.n_states)


class PID(Unit):
    """
    ISA-form PID:  op = bias + Kc [ e + (1/Ti) int(e) dt - Td dPV_f/dt ]   (e scaled by span)
    action 'reverse': op rises when PV < SP;  'direct': op rises when PV > SP.
    Derivative on filtered PV, conditional-integration anti-windup, AUTO/MAN.
    """
    n_states = 2
    kind = "controller"
    io_keys = {"pv", "sp", "op"}

    def __init__(self, name, pv, sp, Kc, Ti, Td=0.0, action="reverse", span=1.0,
                 op_lo=0.0, op_hi=1.0, bias=0.5, N=10.0, mode="auto", units=""):
        super().__init__(name)
        self.pv, self.sp = pv, sp
        self.Kc, self.Ti, self.Td, self.N = Kc, Ti, Td, N
        self.action = action
        self.span, self.op_lo, self.op_hi, self.bias = span, op_lo, op_hi, bias
        self.mode, self.op_man, self.units = mode, bias, units

    @property
    def sign(self):
        return 1.0 if self.action == "reverse" else -1.0

    def atol(self):
        return np.array([1e-3, 1e-7])

    def state_names(self):
        return [f"{self.name}.I", f"{self.name}.pv_f"]

    def outputs(self, t, s):
        pv = float(self.pv())
        e = self.sign * (self.sp - pv) / self.span
        D = 0.0
        if self.Td > 0:
            D = -self.sign * self.Kc * self.Td * (pv - s[1]) / (self.Td / self.N) / self.span
        u = self.bias + self.Kc * (e + s[0] / max(self.Ti, 1e-6)) + D
        op = float(np.clip(u, self.op_lo, self.op_hi)) if self.mode == "auto" else self.op_man
        self._e, self._u, self._pv = e, u, pv
        self.out.update(op=op, pv=pv, sp=float(self.sp), err=e)

    def derivatives(self, t, s):
        dI = self._e
        if self.mode != "auto" or (self._u > self.op_hi and dI > 0) or (self._u < self.op_lo and dI < 0):
            dI = 0.0
        dpf = (self._pv - s[1]) / (self.Td / self.N) if self.Td > 0 else 0.0
        return np.array([dI, dpf])

    def set_mode(self, fs, X, mode, op=None):
        """Bumpless AUTO/MAN switch for use in schedule actions."""
        cur = self.out.get("op", self.bias)
        if mode == "manual":
            self.op_man = cur if op is None else op
        else:
            self.bias = cur
            X[self.sl.start] = 0.0
        self.mode = mode
        return X


class BalanceMonitor(Unit):
    """
    Integrates boundary flows and compares with holdup inventories:
        closure = [inv(t) - inv(0) - (in - out)] / reference
    Because both sides are produced by the same integrator, the closure is a direct
    measure of accumulated numerical (and any model) conservation error.
    """
    n_states = 6
    steady_exclude = True
    kind = "balance monitor"

    def __init__(self, name, inflows, outflows, holdups):
        super().__init__(name)
        self.inflows, self.outflows, self.holdups = inflows, outflows, holdups
        self.inv0 = None

    def atol(self):
        return np.full(6, 1e-6)

    def state_names(self):
        return [f"{self.name}.in_{c}" for c in "EWX"] + [f"{self.name}.out_{c}" for c in "EWX"]

    def on_start(self, X):
        self.inv0 = sum(u.out["inv"] for u in self.holdups)
        X[self.sl] = 0.0
        return X

    def outputs(self, t, s):
        inv = sum(u.out["inv"] for u in self.holdups)
        if self.inv0 is None:
            self.inv0 = inv.copy()
        net = s[:3] - s[3:]
        err = inv - self.inv0 - net
        ref_tot = max(self.inv0.sum(), s[:3].sum(), 1.0)
        ref_E = max(self.inv0[0], s[0], 1e-3)
        self.out.update(closure_total=float(err.sum() / ref_tot), closure_EtOH=float(err[0] / ref_E),
                        inventory_mol=float(inv.sum()), EtOH_in_mol=float(s[0]),
                        EtOH_out_mol=float(s[3]))

    def derivatives(self, t, s):
        d = np.zeros(6)
        for r in self.inflows:
            st = r()
            d[:3] += st["F"] * np.asarray(st["z"])
        for r in self.outflows:
            st = r()
            d[3:] += st["F"] * np.asarray(st["z"])
        return d
