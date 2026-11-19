"""
Dynamic packed column with full energy balance and pressure dynamics.

Discretisation
    The bed is split into N segments of height dz (top = 1). Each segment holds a
    well-mixed liquid holdup (component moles) in phase equilibrium with the vapour
    leaving it at the local pressure. Vapour passes upward in plug flow; the segment
    Murphree efficiency follows from rate-based transfer units

        E_MV = 1 - exp(-dz / H_OG),   H_OG = H_G + lambda H_L,  lambda = m V / L

    with H_G, H_L from packing-specific correlations (packing.py). As N grows this
    converges to a countercurrent rate-based (HTU-NTU) model.

Liquid hydraulics
    Liquid holdup h_L = M / (rho_m A dz); the outflow follows from the inverse
    Billet-Schultes holdup correlation u_L(h_L) (no weirs: pure trickle flow).

Energy balance (index-reduced, exact for bubble-point liquid)
    Stage enthalpy H = sum n_i h_i(T) + C_metal T with T = T_bub(x, P). Differentiating
    with the chain rule gives, for every stage, an equation linear in the unknown vapour
    rate V_k and in dP/dt:

        V_k (H_V,k - sum g_i y_i,k) = E_in,k - sum g_i A_i,k - C_k (dT/dP)_k dP/dt
        g_i = h_i + C_k dT/dn_i

    i.e. sensible-heat storage in liquid + packing metal, flashing on pressure change,
    sub-cooled feeds/reflux condensing vapour, heat losses and reboiler duty are all
    resolved. Solved bottom-up: V_k = a_k + b_k dP/dt.

Pressure dynamics
    One lumped vapour space per column (packing voids + heads + sump space) with
    top pressure P as a state:
        (V_void / R T) dP/dt = V_1(dP/dt) - V_exit(P, P_downstream)
    The packing pressure-drop profile is a fast first-order state toward the
    Billet-Schultes irrigated dP. The large effective capacitance (b_1 << 0) from
    flashing liquid is what makes vacuum columns pressure-sluggish.

Bottom options
    'reboiler'   : kettle sump is an equilibrium stage receiving reboiler duty
    'vapor_feed' : vapour enters below the bed; liquid leaves to an external tank
"""
import numpy as np

from .core import Unit
from .thermo import (E, W, X, G, CP_L, R_GAS, MU_VAP, abv, bubble_point, liquid_props,
                     hv_comp, h_liq, h_vap, mu_liq, sigma_liq, D_liq, D_vap, rho_vap,
                     latent_heat)
from . import packing as pkg

BAR = 1e5
STEEL_CP = 500.0


class PackedColumn(Unit):
    kind = "packed column"
    io_keys = {"P_top_mbar", "P_bottom", "dP_bed_mbar", "T_top", "T_C", "abv_B", "xE_B", "level",
               "V_top", "boilup", "flood_max", "HETP_mean", "P_out"}
    def __init__(self, name, packing, diameter, height, n_seg, bottom="reboiler",
                 P_top0=0.10, dP0=0.003, sump_area=0.8, sump_level0=0.6, sump_metal_kg=1500.0,
                 duty=None, vapor_in=None, vapor_exit=None, feeds=(), bottoms=None,
                 x_init=None, T_init=315.0, L_nom_m3h=5.0, yv0=0.1, mt_factor=1.0,
                 UA_loss=0.5, T_amb=293.15, shell_t=0.005, head_height=1.5, dP_tau=1.0):
        super().__init__(name)
        self.pk_name = packing
        self.pk = pkg.get(packing)
        self.D, self.H, self.N = diameter, height, int(n_seg)
        self.A = np.pi * diameter ** 2 / 4.0
        self.dz = height / n_seg
        self.eq_sump = bottom == "reboiler"
        self.ns = self.N + (1 if self.eq_sump else 0)
        self.n_states = 3 * self.ns + 3
        self.P_top0, self.dP0, self.yv0 = P_top0, dP0, yv0
        self.sump_area, self.sump_level0 = sump_area, sump_level0
        self.duty, self.vapor_in, self.vapor_exit = duty, vapor_in, vapor_exit
        self.feeds, self.bottoms = list(feeds), bottoms
        self.x_init, self.T_init, self.L_nom = x_init, T_init, L_nom_m3h
        self.mt_factor, self.T_amb, self.dP_tau = mt_factor, T_amb, dP_tau
        # metal heat capacity per stage (packing + shell), sump vessel
        seg_metal = self.pk["rho_bulk"] * self.A * self.dz + np.pi * diameter * self.dz * shell_t * 7900
        self.Cm = np.full(self.ns, seg_metal * STEEL_CP)
        if self.eq_sump:
            self.Cm[-1] = sump_metal_kg * STEEL_CP
        # external area for heat loss (W/K per stage), insulated shell
        self.UAl = np.full(self.ns, UA_loss * np.pi * diameter * self.dz)
        if self.eq_sump:
            self.UAl[-1] = UA_loss * (np.pi * diameter * 2.0 + self.A)
        # vapour space volume
        self.V_void = self.pk["eps"] * self.A * height + self.A * head_height \
            + (sump_area * 1.0 if self.eq_sump else self.A * 1.0)
        self.frac = (np.arange(self.N) + 0.5) / self.N
        self._T = None

    # ------------------------------------------------------------------ setup
    def add_feed(self, segment, stream):
        self.feeds.append((segment, stream))

    def x0(self):
        xi = np.atleast_2d(self.x_init)
        if xi.shape[0] == 1:
            xi = np.repeat(xi, self.ns, axis=0)
        T = np.full(self.ns, self.T_init)
        rho_m, rho, _ = liquid_props(xi)
        mu = mu_liq(T, xi)
        uL = self.L_nom / 3600.0 / self.A
        hL = pkg.holdup(self.pk, uL, rho[:self.N], mu[:self.N])
        M = np.empty(self.ns)
        M[:self.N] = hL * self.A * self.dz * rho_m[:self.N]
        if self.eq_sump:
            M[-1] = self.sump_area * self.sump_level0 * rho_m[-1]
        n = M[:, None] * xi
        return np.concatenate([n.ravel(), [self.P_top0, self.dP0, self.yv0]])

    def atol(self):
        a = np.full(self.n_states, 1e-5)
        a[-3:] = [1e-7, 1e-8, 1e-7]
        return a

    def state_names(self):
        st = [f"seg{k+1}" for k in range(self.N)] + (["sump"] if self.eq_sump else [])
        nm = [f"{self.name}.{s}.n{c}" for s in st for c in ("EtOH", "Water", "Extract")]
        return nm + [f"{self.name}.P_top", f"{self.name}.dP_bed", f"{self.name}.y_vap"]

    # -------------------------------------------------------- algebraic outputs
    def outputs(self, t, s):
        N, ns = self.N, self.ns
        n = np.maximum(s[:3 * ns].reshape(ns, 3), 1e-12)
        M = n.sum(axis=1)
        x = n / M[:, None]
        Ptop, dP, yv = s[3 * ns], max(s[3 * ns + 1], 0.0), float(np.clip(s[3 * ns + 2], 0, 1))
        P = Ptop + dP * self.frac
        if self.eq_sump:
            P = np.append(P, Ptop + dP)
        T, ys, _ = bubble_point(x, P, self._T)
        self._T = T
        rho_m, rho, _ = liquid_props(x)
        mu = mu_liq(T, x)
        hL = M[:N] / rho_m[:N] / (self.A * self.dz)
        uL = pkg.uL_from_holdup(self.pk, hL, rho[:N], mu[:N])
        L = uL * self.A * rho_m[:N]
        self._st = dict(n=n, M=M, x=x, P=P, T=T, ys=ys, rho_m=rho_m, rho=rho, mu=mu,
                        hL=hL, uL=uL, L=L)
        n_v = Ptop * BAR * self.V_void / (R_GAS * T[0])
        zv = np.array([yv, 1.0 - yv, 0.0])
        o = self.out
        o.update(P_top=Ptop, P_top_mbar=Ptop * 1e3, dP_bed_mbar=dP * 1e3,
                 P_bottom=Ptop + dP, vapor={"z": zv, "T": T[0], "P": Ptop},
                 T_top=T[0] - 273.15, T_bed_bottom=T[N - 1] - 273.15,
                 inv=n.sum(axis=0) + n_v * zv, T_profile=T - 273.15,
                 abv_profile=abv(x), hL_profile=hL, x_profile=x)
        for k in range(N):
            o[f"T{k+1}"] = T[k] - 273.15
        if self.eq_sump:
            level = M[-1] / rho_m[-1] / self.sump_area
            o.update(level=level, P_out=Ptop + dP + rho[-1] * G * level / BAR, z=x[-1],
                     T=T[-1], T_C=T[-1] - 273.15, T_sump=T[-1], SG=rho[-1] / 1000.0,
                     rho_m=rho_m[-1], abv_B=abv(x[-1])[0], xE_B=x[-1, E])
        else:
            o.update(liquid_out={"F": L[-1], "z": x[N - 1], "T": T[N - 1]})

    # ----------------------------------------------------------- derivatives
    def derivatives(self, t, s):
        N, ns, st, pk = self.N, self.ns, self._st, self.pk
        n, M, x, P, T, ys, L = st["n"], st["M"], st["x"], st["P"], st["T"], st["ys"], st["L"]
        Ptop = s[3 * ns]

        # --- sensitivities of bubble T (dT/dn_i, dT/dP) and equilibrium slope m
        dn = 1e-5 * M
        nn = np.repeat(n[None], 3, axis=0)                     # (3, ns, 3)
        for i in range(3):
            nn[i, :, i] += dn
        xx = (nn / nn.sum(axis=2)[:, :, None]).reshape(3 * ns, 3)
        xs_all = np.vstack([xx, x])
        Ps_all = np.concatenate([P, P, P, P + 1e-5])
        Tp, yp, _ = bubble_point(xs_all, Ps_all, np.tile(T, 4), maxit=6)
        Tp = Tp.reshape(4, ns)
        dTdn = ((Tp[:3] - T[None]) / dn[None]).T
        dTdP = (Tp[3] - T) / 1e-5
        dxE = np.maximum(xx[:ns, E] - x[:, E], 1e-14)
        m_slope = np.maximum((yp[:ns, E] - ys[:, E]) / dxE, 1e-3)

        hl_i = CP_L[None, :] * (T[:, None] - 273.15 - 25.0)
        h = np.sum(x * hl_i, axis=1)
        hv = hv_comp(T)
        Ck = M * (x @ CP_L) + self.Cm
        g = hl_i + Ck[:, None] * dTdn

        # --- liquid in/out per stage (component & enthalpy flows)
        lic, liH = np.zeros((ns, 3)), np.zeros(ns)
        loc, loH = np.zeros((ns, 3)), np.zeros(ns)
        loc[:N] = L[:, None] * x[:N]
        loH[:N] = L * h[:N]
        lic[1:N + (1 if self.eq_sump else 0)] += loc[:N - (0 if self.eq_sump else 1)]
        liH[1:N + (1 if self.eq_sump else 0)] += loH[:N - (0 if self.eq_sump else 1)]
        Ffeed = 0.0
        for seg, ref in self.feeds:
            f = ref()
            z = np.asarray(f["z"])
            lic[seg - 1] += f["F"] * z
            liH[seg - 1] += f["F"] * float(h_liq(f["T"], z))
            Ffeed += f["F"]
        Q = -self.UAl * (T - self.T_amb)
        if self.eq_sump:
            B = self.bottoms() if self.bottoms else 0.0
            loc[-1] = B * x[-1]
            loH[-1] = B * h[-1]
            Q[-1] += self.duty()

        # --- vapour-rate estimate for hydraulics / transfer units (state based)
        V_exit = self.vapor_exit() if self.vapor_exit else 0.0
        if self.eq_sump:
            V_bot = max(self.duty(), 0.0) / float(latent_heat(T[-1]) @ ys[-1, :2])
        else:
            vin = self.vapor_in()
            V_bot = vin["F"]
        V_est = np.maximum(V_exit + (V_bot - V_exit) * self.frac, 1e-3)
        Ts, Ps = T[:N], P[:N]
        rhoV, mwV = rho_vap(Ts, Ps, ys[:N])
        uV_est = V_est * mwV / rhoV / self.A
        rhoL, muL = st["rho"][:N], st["mu"][:N]
        sig = sigma_liq(Ts, x[:N])
        HG, HLt, ae = pkg.transfer_units(pk, uV_est, st["uL"], st["hL"], rhoL, muL, sig,
                                         D_liq(Ts, muL), rhoV, MU_VAP, D_vap(Ts, Ps),
                                         self.mt_factor)
        lam = m_slope[:N] * V_est / np.maximum(L, 1e-6)
        HOG = HG + lam * HLt
        Emv = 1.0 - np.exp(-self.dz / HOG)

        # --- bottom-up energy-balance sweep: V_k = a_k + b_k dP/dt
        a, b = np.empty(ns), np.empty(ns)
        Y, Yin = np.empty((ns, 3)), np.empty((ns, 3))
        ain, bin_ = np.empty(ns), np.empty(ns)
        if self.eq_sump:
            a_i, b_i, y_i, Hv_i = 0.0, 0.0, np.zeros(3), 0.0
        else:
            a_i, b_i = vin["F"], 0.0
            y_i = np.asarray(vin["z"], dtype=float)
            Hv_i = float(h_vap(vin["T"], y_i))
        for k in range(ns - 1, -1, -1):
            yk = ys[k] if (self.eq_sump and k == ns - 1) else y_i + Emv[k] * (ys[k] - y_i)
            gk = g[k]
            A0 = lic[k] - loc[k] + a_i * y_i
            N0 = liH[k] - loH[k] + a_i * Hv_i + Q[k] - gk @ A0
            N1 = b_i * (Hv_i - gk @ y_i) - Ck[k] * dTdP[k]
            Hvk = hv[k] @ yk
            Dk = Hvk - gk @ yk
            a[k], b[k] = N0 / Dk, N1 / Dk
            Y[k], Yin[k], ain[k], bin_[k] = yk, y_i, a_i, b_i
            a_i, b_i, y_i, Hv_i = a[k], b[k], yk, Hvk

        c = BAR * self.V_void / (R_GAS * T[0])
        dPdt = (a[0] - V_exit) / (c - b[0])
        V = a + b * dPdt
        Vin = ain + bin_ * dPdt
        dndt = lic - loc + Vin[:, None] * Yin - V[:, None] * Y

        n_v = c * Ptop
        dyv = max(V[0], 0.0) * (Y[0, E] - s[3 * ns + 2]) / n_v

        # --- packing pressure drop with actual vapour flows
        uV = V[:N] * mwV / rhoV / self.A
        dPm, ff = pkg.pressure_drop(pk, self.D, uV, rhoV, MU_VAP, st["uL"], st["hL"])
        dP_target = float(np.sum(dPm) * self.dz) / BAR
        ddP = (dP_target - s[3 * ns + 1]) / self.dP_tau

        het = pkg.hetp(HOG, lam)
        self.out.update(V_profile=V, Emv_profile=Emv, HETP_profile=het, flood_profile=ff,
                        V_top=V[0], V_bottom=V[N - 1], dPdt_mbar_s=dPdt * 1e3,
                        flood_max=float(ff.max()) * 100.0, HETP_mean=float(np.mean(het)),
                        uL_m3m2h=float(st["uL"].mean() * 3600), V_min=float(V.min()),
                        hL_mean=float(st["hL"].mean()))
        if self.eq_sump:
            self.out.update(boilup=V[-1])
        return np.concatenate([dndt.ravel(), [dPdt, ddP, dyv]])
