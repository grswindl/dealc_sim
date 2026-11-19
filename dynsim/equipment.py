"""
Equipment models: sources, valves, pumps, liquid & vapour lines, heat exchangers,
tanks, vacuum condenser (with non-condensables) and liquid-ring vacuum pump.

Stream convention: dict(F=[mol/s], z=array(3) mole fractions, T=[K]).
Any unit that liquid can be drawn from exposes out keys:
    P_out [bar at outlet nozzle], z, T, SG, rho_m [mol/m3], level [m]
"""
import numpy as np

from .core import Unit, as_signal
from .thermo import (E, W, X, G, CP_L, R_GAS, MW_AIR, abv, beer_composition,
                     bubble_point, latent_heat, liquid_props, h_vap, rho_vap)

BAR = 1e5
CP_WATER = 4186.0
P_ATM = 1.01325


def softplus(v, d=0.01):
    """Smooth max(v, 0)."""
    return d * np.logaddexp(0.0, np.asarray(v) / d)


def _smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def stream_heat_capacity(st):
    return st["F"] * float(np.asarray(st["z"]) @ CP_L)


def counterflow_effectiveness(NTU, Cr):
    if Cr > 0.999:
        return NTU / (1.0 + NTU)
    e = np.exp(-NTU * (1.0 - Cr))
    return (1.0 - e) / (1.0 - Cr * e)


# =============================================================================
class FeedSource(Unit):
    """Beer supply (bright-beer tank + feed pump) at fixed pressure and temperature."""
    kind = "source"

    def __init__(self, name, abv=5.0, extract_gL=50.0, P=3.0, T=277.15):
        super().__init__(name)
        self.abv, self.extract_gL, self.P, self.T = abv, extract_gL, P, T

    def outputs(self, t, s):
        z = beer_composition(self.abv, self.extract_gL)
        rho_m, rho, _ = liquid_props(z)
        self.out.update(z=z, P_out=self.P, SG=rho[0] / 1000.0, rho_m=rho_m[0],
                        level=10.0, T=self.T, abv=self.abv, T_C=self.T - 273.15)


class Valve(Unit):
    """
    Control valve: inherent characteristic + first-order actuator.
    Kv [m3/h, 1 bar, SG 1] for liquids; for gas service Kv is in mol/(s bar).
    Faults: ``stuck = True`` freezes the stem.
    """
    kind = "valve"
    io_keys = {"stem"}
    n_states = 1

    def __init__(self, name, Kv, characteristic="equal_pct", rangeability=30.0,
                 tau=4.0, cmd=None, stem0=0.5):
        super().__init__(name)
        self.Kv, self.char, self.R, self.tau = Kv, characteristic, rangeability, tau
        self.cmd, self.stem0, self.manual, self.stuck = cmd, stem0, stem0, False

    def x0(self):
        return np.array([self.stem0])

    def state_names(self):
        return [f"{self.name}.stem"]

    def f(self, u):
        if self.char == "linear":
            return u
        if self.char == "quick":
            return np.sqrt(u)
        return (self.R ** u - 1.0) / (self.R - 1.0)

    def outputs(self, t, s):
        u = float(np.clip(s[0], 0.0, 1.0))
        f = self.f(u)
        self.out.update(stem=u, f=f, Kv_eff=self.Kv * f)

    def derivatives(self, t, s):
        if self.stuck:
            return np.zeros(1)
        u = self.cmd() if self.cmd is not None else self.manual
        return np.array([(np.clip(u, 0.0, 1.0) - s[0]) / self.tau])


class CentrifugalPump(Unit):
    """dP = dP0 n^2 - k Q^2 (affinity laws) with a VFD speed ramp; ``tripped`` stops it."""
    kind = "pump"
    io_keys = {"speed"}
    n_states = 1

    def __init__(self, name, dp_shutoff, dp_rated, Q_rated, eff=0.55, speed_cmd=1.0, tau=5.0):
        super().__init__(name)
        self.dp0 = dp_shutoff
        self.k = (dp_shutoff - dp_rated) / Q_rated ** 2
        self.eff, self.tau, self.tripped = eff, tau, False
        self.speed_cmd = as_signal(speed_cmd)
        self.speed0 = speed_cmd if not callable(speed_cmd) else 1.0

    def x0(self):
        return np.array([float(self.speed0)])

    def state_names(self):
        return [f"{self.name}.speed"]

    def outputs(self, t, s):
        self.out.update(speed=float(max(s[0], 0.0)))

    def derivatives(self, t, s):
        tgt = 0.0 if self.tripped else self.speed_cmd()
        return np.array([(tgt - s[0]) / self.tau])


class VacuumPump(Unit):
    """
    Liquid-ring vacuum pump: suction capacity S = S_max n (1 - (P_ult/P)^2) [m3/h, actual].
    P_ult is set by the seal-liquid vapour pressure. Speed via VFD (signal 0-1).
    """
    kind = "vacuum pump"
    io_keys = {"speed"}
    n_states = 1

    def __init__(self, name, S_max_m3h=200.0, P_ult=0.033, speed_cmd=1.0, tau=5.0, speed0=0.8):
        super().__init__(name)
        self.S_max, self.P_ult, self.tau, self.tripped = S_max_m3h, P_ult, tau, False
        self.speed_cmd, self.speed0 = as_signal(speed_cmd), speed0

    def x0(self):
        return np.array([self.speed0])

    def state_names(self):
        return [f"{self.name}.speed"]

    def capacity(self, P):
        n = self.out["speed"]
        return self.S_max / 3600.0 * n * softplus(1.0 - (self.P_ult / max(P, 1e-4)) ** 2, 0.01)

    def outputs(self, t, s):
        self.out.update(speed=float(np.clip(s[0], 0.0, 1.2)))

    def derivatives(self, t, s):
        tgt = 0.0 if self.tripped else float(np.clip(self.speed_cmd(), 0.0, 1.2))
        return np.array([(tgt - s[0]) / self.tau])


class LiquidLine(Unit):
    """
    source -> [pump] -> control valve -> destination, solved in closed form:
      P_src + dP0 n^2 - k_p Q^2 - SG (Q/Kv f)^2 - k_line Q^2 - rho g dz = P_dest
    Non-return valve (no back-flow) and loss-of-suction ramp near ``min_level``.
    """
    kind = "liquid line"

    def __init__(self, name, source, valve, p_dest, pump=None, dz=0.0, k_line=0.0, min_level=0.05):
        super().__init__(name)
        self.source, self.valve, self.pump = source, valve, pump
        self.p_dest = as_signal(p_dest)
        self.dz, self.k_line, self.min_level = dz, k_line, min_level

    def outputs(self, t, s):
        so = self.source.out
        SG = so["SG"]
        kv = max(self.valve.out["Kv_eff"], 1e-6)
        res = SG / kv ** 2 + self.k_line
        drive = so["P_out"] - self.p_dest() - SG * 1000.0 * G * self.dz / BAR
        if self.pump is not None:
            drive += self.pump.dp0 * self.pump.out["speed"] ** 2
            res += self.pump.k
        Q = float(np.sqrt(softplus(drive) / res))
        Q *= float(_smoothstep(so.get("level", 10.0) / self.min_level - 1.0))
        F = Q / 3600.0 * so["rho_m"]
        st = {"F": F, "z": so["z"], "T": so["T"]}
        self.out.update(Q=Q, F=F, dp_valve=SG * (Q / kv) ** 2, stream=st)
        if self.pump is not None:
            dpp = self.pump.dp0 * self.pump.out["speed"] ** 2 - self.pump.k * Q ** 2
            self.out.update(dp_pump=dpp, power_kW=Q / 3600 * dpp * BAR / self.pump.eff / 1e3)


class VaporLine(Unit):
    """
    Vapour line / nozzle: m = CdA sqrt(2 rho dP) (regularised, bidirectional).
    ``upstream`` -> dict(z, T, P); ``p_down`` -> bar.
    """
    kind = "vapour line"

    def __init__(self, name, upstream, p_down, CdA):
        super().__init__(name)
        self.upstream, self.p_down, self.CdA = upstream, as_signal(p_down), CdA

    def outputs(self, t, s):
        up = self.upstream()
        rho, mw = rho_vap(up["T"], up["P"], up["z"])
        dP = (up["P"] - self.p_down()) * BAR
        m = self.CdA * np.sqrt(2.0 * rho) * dP / np.sqrt(abs(dP) + 1.0)
        F = float(m / mw)
        self.out.update(F=F, dP_mbar=dP / 100.0, stream={"F": F, "z": up["z"], "T": up["T"]})


class AirBleed(Unit):
    """Atmospheric air bleed through a gas valve (choked): n = Kv f P_atm sqrt(1-(P/P_atm)^2)."""
    kind = "air bleed"

    def __init__(self, name, valve, p_down):
        super().__init__(name)
        self.valve, self.p_down = valve, as_signal(p_down)

    def outputs(self, t, s):
        r = min(self.p_down() / P_ATM, 1.0)
        F = self.valve.out["Kv_eff"] * P_ATM * np.sqrt(max(1.0 - r * r, 0.0))
        self.out.update(F=float(F), kg_h=float(F * MW_AIR * 3.6))


# =============================================================================
# Heat exchangers
# =============================================================================
class CounterflowHX(Unit):
    """
    Liquid/liquid plate exchanger (e.g. regenerative feed/product), effectiveness-NTU
    with first-order outlet-temperature dynamics (plate/fluid thermal inertia).
    """
    kind = "heat exchanger"
    io_keys = {"Tc_out_C", "Th_out_C", "Q_kW"}
    n_states = 2

    def __init__(self, name, cold_in, hot_in, UA, tau=30.0, T0=(310.0, 290.0)):
        super().__init__(name)
        self.cold_in, self.hot_in, self.UA, self.tau, self.T0 = cold_in, hot_in, UA, tau, T0

    def x0(self):
        return np.array(self.T0, dtype=float)

    def atol(self):
        return np.full(2, 1e-5)

    def state_names(self):
        return [f"{self.name}.Tc_out", f"{self.name}.Th_out"]

    def outputs(self, t, s):
        c, h = self.cold_in(), self.hot_in()
        self.out.update(cold_out={"F": c["F"], "z": c["z"], "T": s[0]},
                        hot_out={"F": h["F"], "z": h["z"], "T": s[1]},
                        Tc_out_C=s[0] - 273.15, Th_out_C=s[1] - 273.15,
                        Q_kW=stream_heat_capacity(c) * (s[0] - c["T"]) / 1e3)

    def derivatives(self, t, s):
        c, h = self.cold_in(), self.hot_in()
        Cc, Ch = max(stream_heat_capacity(c), 1e-3), max(stream_heat_capacity(h), 1e-3)
        Cmin, Cmax = min(Cc, Ch), max(Cc, Ch)
        eps = counterflow_effectiveness(self.UA / Cmin, Cmin / Cmax)
        Q = eps * Cmin * (h["T"] - c["T"])
        return (np.array([c["T"] + Q / Cc, h["T"] - Q / Ch]) - s) / self.tau


class HotWaterHeater(Unit):
    """Trim heater: process stream heated by hot water whose flow is set by a valve."""
    kind = "heat exchanger"
    io_keys = {"T_out_C", "Q_kW"}
    n_states = 1

    def __init__(self, name, inlet, valve, UA, T_hw, m_hw_max, tau=20.0, T0=315.0):
        super().__init__(name)
        self.inlet, self.valve, self.UA, self.T_hw = inlet, valve, UA, T_hw
        self.m_hw_max, self.tau, self.T0 = m_hw_max, tau, T0

    def x0(self):
        return np.array([self.T0])

    def atol(self):
        return np.array([1e-5])

    def state_names(self):
        return [f"{self.name}.T_out"]

    def outputs(self, t, s):
        st = self.inlet()
        self.out.update(stream={"F": st["F"], "z": st["z"], "T": s[0]}, T_out_C=s[0] - 273.15,
                        Q_kW=stream_heat_capacity(st) * (s[0] - st["T"]) / 1e3)

    def derivatives(self, t, s):
        st = self.inlet()
        Cp = max(stream_heat_capacity(st), 1e-3)
        Cw = max(self.valve.out["f"] * self.m_hw_max * CP_WATER, 1e-3)
        Cmin, Cmax = min(Cp, Cw), max(Cp, Cw)
        Q = counterflow_effectiveness(self.UA / Cmin, Cmin / Cmax) * Cmin * (self.T_hw - st["T"])
        return np.array([(st["T"] + Q / Cp - s[0]) / self.tau])


class HotWaterReboiler(Unit):
    """
    Reboiler heated by hot water (boiling side isothermal):
    Q* = C_hw (1 - exp(-UA/C_hw)) (T_hw - T_boil), C_hw = f(valve) m_max cp; Q lags Q*.
    """
    kind = "reboiler"
    io_keys = {"Q_kW"}
    n_states = 1

    def __init__(self, name, valve, UA, T_hw, m_hw_max, T_process, tau=60.0, Q0=3.5e5):
        super().__init__(name)
        self.valve, self.UA, self.T_hw, self.m_hw_max = valve, UA, T_hw, m_hw_max
        self.T_process, self.tau, self.Q0 = T_process, tau, Q0

    def x0(self):
        return np.array([self.Q0])

    def atol(self):
        return np.array([1e-2])

    def state_names(self):
        return [f"{self.name}.Q"]

    def outputs(self, t, s):
        self.out.update(Q=float(max(s[0], 0.0)), Q_kW=float(max(s[0], 0.0)) / 1e3)

    def derivatives(self, t, s):
        Cw = max(self.valve.out["f"] * self.m_hw_max * CP_WATER, 1e-3)
        Qt = Cw * (1.0 - np.exp(-self.UA / Cw)) * max(self.T_hw - self.T_process(), 0.0)
        return np.array([(Qt - s[0]) / self.tau])


# =============================================================================
# Holdups
# =============================================================================
class LiquidTank(Unit):
    """
    Well-mixed liquid vessel (sump, reflux drum) with component and energy balances.
    Liquid may be sub-cooled (no phase change). States: n_EtOH, n_water, n_extract, T.
    """
    kind = "vessel"
    io_keys = {"level", "P_out", "T_C", "abv", "xE"}
    n_states = 4

    def __init__(self, name, P, area, inlets=(), outlets=(), x_init=(0.02, 0.98, 0.0),
                 T0=315.0, level0=0.5, metal_kg=300.0, UA_loss=0.0, T_amb=293.15):
        super().__init__(name)
        self.P, self.area = as_signal(P), area
        self.inlets, self.outlets = list(inlets), list(outlets)
        self.x_init, self.T0, self.level0 = np.asarray(x_init, float), T0, level0
        self.Cm, self.UA_loss, self.T_amb = metal_kg * 500.0, UA_loss, T_amb

    def x0(self):
        rho_m, _, _ = liquid_props(self.x_init)
        M = rho_m[0] * self.area * self.level0
        return np.concatenate([M * self.x_init, [self.T0]])

    def atol(self):
        return np.array([1e-5, 1e-4, 1e-6, 1e-6])

    def state_names(self):
        return [f"{self.name}.{v}" for v in ("nEtOH", "nWater", "nExtract", "T")]

    def outputs(self, t, s):
        n = np.maximum(s[:3], 1e-9)
        M = n.sum()
        x = n / M
        rho_m, rho, _ = liquid_props(x)
        level = M / rho_m[0] / self.area
        P = self.P()
        self._x, self._M = x, M
        self.out.update(level=level, P_out=P + rho[0] * G * level / BAR, z=x, T=s[3],
                        T_C=s[3] - 273.15, SG=rho[0] / 1000.0, rho_m=rho_m[0],
                        abv=abv(x)[0], xE=x[E], inv=n.copy())

    def derivatives(self, t, s):
        d = np.zeros(4)
        heat = -self.UA_loss * (s[3] - self.T_amb)
        for ref in self.inlets:
            st = ref()
            z = np.asarray(st["z"])
            d[:3] += st["F"] * z
            heat += st["F"] * float(z @ CP_L) * (st["T"] - s[3])
        Fout = sum(r() for r in self.outlets)
        d[:3] -= Fout * self._x
        C = self._M * float(self._x @ CP_L) + self.Cm
        d[3] = heat / C
        return d


class VacuumCondenser(Unit):
    """
    Vacuum condenser with non-condensable (air) accumulation and a vacuum pump on the vent.

    States: total gas moles n_g, air moles n_air, ethanol fraction of condensable gas y_c.
      P      = n_g R T_g / V
      p_v    = P (1 - y_air)       condensable partial pressure
      T_c    = bubble T of the condensate (comp. y_c) at p_v
      Q      = C_cw (1 - exp(-UA_eff/C_cw)) (T_c - T_cw),  UA_eff = UA (1 - y_air)^2  (blanketing)
      N_cond = Q / lambda;  vent = S(P) P / (R T_g) removes air + condensables (aroma/ethanol loss)
    Air sources: lumped leakage of the whole vacuum system + bleed valve.
    """
    kind = "condenser"
    io_keys = {"P_mbar", "T_c_C", "Q_kW", "N_cond", "vent_mol_s", "vent_EtOH_kg_h", "y_air"}
    n_states = 3

    def __init__(self, name, volume, UA, cw_flow, T_cw, air_leak_kgh, vapor_in, pump,
                 bleed=None, approach=5.0, P0=0.092, y_air0=0.3, yc0=0.7):
        super().__init__(name)
        self.Vol, self.UA, self.cw_flow, self.T_cw = volume, UA, cw_flow, T_cw
        self.air_leak_kgh, self.vapor_in, self.pump, self.bleed = air_leak_kgh, vapor_in, pump, bleed
        self.approach, self.P0, self.y_air0, self.yc0 = approach, P0, y_air0, yc0
        self._T = None

    @property
    def T_g(self):
        return self.T_cw + self.approach

    def x0(self):
        ng = self.P0 * BAR * self.Vol / (R_GAS * self.T_g)
        return np.array([ng, ng * self.y_air0, self.yc0])

    def atol(self):
        return np.array([1e-6, 1e-7, 1e-7])

    def state_names(self):
        return [f"{self.name}.n_gas", f"{self.name}.n_air", f"{self.name}.y_EtOH"]

    def outputs(self, t, s):
        ng, nair = max(s[0], 1e-9), min(max(s[1], 0.0), max(s[0], 1e-9))
        yc = float(np.clip(s[2], 0.0, 1.0))
        P = ng * R_GAS * self.T_g / self.Vol / BAR
        y_air = nair / ng
        p_v = max(P * (1.0 - y_air), 2e-3)
        zc = np.array([yc, 1.0 - yc, 0.0])
        Tc, _, _ = bubble_point(zc[None], p_v, self._T)
        self._T = Tc
        Tc = float(Tc[0])
        Ccw = self.cw_flow * CP_WATER
        UAe = self.UA * (1.0 - y_air) ** 2
        Q = Ccw * (1.0 - np.exp(-UAe / Ccw)) * float(softplus(Tc - self.T_cw, 0.05))
        lam = float(latent_heat(Tc) @ zc[:2])
        Nc = Q / lam
        n_pump = self.pump.capacity(P) * P * BAR / (R_GAS * self.T_g)
        vent_c = n_pump * (1.0 - y_air)
        self._d = (Nc, n_pump, y_air, ng - nair)
        self.out.update(P=P, P_mbar=P * 1000.0, T_c=Tc, T_c_C=Tc - 273.15, Q_kW=Q / 1e3,
                        y_air=y_air, N_cond=Nc, vent_mol_s=n_pump,
                        vent_EtOH_kg_h=vent_c * yc * 46.068e-3 * 3600,
                        condensate={"F": Nc, "z": zc, "T": Tc},
                        vent={"F": vent_c, "z": zc, "T": self.T_g},
                        inv=(ng - nair) * zc, pump_S_m3h=self.pump.capacity(P) * 3600)

    def derivatives(self, t, s):
        Nc, n_pump, y_air, nc = self._d
        vin = self.vapor_in()
        air = self.air_leak_kgh / 3.6 / MW_AIR + (self.bleed() if self.bleed else 0.0)
        Fv = vin["F"]
        dyc = max(Fv, 0.0) * (vin["z"][E] - s[2]) / max(nc, 1e-6)
        return np.array([Fv + air - Nc - n_pump, air - n_pump * y_air, dyc])
