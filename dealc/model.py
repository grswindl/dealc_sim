"""
Two-column vacuum de-alcoholisation flowsheet, built from a configuration dict.

  beer 4C -FV-100-> E-101 regen -> E-102 trim (TIC-102) -> C-100 STRIPPING bed (packed)
                                                             | vapour V-100
  C-100 kettle sump <- E-103 hot-water reboiler (AIC-103)    v
     |                                                     C-200 RECTIFYING bed (packed)
     +-P-101/LV-101 (LIC-101)-> E-101 -> de-alc beer          |  vapour V-200
                                                             v
  T-200 (C-200 sump) -P-203/LV-203 (LIC-203)-> C-100 top    E-201 vacuum condenser <- air leak
                                                             |   + PV-300 bleed (PIC-300)
                                                             |   -> K-300 liquid-ring pump -> vent
                                                           D-201 drum -P-202/TV-202 (TIC-202)-> reflux
                                                                     -P-201/LV-201 (LIC-201)-> concentrate
"""
import copy
import json

import numpy as np

from dynsim import (Flowsheet, Ref, FeedSource, Valve, CentrifugalPump, VacuumPump, LiquidLine,
                    VaporLine, AirBleed, CounterflowHX, HotWaterHeater, HotWaterReboiler,
                    LiquidTank, VacuumCondenser, PackedColumn, PID, Analyzer, BalanceMonitor)
from dynsim.thermo import beer_composition
from dynsim.packing import PACKINGS

K0 = 273.15

DEFAULT = {
    "feed": {"flow_m3h": 5.0, "abv": 5.0, "extract_gL": 50.0, "T_C": 4.0, "P_bar": 3.0},
    "preheat": {"regen_enabled": True, "regen_UA_kW_K": 25.0, "trim_UA_kW_K": 10.0,
                "trim_hw_T_C": 75.0, "trim_hw_max_kg_s": 1.0, "T_sp_C": 42.0},
    "C100": {"packing": "Mellapak 250Y (structured)", "diameter_m": 0.7, "height_m": 6.0,
             "n_seg": 12, "mt_factor": 1.0, "sump_area_m2": 0.8, "level_sp_m": 0.6,
             "UA_loss_W_m2K": 0.5},
    "C200": {"packing": "Mellapak 500Y (structured)", "diameter_m": 0.9, "height_m": 4.0,
             "n_seg": 8, "mt_factor": 1.0, "sump_area_m2": 0.3, "level_sp_m": 0.5,
             "UA_loss_W_m2K": 0.5},
    "reboiler": {"UA_kW_K": 30.0, "hw_T_C": 75.0, "hw_max_kg_s": 8.0, "tau_s": 60.0},
    "condenser": {"UA_kW_K": 60.0, "cw_T_C": 6.0, "cw_flow_kg_s": 15.0, "volume_m3": 2.0},
    "vacuum": {"S_max_m3h": 200.0, "P_ult_mbar": 33.0, "air_leak_kg_h": 1.5,
               "bleed_Kv": 0.06, "P_sp_mbar": 100.0, "pic_pv": "C-100 top",
               "pic_mv": "vacuum pump speed"},
    "vapour_lines": {"CdA_100_m2": 0.05, "CdA_200_m2": 0.05},
    "pumps": {
        "P-101": {"dp_shutoff": 4.5, "dp_rated": 4.0, "Q_rated": 6.0, "eff": 0.55},
        "P-201": {"dp_shutoff": 3.0, "dp_rated": 2.5, "Q_rated": 0.5, "eff": 0.40},
        "P-202": {"dp_shutoff": 3.0, "dp_rated": 2.5, "Q_rated": 2.0, "eff": 0.45},
        "P-203": {"dp_shutoff": 3.0, "dp_rated": 2.5, "Q_rated": 1.0, "eff": 0.45},
    },
    "valves": {   # Kv = 0 -> auto-sized for ~50 % opening at design flow
        "FV-100": {"Kv": 0.0, "characteristic": "equal_pct", "tau": 3.0},
        "LV-101": {"Kv": 0.0, "characteristic": "equal_pct", "tau": 4.0},
        "HV-102": {"Kv": 1.0, "characteristic": "linear", "tau": 5.0},
        "HV-103": {"Kv": 1.0, "characteristic": "linear", "tau": 8.0},
        "LV-201": {"Kv": 0.0, "characteristic": "equal_pct", "tau": 4.0},
        "TV-202": {"Kv": 0.0, "characteristic": "equal_pct", "tau": 4.0},
        "LV-203": {"Kv": 0.0, "characteristic": "equal_pct", "tau": 4.0},
        "PV-300": {"Kv": 0.0, "characteristic": "linear", "tau": 2.0},
    },
    "drum": {"area_m2": 0.3, "level_sp_m": 0.5},
    "structure": {"recycle_to": "C-100 top", "product_control": "analyser (ABV)",
                  "reflux_control": "C-200 temperature", "tic202_segment": 4,
                  "product_abv_sp": 0.45, "analyser_tau_s": 60.0, "c100_T_sp_C": 47.0, "c200_T_sp_C": 36.0,
                  "reflux_flow_sp_m3h": 1.1},
    "tuning": {
        "FIC-100": {"Kc": 0.3, "Ti": 8.0, "mode": "auto"},
        "TIC-102": {"Kc": 1.5, "Ti": 120.0, "mode": "auto"},
        "LIC-101": {"Kc": 1.5, "Ti": 600.0, "mode": "auto"},
        "AIC-103": {"Kc": 0.5, "Ti": 900.0, "mode": "auto"},
        "LIC-201": {"Kc": 1.5, "Ti": 600.0, "mode": "auto"},
        "TIC-202": {"Kc": 1.0, "Ti": 600.0, "mode": "auto"},
        "LIC-203": {"Kc": 1.5, "Ti": 600.0, "mode": "auto"},
        "PIC-300": {"Kc": 1.0, "Ti": 120.0, "mode": "auto"},
        "TIC-103": {"Kc": 1.0, "Ti": 900.0, "mode": "auto"},
        "FIC-202": {"Kc": 0.3, "Ti": 10.0, "mode": "auto"},
    },
    "solver": {"rtol": 1e-4, "atol_scale": 1.0, "jacobian": "sparse", "report_dt_s": 20.0},
    "sim": {"t_end_h": 4.0, "init_horizon_h": 6.0},
    "schedule": [
        {"t_h": 0.5, "param": "Feed ABV [%]", "value": 6.0},
        {"t_h": 1.5, "param": "Feed flow SP [m3/h]", "value": 5.75},
        {"t_h": 2.5, "param": "C-100 pressure SP [mbar]", "value": 93.0},
        {"t_h": 3.2, "param": "Air leak [kg/h]", "value": 3.0},
    ],
}


def default_config():
    return copy.deepcopy(DEFAULT)


def merge_config(loaded):
    """Deep-merge a (possibly older / partial) case onto the defaults."""
    def merge(base, new):
        for k, v in new.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                merge(base[k], v)
            else:
                base[k] = v
        return base
    return merge(default_config(), copy.deepcopy(loaded))


def structural_key(cfg):
    """Configuration items that change the initial steady state (not tuning, schedule, solver)."""
    d = {k: v for k, v in cfg.items() if k not in ("schedule", "sim", "solver", "tuning")}
    d["modes"] = {k: v.get("mode", "auto") for k, v in cfg["tuning"].items()}
    return json.dumps(d, sort_keys=True)


# controller tag -> (config section, key) holding its set point
SP_MAP = {
    "FIC-100": ("feed", "flow_m3h"), "TIC-102": ("preheat", "T_sp_C"),
    "LIC-101": ("C100", "level_sp_m"), "AIC-103": ("structure", "product_abv_sp"),
    "TIC-103": ("structure", "c100_T_sp_C"), "LIC-201": ("drum", "level_sp_m"),
    "TIC-202": ("structure", "c200_T_sp_C"), "FIC-202": ("structure", "reflux_flow_sp_m3h"),
    "LIC-203": ("C200", "level_sp_m"), "PIC-300": ("vacuum", "P_sp_mbar"),
}
LOOP_INFO = {
    "FIC-100": ("Beer feed flow [m3/h]", "FV-100 feed valve"),
    "TIC-102": ("Feed temperature to C-100 [C]", "HV-102 trim hot water"),
    "LIC-101": ("C-100 sump level [m]", "LV-101 product valve"),
    "AIC-103": ("Product ABV (analyser) [%]", "HV-103 reboiler hot water"),
    "TIC-103": ("C-100 bed-bottom temperature [C]", "HV-103 reboiler hot water"),
    "LIC-201": ("Reflux drum level [m]", "LV-201 concentrate valve"),
    "TIC-202": ("C-200 segment temperature [C]", "TV-202 reflux valve"),
    "FIC-202": ("Reflux flow [m3/h]", "TV-202 reflux valve"),
    "LIC-203": ("T-200 level [m]", "LV-203 recycle valve"),
    "PIC-300": ("Vacuum pressure [mbar]", "K-300 speed / PV-300 bleed"),
}


# Parameters that can be RESET in the schedule: label -> (unit, attribute, kind, to_internal)
PARAMS = {
    "Feed ABV [%]": ("FEED", "abv", "float", None),
    "Feed extract [g/L]": ("FEED", "extract_gL", "float", None),
    "Feed temperature [C]": ("FEED", "T", "float", lambda v: v + K0),
    "Feed flow SP [m3/h]": ("FIC-100", "sp", "float", None),
    "Preheat T SP [C]": ("TIC-102", "sp", "float", None),
    "Product ABV SP [%]": ("AIC-103", "sp", "float", None),
    "C-100 bottom T SP [C]": ("TIC-103", "sp", "float", None),
    "C-100 pressure SP [mbar]": ("PIC-300", "sp", "float", None),
    "C-200 temperature SP [C]": ("TIC-202", "sp", "float", None),
    "Reflux flow SP [m3/h]": ("FIC-202", "sp", "float", None),
    "C-100 level SP [m]": ("LIC-101", "sp", "float", None),
    "Reboiler hot-water T [C]": ("E-103", "T_hw", "float", lambda v: v + K0),
    "Trim hot-water T [C]": ("E-102", "T_hw", "float", lambda v: v + K0),
    "Cooling water T [C]": ("E-201", "T_cw", "float", lambda v: v + K0),
    "Cooling water flow [kg/s]": ("E-201", "cw_flow", "float", None),
    "Air leak [kg/h]": ("E-201", "air_leak_kgh", "float", None),
    "Vacuum pump K-300 trip (1/0)": ("K-300", "tripped", "bool", None),
    "Product pump P-101 trip (1/0)": ("P-101", "tripped", "bool", None),
    "Reflux pump P-202 trip (1/0)": ("P-202", "tripped", "bool", None),
    "Reboiler valve HV-103 stuck (1/0)": ("HV-103", "stuck", "bool", None),
    "Bleed valve PV-300 stuck (1/0)": ("PV-300", "stuck", "bool", None),
    "AIC-103 auto (1) / manual (0)": ("AIC-103", "mode", "mode", None),
    "PIC-300 auto (1) / manual (0)": ("PIC-300", "mode", "mode", None),
    "TIC-202 auto (1) / manual (0)": ("TIC-202", "mode", "mode", None),
    "AIC-103 manual output [0-1]": ("AIC-103", "op_man", "float", None),
    "PIC-300 manual output [0-1]": ("PIC-300", "op_man", "float", None),
}


def apply_schedule(fs, schedule):
    fs.clear_schedule()
    skipped = []
    for ev in schedule:
        lab = ev["param"]
        if lab not in PARAMS:
            skipped.append(lab)
            continue
        unit, attr, kind, conv = PARAMS[lab]
        if unit not in fs:
            skipped.append(f"{lab} (unit {unit} not in flowsheet)")
            continue
        v = float(ev["value"])
        t = float(ev["t_h"]) * 3600.0
        label = f"{lab} -> {v:g}"
        if kind == "mode":
            def act(f, X, u=unit, auto=v >= 0.5):
                return f[u].set_mode(f, X, "auto" if auto else "manual")
            fs.at(t, act, label, touches=[(unit, "mode"), (unit, "bias"), (unit, "op_man")])
        else:
            val = bool(v >= 0.5) if kind == "bool" else (conv(v) if conv else v)
            fs.reset(t, f"{unit}.{attr}", val, label)
    return skipped


def size_kv(Q, dp, SG=1.0, opening=0.5, R=30.0, char="equal_pct"):
    f = opening if char == "linear" else (R ** opening - 1) / (R - 1)
    return Q / (f * np.sqrt(dp / SG))


def build(cfg=None):
    cfg = default_config() if cfg is None else cfg
    fd, ph, c1, c2 = cfg["feed"], cfg["preheat"], cfg["C100"], cfg["C200"]
    rb, cd, vac, st, tu = cfg["reboiler"], cfg["condenser"], cfg["vacuum"], cfg["structure"], cfg["tuning"]
    fs = Flowsheet("Two-column vacuum de-alcoholisation")
    Qf = fd["flow_m3h"]
    Pv = vac["P_sp_mbar"] / 1e3
    recycle = st["recycle_to"] == "C-100 top"

    # ---------------------------------------------------------------- equipment
    feed = FeedSource("FEED", abv=fd["abv"], extract_gL=fd["extract_gL"], P=fd["P_bar"], T=fd["T_C"] + K0)
    vcfg, pcfg = cfg["valves"], cfg["pumps"]

    def valve(tag, auto_kv, stem0=0.5):
        v = vcfg.get(tag, {})
        kv = v.get("Kv", 0.0) or auto_kv
        return Valve(tag, Kv=kv, characteristic=v.get("characteristic", "equal_pct"),
                     tau=v.get("tau", 4.0), stem0=stem0)

    def pump(tag):
        p = pcfg[tag]
        return CentrifugalPump(tag, p["dp_shutoff"], p["dp_rated"], p["Q_rated"], eff=p.get("eff", 0.55))

    FV100 = valve("FV-100", size_kv(Qf, 1.7, 1.005, 0.55), 0.55)
    HV102 = valve("HV-102", 1.0, 0.3)
    LV101 = valve("LV-101", size_kv(0.95 * Qf, 2.1, 1.006))
    HV103 = valve("HV-103", 1.0, 0.5)
    LV201 = valve("LV-201", size_kv(0.25, 2.8, 0.82))
    TV202 = valve("TV-202", size_kv(1.1, 2.8, 0.82))
    LV203 = valve("LV-203", size_kv(0.5, 2.0, 0.99))
    PV300 = valve("PV-300", vac["bleed_Kv"], 0.4)
    P101, P201, P202, P203 = pump("P-101"), pump("P-201"), pump("P-202"), pump("P-203")
    K300 = VacuumPump("K-300", S_max_m3h=vac["S_max_m3h"], P_ult=vac["P_ult_mbar"] / 1e3, speed_cmd=1.0)

    xb = beer_composition(fd["abv"], fd["extract_gL"])
    n1 = c1["n_seg"] + 1
    xE1 = np.geomspace(xb[0] * 1.05, xb[0] * 0.09, n1)
    x1 = np.column_stack([xE1, 1 - xE1 - xb[2], np.full(n1, xb[2])])
    C100 = PackedColumn("C-100", c1["packing"], c1["diameter_m"], c1["height_m"], c1["n_seg"],
                        bottom="reboiler", P_top0=Pv, dP0=0.008, sump_area=c1["sump_area_m2"],
                        sump_level0=c1["level_sp_m"], x_init=x1, T_init=318.0,
                        L_nom_m3h=1.1 * Qf, yv0=0.14, mt_factor=c1["mt_factor"],
                        UA_loss=c1["UA_loss_W_m2K"])
    n2 = c2["n_seg"]
    xE2 = np.geomspace(0.72, 0.016, n2)
    x2 = np.column_stack([xE2, 1 - xE2, np.zeros(n2)])
    C200 = PackedColumn("C-200", c2["packing"], c2["diameter_m"], c2["height_m"], c2["n_seg"],
                        bottom="vapor_feed", P_top0=Pv - 0.006, dP0=0.005, x_init=x2, T_init=310.0,
                        L_nom_m3h=1.0, yv0=0.72, mt_factor=c2["mt_factor"],
                        UA_loss=c2["UA_loss_W_m2K"])
    T200 = LiquidTank("T-200", P=Ref(C200, "P_bottom"), area=c2["sump_area_m2"],
                      x_init=(0.016, 0.984, 0.0), T0=316.0, level0=c2["level_sp_m"], metal_kg=300.0)
    E201 = VacuumCondenser("E-201", volume=cd["volume_m3"], UA=cd["UA_kW_K"] * 1e3,
                           cw_flow=cd["cw_flow_kg_s"], T_cw=cd["cw_T_C"] + K0,
                           air_leak_kgh=vac["air_leak_kg_h"], vapor_in=None, pump=K300,
                           P0=Pv - 0.008, y_air0=0.3, yc0=0.72)
    D201 = LiquidTank("D-201", P=Ref(E201, "P"), area=cfg["drum"]["area_m2"], x_init=(0.72, 0.28, 0.0),
                      T0=293.0, level0=cfg["drum"]["level_sp_m"], metal_kg=300.0)
    E103 = HotWaterReboiler("E-103", HV103, UA=rb["UA_kW_K"] * 1e3, T_hw=rb["hw_T_C"] + K0,
                            m_hw_max=rb["hw_max_kg_s"], T_process=Ref(C100, "T_sump"),
                            tau=rb["tau_s"], Q0=3.6e5)
    C100.duty = Ref(E103, "Q")

    # ---------------------------------------------------------- lines and HX
    L_feed = LiquidLine("L-feed", feed, FV100, p_dest=Pv, dz=12.0)
    L_prod = LiquidLine("L-product", C100, LV101, p_dest=2.5, pump=P101, dz=-3.0)
    L_rec = LiquidLine("L-recycle", T200, LV203, p_dest=Pv if recycle else 1.0, pump=P203,
                       dz=9.0 if recycle else -2.0)
    L_refl = LiquidLine("L-reflux", D201, TV202, p_dest=Pv, pump=P202, dz=1.0)
    L_dist = LiquidLine("L-distillate", D201, LV201, p_dest=1.5, pump=P201, dz=-14.0)
    V100 = VaporLine("V-100", Ref(C100, "vapor"), Ref(C200, "P_bottom"), cfg["vapour_lines"]["CdA_100_m2"])
    V200 = VaporLine("V-200", Ref(C200, "vapor"), Ref(E201, "P"), cfg["vapour_lines"]["CdA_200_m2"])
    bleed = AirBleed("AB-300", PV300, Ref(E201, "P"))
    units_hx = []
    if ph["regen_enabled"]:
        E101 = CounterflowHX("E-101", Ref(L_feed, "stream"), Ref(L_prod, "stream"),
                             UA=ph["regen_UA_kW_K"] * 1e3, T0=(311.0, 284.0))
        trim_in, product_out = Ref(E101, "cold_out"), Ref(E101, "hot_out")
        units_hx.append(E101)
    else:
        trim_in, product_out = Ref(L_feed, "stream"), Ref(L_prod, "stream")
    E102 = HotWaterHeater("E-102", trim_in, HV102, UA=ph["trim_UA_kW_K"] * 1e3,
                          T_hw=ph["trim_hw_T_C"] + K0, m_hw_max=ph["trim_hw_max_kg_s"],
                          T0=ph["T_sp_C"] + K0)
    units_hx.append(E102)

    # wiring
    C100.add_feed(1, Ref(E102, "stream"))
    if recycle:
        C100.add_feed(1, Ref(L_rec, "stream"))
    C100.bottoms = Ref(L_prod, "F")
    C100.vapor_exit = Ref(V100, "F")
    C200.vapor_in = Ref(V100, "stream")
    C200.vapor_exit = Ref(V200, "F")
    C200.add_feed(1, Ref(L_refl, "stream"))
    T200.inlets = [Ref(C200, "liquid_out")]
    T200.outlets = [Ref(L_rec, "F")]
    E201.vapor_in = Ref(V200, "stream")
    E201.bleed = Ref(bleed, "F")
    D201.inlets = [Ref(E201, "condensate")]
    D201.outlets = [Ref(L_refl, "F"), Ref(L_dist, "F")]

    # --------------------------------------------------------- control system
    def pid(tag, pv, sp, action, span, bias, units, Td=0.0):
        t = tu.get(tag, {"Kc": 1.0, "Ti": 600.0, "mode": "auto"})
        return PID(tag, pv, sp=sp, Kc=t["Kc"], Ti=t["Ti"], Td=Td, action=action, span=span,
                   bias=bias, mode=t.get("mode", "auto"), units=units)

    ctl = []
    FIC100 = pid("FIC-100", Ref(L_feed, "Q"), Qf, "reverse", 10.0, 0.55, "m3/h")
    TIC102 = pid("TIC-102", Ref(E102, "T_out_C"), ph["T_sp_C"], "reverse", 50.0, 0.3, "C")
    LIC101 = pid("LIC-101", Ref(C100, "level"), c1["level_sp_m"], "direct", 1.2, 0.5, "m")
    ctl += [FIC100, TIC102, LIC101]
    FV100.cmd, HV102.cmd, LV101.cmd = Ref(FIC100, "op"), Ref(TIC102, "op"), Ref(LIC101, "op")
    extra = []
    if st["product_control"].startswith("analyser"):
        AT103 = Analyzer("AT-103", Ref(C100, "abv_B"), tau=st.get("analyser_tau_s", 60.0), n_lags=2, init=st["product_abv_sp"])
        AIC103 = pid("AIC-103", Ref(AT103, "value"), st["product_abv_sp"], "direct", 1.0, 0.5, "% ABV")
        extra.append(AT103)
        ctl.append(AIC103)
        HV103.cmd = Ref(AIC103, "op")
    else:
        seg = c1["n_seg"]
        tic = pid("TIC-103", Ref(C100, f"T{seg}"), st["c100_T_sp_C"], "reverse", 20.0, 0.5, "C")
        tic.name = "TIC-103"
        ctl.append(tic)
        HV103.cmd = Ref(tic, "op")
    LIC201 = pid("LIC-201", Ref(D201, "level"), cfg["drum"]["level_sp_m"], "direct", 1.0, 0.5, "m")
    LIC203 = pid("LIC-203", Ref(T200, "level"), c2["level_sp_m"], "direct", 1.0, 0.5, "m")
    LV201.cmd, LV203.cmd = Ref(LIC201, "op"), Ref(LIC203, "op")
    if st["reflux_control"] == "C-200 temperature":
        segT = int(min(max(st["tic202_segment"], 1), c2["n_seg"]))
        rc = pid("TIC-202", Ref(C200, f"T{segT}"), st["c200_T_sp_C"], "direct", 20.0, 0.5, "C")
    else:
        rc = pid("FIC-202", Ref(L_refl, "Q"), st["reflux_flow_sp_m3h"], "reverse", 3.0, 0.5, "m3/h")
        rc.name = "FIC-202"
    TV202.cmd = Ref(rc, "op")
    ctl += [LIC201, rc, LIC203]
    pv_map = {"C-100 top": Ref(C100, "P_top_mbar"), "C-200 top": Ref(C200, "P_top_mbar"),
              "condenser": Ref(E201, "P_mbar")}
    if vac["pic_mv"] == "air bleed valve":
        PIC = pid("PIC-300", pv_map[vac["pic_pv"]], vac["P_sp_mbar"], "reverse", 50.0, 0.4, "mbar")
        PV300.cmd = Ref(PIC, "op")
    else:
        PIC = pid("PIC-300", pv_map[vac["pic_pv"]], vac["P_sp_mbar"], "direct", 50.0, 0.8, "mbar")
        K300.speed_cmd = Ref(PIC, "op")
        PV300.stem0 = 0.0
        PV300.manual = 0.0
    ctl.append(PIC)

    outflows = [product_out, Ref(L_dist, "stream"), Ref(E201, "vent")]
    if not recycle:
        outflows.append(Ref(L_rec, "stream"))
    mon = BalanceMonitor("BAL", inflows=[Ref(L_feed, "stream")], outflows=outflows,
                         holdups=[C100, C200, T200, E201, D201])

    # causal order
    fs.add(feed, FV100, HV102, LV101, HV103, LV201, TV202, LV203, PV300,
           P101, P201, P202, P203, K300, E103,
           C100, C200, T200, E201, D201,
           L_feed, L_prod, L_rec, L_refl, L_dist, V100, V200, bleed, *units_hx,
           *extra, *ctl, mon)
    fs.meta = dict(cfg=cfg, controllers=[c.name for c in ctl], recycle=recycle,
                   product_tag="AIC-103" if "AIC-103" in [c.name for c in ctl] else "TIC-103",
                   reflux_tag=rc.name, product_out=product_out)
    return fs


def kpis(fs):
    """Headline numbers at the current state."""
    o = lambda u, k: fs[u].out[k]
    prod = fs.meta["product_out"]()
    return {
        "Product ABV [%]": o("C-100", "abv_B"),
        "Product flow [m3/h]": o("L-product", "Q"),
        "Product T out [C]": prod["T"] - K0,
        "Concentrate ABV [%]": o("D-201", "abv"),
        "C-100 top P [mbar]": o("C-100", "P_top_mbar"),
        "C-100 bed dP [mbar]": o("C-100", "dP_bed_mbar"),
        "Reboiler duty [kW]": o("E-103", "Q_kW"),
        "Condenser duty [kW]": o("E-201", "Q_kW"),
        "Sump T [C]": o("C-100", "T_C"),
        "Ethanol vent loss [kg/h]": o("E-201", "vent_EtOH_kg_h"),
        "C-100 max flood [%]": o("C-100", "flood_max"),
        "C-200 max flood [%]": o("C-200", "flood_max"),
    }
