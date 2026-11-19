"""
Streamlit front end for the two-column vacuum beer de-alcoholisation simulator.

    streamlit run app.py
"""
import copy
import io
import json
import time

import numpy as np
import pandas as pd
import streamlit as st

import os

from dealc.model import (build, default_config, merge_config, apply_schedule, structural_key, kpis,
                         PARAMS, SP_MAP, LOOP_INFO)
from dealc.plots import process_figure, profile_figure, solver_figure
from dealc.pid import pid_svg, unit_info, FORMS, LINE_FORM, get_path, set_path, CSS as PID_CSS, JS as PID_JS
from dealc.trust import (MODELS, REFS, GROUPS, models_for_unit, packing_table, run_checks,
                         markdown_document, _eq_prefix)
from dealc.export import SCOPES, build_export, classify, unit_catalogue, case_to_json, case_from_json
from dynsim.packing import PACKINGS

st.set_page_config(page_title="De-alc dynamic simulator", layout="wide", page_icon="🍺")

# --------------------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("cfg", default_config())
ss.setdefault("results", None)
ss.setdefault("ref", None)
ss.setdefault("page", "Flowsheet")
ss.setdefault("upload_id", None)
cfg = ss.cfg


@st.cache_resource
def init_cache():
    return {}


@st.cache_resource
def pid_component():
    return st.components.v2.component("dealc_pid", css=PID_CSS, js=PID_JS)


def flash(msg, kind="success"):
    ss.setdefault("flash", []).append((kind, msg))


def apply_case(text, source):
    cfg_new, steady = case_from_json(text)
    cfg_new = merge_config(cfg_new)
    msg = f"Case loaded from {source}."
    if steady is not None:
        key = structural_key(cfg_new)
        if steady.get("key") == key:
            hit = {"X": steady["X"], "n": steady["n"], "r": steady.get("r", float("nan")),
                   "ops": steady["ops"], "key": key}
            init_cache()[key] = hit
            init_cache()[("last", steady["n"])] = hit
            msg += " Embedded steady state restored (no re-initialisation needed)."
        else:
            msg += " Embedded steady state does not match this model version – it will be re-initialised."
    ss.cfg = cfg_new
    for k in [k for k in ss if isinstance(k, str) and (k.startswith("dlg.") or "." in k and k.split(".")[0] in cfg_new)]:
        del ss[k]
    flash(msg)


def initial_state(fs, cfg, status=None):
    """Steady state for this configuration (cached; warm-started from the last one)."""
    cache = init_cache()
    key = structural_key(cfg)
    hit = cache.get(key)
    if hit is None or hit["n"] != fs.n:
        warm = cache.get(("last", fs.n))
        prog = status.progress(0.0, text="Initialising steady state…") if status else None

        def cb(i, frac):
            if prog:
                prog.progress(min((i + frac) / 3.0, 1.0), text=f"Initialising steady state (round {i+1})…")
        t0 = time.time()
        X, r = fs.initialize(X0=None if warm is None else warm["X"].copy(), progress=cb,
                             verbose=False, rtol=1e-4)
        if prog:
            prog.empty()
        hit = {"X": X, "n": fs.n, "r": r, "wall": time.time() - t0, "key": key,
               "ops": {c: fs[c].out["op"] for c in fs.meta["controllers"]}}
        cache[key] = hit
        cache[("last", fs.n)] = hit
    X = hit["X"].copy()
    for tag, op in hit["ops"].items():        # re-seat integrators for the current tuning
        pid = fs[tag]
        if pid.mode == "auto":
            X[pid.sl.start] = (op - pid.bias) * pid.Ti / max(pid.Kc, 1e-9)
    fs.evaluate(0.0, X)
    return X, hit


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🍺 De-alc simulator")
    st.caption("Two-column vacuum distillation · packed beds · full energy balance · "
               "pressure & vacuum dynamics")
    sim = cfg["sim"]
    sol = cfg["solver"]
    sim["t_end_h"] = st.number_input("Simulation horizon [h]", 0.1, 48.0, float(sim["t_end_h"]), 0.5)
    sol["report_dt_s"] = st.number_input("Report interval [s]", 2.0, 600.0, float(sol["report_dt_s"]), 5.0)
    sol["rtol"] = st.select_slider("Relative tolerance (rtol)", [1e-3, 1e-4, 1e-5, 1e-6],
                                   value=sol["rtol"], format_func=lambda v: f"{v:.0e}")
    sol["jacobian"] = st.radio("Jacobian", ["dense", "sparse"], index=0 if sol["jacobian"] == "dense" else 1,
                               horizontal=True, help="'sparse' detects the Jacobian structure numerically "
                               "and uses grouped finite differences.")
    refresh = st.slider("Live refresh [s wall]", 0.2, 3.0, 0.6, 0.1)
    c1, c2 = st.columns(2)
    if c1.button("▶ Run", type="primary", width="stretch"):
        ss.run_request = True
        ss.page = "Simulation"
    if c2.button("Initialise", width="stretch"):
        ss.init_request = True
        ss.page = "Simulation"
    c1, c2 = st.columns(2)
    if c1.button("Keep as ref.", width="stretch", disabled=ss.results is None,
                 help="Overlay the current results as dotted reference lines in later runs"):
        ss.ref = ss.results["df"].copy()
    if c2.button("Clear ref.", width="stretch", disabled=ss.ref is None):
        ss.ref = None
    st.divider()
    st.subheader("Case file (.json)")
    st.caption("All inputs: flowsheet structure, equipment, packing, pumps/valves, control, schedule, solver.")
    hit_now = init_cache().get(structural_key(cfg))
    inc_ss = st.checkbox("Embed steady-state initial condition", value=True,
                         help="Stores the initialised plant state so a reloaded case starts immediately.",
                         disabled=hit_now is None)
    summ = None
    if ss.results is not None and len(ss.results["df"]):
        last = ss.results["df"].iloc[-1]
        summ = {k: float(last[k]) for k in ("C-100.abv_B", "D-201.abv", "C-100.P_top_mbar", "E-103.Q_kW") if k in last}
    case_text = case_to_json(cfg, steady=hit_now if (inc_ss and hit_now) else None, results_summary=summ)
    st.download_button("⬇ Download case", case_text, "dealc_case.json", "application/json", width="stretch")
    up = st.file_uploader("⬆ Upload case to restore", type="json")
    if up is not None and up.file_id != ss.upload_id:
        ss.upload_id = up.file_id
        try:
            apply_case(up.getvalue().decode("utf-8"), up.name)
        except Exception as exc:
            flash(f"Could not load case: {exc}", "error")
        st.rerun()
    case_path = st.text_input("Case file on this machine", ss.get("case_path", "cases/dealc_case.json"))
    ss.case_path = case_path
    c1, c2 = st.columns(2)
    if c1.button("💾 Save", width="stretch", help="Write the case JSON to the path above"):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(case_path)), exist_ok=True)
            with open(case_path, "w") as fh:
                fh.write(case_text)
            flash(f"Case saved to {os.path.abspath(case_path)}")
        except Exception as exc:
            flash(f"Save failed: {exc}", "error")
        st.rerun()
    if c2.button("📂 Load", width="stretch", help="Restore the case JSON from the path above"):
        try:
            with open(case_path) as fh:
                apply_case(fh.read(), os.path.abspath(case_path))
        except Exception as exc:
            flash(f"Load failed: {exc}", "error")
        st.rerun()
    if st.button("Reset to defaults", width="stretch"):
        ss.cfg = default_config()
        st.rerun()

for kind, msg in ss.pop("flash", []):
    getattr(st, kind)(msg)
cfg = ss.cfg

page = st.segmented_control("Page", ["Flowsheet", "Equipment", "Control", "Schedule", "Simulation", "Export",
                                     "Trust Center"],
                            key="page", label_visibility="collapsed")


def num(sec, key, label, lo, hi, step, fmt=None, help=None, container=st):
    d = cfg[sec]
    d[key] = container.number_input(label, float(lo), float(hi), float(d[key]), float(step),
                                    format=fmt, help=help, key=f"{sec}.{key}")
    return d[key]


# =========================================================================== pages
def render_model(key, level=3, show_meta=True):
    m = MODELS[key]
    st.markdown(f"{'#' * level} {m['title']}")
    if show_meta:
        st.caption(f"Implementation: {m['code']}")
    st.markdown(m["purpose"])
    pre = _eq_prefix(key)
    for i, (head, tex, note) in enumerate(m["equations"], 1):
        st.markdown(f"**({pre}-{i}) {head}**")
        st.latex(tex)
        if note:
            st.caption(note)
    if m["nomenclature"]:
        st.markdown("| Symbol | Meaning | Unit |\n|---|---|---|\n"
                    + "\n".join(f"| {a} | {b} | {c} |" for a, b, c in m["nomenclature"]))
    if m["assumptions"]:
        st.markdown("**Assumptions**\n" + "\n".join(f"- {a}" for a in m["assumptions"]))
    if m["refs"] and show_meta:
        st.caption("References: " + " · ".join(REFS[r] for r in m["refs"]))


def unit_parameter_table(tag):
    ftag = LINE_FORM.get(tag, tag)
    rows = []
    if ftag in FORMS:
        for path, label, *_ in FORMS[ftag][1]:
            rows.append({"Parameter": label, "Value": str(get_path(cfg, path))})
    if tag in ("C-100", "C-200"):
        sec = "C100" if tag == "C-100" else "C200"
        rows += [{"Parameter": f"Packing constant {k}", "Value": str(v)} for k, v in packing_table(cfg, sec)]
    return pd.DataFrame(rows)


def render_unit_equations(tag, compact=False):
    keys = models_for_unit(tag, cfg)
    if not keys:
        st.info(f"No model equations are associated with {tag}.")
        return
    st.markdown(f"Models governing **{tag}**: " + ", ".join(MODELS[k]["title"] for k in keys))
    pt = unit_parameter_table(tag)
    if len(pt):
        with st.expander("Parameter values used in this case", expanded=not compact):
            st.dataframe(pt, hide_index=True, width="stretch")
    if compact:
        tabs = st.tabs([MODELS[k]["title"].split(" – ")[0].split(" (")[0] for k in keys])
        for t, k in zip(tabs, keys):
            with t:
                render_model(k, level=4)
    else:
        for k in keys:
            render_model(k, level=3)
            st.divider()


def _clear_dialog_keys():
    for k in [k for k in ss if isinstance(k, str) and k.startswith("dlg.")]:
        del ss[k]


@st.dialog("Unit specification", width="large")
def edit_unit(tag):
    ftag = LINE_FORM.get(tag, tag)
    if ftag not in FORMS:
        st.info(f"{tag} has no editable parameters.")
        return
    title, fields = FORMS[ftag]
    st.markdown(f"#### {title}")
    if tag != ftag:
        st.caption(f"{tag} hydraulics are set by its control valve {ftag}.")
    new = {}
    cols = st.columns(2)
    for i, (path, label, kind, a, b, c, hlp) in enumerate(fields):
        cur = get_path(cfg, path)
        k = "dlg." + ".".join(path)
        with cols[i % 2]:
            if kind == "num":
                new[path] = st.number_input(label, float(a), float(b), float(cur), float(c), help=hlp, key=k)
            elif kind == "int":
                new[path] = int(st.number_input(label, int(a), int(b), int(cur), int(c), help=hlp, key=k))
            elif kind == "select":
                new[path] = st.selectbox(label, a, index=a.index(cur) if cur in a else 0, help=hlp, key=k)
            elif kind == "bool":
                new[path] = st.toggle(label, bool(cur), help=hlp, key=k)
    changed = {p: v for p, v in new.items() if v != get_path(cfg, p)}
    if changed:
        st.caption("Changes: " + ", ".join(f"{'.'.join(p[1:]) or p[0]} → {v}" for p, v in changed.items()))
    b1, b2 = st.columns(2)
    if b1.button("Apply", type="primary", width="stretch"):
        for p, v in changed.items():
            set_path(cfg, p, v)
        _clear_dialog_keys()
        if changed:
            flash(f"{ftag}: {len(changed)} parameter(s) updated.")
        st.rerun()
    if b2.button("Cancel", width="stretch"):
        _clear_dialog_keys()
        st.rerun()
    st.divider()
    if st.toggle("📐 Show model equations for this unit", key="dlg.show_eq",
                 help="The governing equations, parameter values and assumptions (also in the Trust Center tab)."):
        render_unit_equations(tag, compact=True)


if page == "Flowsheet":
    st.header("Process & instrumentation diagram")
    st.caption("Hover over any equipment, line or instrument for design data and the latest simulated values. "
               "**Double-click** to edit its specification.")
    row = None
    if ss.results is not None and len(ss.results["df"]) and \
            structural_key(ss.results["cfg"]) == structural_key(cfg):
        row = ss.results["df"].iloc[-1].to_dict()
    fs_view = build(copy.deepcopy(cfg))
    res_pid = pid_component()(data={"svg": pid_svg(cfg, row), "info": unit_info(cfg, fs_view, row)},
                              key="pid", on_edit_change=lambda: None)
    tag = getattr(res_pid, "edit", None)
    if tag:
        _clear_dialog_keys()
        edit_unit(tag)
    with st.expander("Flowsheet structure options", expanded=False):
        s = cfg["structure"]
        v = cfg["vacuum"]
        c1, c2, c3, c4 = st.columns(4)
        s["recycle_to"] = c1.selectbox("C-200 bottoms (weak water)", ["C-100 top", "drain"],
                                       index=["C-100 top", "drain"].index(s["recycle_to"]))
        cfg["preheat"]["regen_enabled"] = c1.toggle("Regenerative HX E-101", cfg["preheat"]["regen_enabled"])
        opts = ["analyser (ABV)", "C-100 bed-bottom temperature"]
        s["product_control"] = c2.selectbox("Product quality → reboiler", opts, index=opts.index(s["product_control"]))
        opts = ["C-200 temperature", "reflux flow"]
        s["reflux_control"] = c2.selectbox("Reflux control", opts, index=opts.index(s["reflux_control"]))
        if s["reflux_control"] == "C-200 temperature":
            s["tic202_segment"] = int(c3.number_input("TIC-202 sensor segment", 1, int(cfg["C200"]["n_seg"]),
                                                      int(min(s["tic202_segment"], cfg["C200"]["n_seg"]))))
        opts = ["C-100 top", "C-200 top", "condenser"]
        v["pic_pv"] = c3.selectbox("PIC-300 measurement", opts, index=opts.index(v["pic_pv"]))
        opts = ["vacuum pump speed", "air bleed valve"]
        v["pic_mv"] = c4.selectbox("PIC-300 manipulates", opts, index=opts.index(v["pic_mv"]),
                                   help="Bleed control is simple but vents more ethanol; VFD speed control "
                                   "minimises vent losses.")
        st.caption("Structural or design changes trigger a new steady-state initialisation (≈10 s, cached).")

elif page == "Equipment":
    st.header("Equipment & physical parameters")
    t1, t2, t3, t5, t4 = st.tabs(["Feed & preheat", "Packed columns", "Reboiler / condenser / vacuum",
                                  "Pumps & valves", "Packing library"])
    with t5:
        st.subheader("Centrifugal pumps")
        pdf = pd.DataFrame(cfg["pumps"]).T.rename_axis("tag").reset_index()
        pe = st.data_editor(pdf, hide_index=True, disabled=["tag"], key="pump_editor", width="stretch")
        for r in pe.itertuples():
            cfg["pumps"][r.tag] = {"dp_shutoff": float(r.dp_shutoff), "dp_rated": float(r.dp_rated),
                                   "Q_rated": float(r.Q_rated), "eff": float(r.eff)}
        st.subheader("Control valves")
        vdf = pd.DataFrame(cfg["valves"]).T.rename_axis("tag").reset_index()
        ve = st.data_editor(vdf, hide_index=True, disabled=["tag"], key="valve_editor", width="stretch",
                            column_config={"characteristic": st.column_config.SelectboxColumn(
                                options=["equal_pct", "linear", "quick"]),
                                "Kv": st.column_config.NumberColumn(help="0 = auto-size for ~50 % opening")})
        for r in ve.itertuples():
            cfg["valves"][r.tag] = {"Kv": float(r.Kv), "characteristic": r.characteristic, "tau": float(r.tau)}
        st.caption("Pump curve: ΔP = ΔP₀·n² − k·Q² with k from the rated point. HV-102/HV-103 are hot-water "
                   "valves – only the characteristic and stroke time matter (flow scales with f(stem)). "
                   "PV-300 Kv = 0 uses the bleed capacity set on the vacuum tab.")
    with t1:
        a, b = st.columns(2)
        with a:
            st.subheader("Beer feed")
            num("feed", "flow_m3h", "Feed flow SP [m³/h]", 0.5, 20, 0.25)
            num("feed", "abv", "Feed ABV [%]", 0.5, 15, 0.1)
            num("feed", "extract_gL", "Extract [g/L]", 0, 150, 5)
            num("feed", "T_C", "Feed temperature [°C]", 0, 40, 1)
            num("feed", "P_bar", "Supply pressure [bar a]", 1.5, 8, 0.1)
        with b:
            st.subheader("Preheat train")
            num("preheat", "regen_UA_kW_K", "E-101 regen UA [kW/K]", 1, 200, 1)
            num("preheat", "trim_UA_kW_K", "E-102 trim UA [kW/K]", 1, 100, 1)
            num("preheat", "trim_hw_T_C", "Trim hot-water T [°C]", 50, 95, 1)
            num("preheat", "trim_hw_max_kg_s", "Trim hot-water max flow [kg/s]", 0.1, 10, 0.1)
            num("preheat", "T_sp_C", "Column feed T SP (TIC-102) [°C]", 20, 60, 0.5)
    with t2:
        cols = st.columns(2)
        for col, tag, title in ((cols[0], "C100", "C-100 stripping column"), (cols[1], "C200", "C-200 rectifying column")):
            with col:
                st.subheader(title)
                d = cfg[tag]
                names = list(PACKINGS)
                d["packing"] = st.selectbox("Packing", names, index=names.index(d["packing"]), key=f"{tag}.pk")
                num(tag, "diameter_m", "Diameter [m]", 0.2, 3.0, 0.05, container=col)
                num(tag, "height_m", "Packed height [m]", 0.5, 25, 0.5, container=col)
                d["n_seg"] = int(st.number_input("Segments (discretisation)", 2, 60, int(d["n_seg"]), 1, key=f"{tag}.nseg"))
                num(tag, "mt_factor", "Mass-transfer multiplier (calibration)", 0.2, 3.0, 0.05, container=col,
                    help="Scales k_G and k_L – fit to plant HETP data.")
                num(tag, "sump_area_m2", "Sump area [m²]" if tag == "C100" else "T-200 sump area [m²]",
                    0.05, 5, 0.05, container=col)
                num(tag, "level_sp_m", "Level SP [m]", 0.1, 2, 0.05, container=col)
                num(tag, "UA_loss_W_m2K", "Insulated-shell U [W/m²K]", 0.0, 10, 0.1, container=col)
                st.caption(f"Segment height {d['height_m']/d['n_seg']:.2f} m")
        with st.expander("Vapour lines & reflux drum"):
            a, b, c = st.columns(3)
            num("vapour_lines", "CdA_100_m2", "V-100 C-100→C-200 CdA [m²]", 0.005, 0.5, 0.005, "%.3f", container=a)
            num("vapour_lines", "CdA_200_m2", "V-200 C-200→E-201 CdA [m²]", 0.005, 0.5, 0.005, "%.3f", container=b)
            num("drum", "area_m2", "D-201 area [m²]", 0.05, 3, 0.05, container=c)
    with t3:
        a, b, c = st.columns(3)
        with a:
            st.subheader("E-103 reboiler (hot water)")
            num("reboiler", "UA_kW_K", "UA [kW/K]", 5, 200, 1)
            num("reboiler", "hw_T_C", "Hot-water supply T [°C]", 55, 95, 1)
            num("reboiler", "hw_max_kg_s", "Max hot-water flow [kg/s]", 1, 40, 0.5)
            num("reboiler", "tau_s", "Thermal lag [s]", 5, 600, 5)
        with b:
            st.subheader("E-201 vacuum condenser")
            num("condenser", "UA_kW_K", "UA (air-free) [kW/K]", 5, 300, 1)
            num("condenser", "cw_T_C", "Coolant inlet T [°C]", -10, 30, 0.5)
            num("condenser", "cw_flow_kg_s", "Coolant flow [kg/s]", 1, 100, 1)
            num("condenser", "volume_m3", "Gas-side volume [m³]", 0.2, 20, 0.1)
        with c:
            st.subheader("K-300 vacuum system")
            num("vacuum", "S_max_m3h", "Pump capacity at full speed [m³/h]", 20, 2000, 10)
            num("vacuum", "P_ult_mbar", "Ultimate pressure [mbar]", 5, 80, 1)
            num("vacuum", "air_leak_kg_h", "Air in-leakage [kg/h]", 0, 30, 0.1)
            num("vacuum", "bleed_Kv", "Bleed valve capacity [mol/(s·bar)]", 0.005, 1, 0.005, "%.3f")
            num("vacuum", "P_sp_mbar", "Pressure SP (PIC-300) [mbar]", 30, 300, 1)
    with t4:
        df = pd.DataFrame(PACKINGS).T
        st.dataframe(df, width="stretch")
        st.caption("Hydraulics: Billet–Schultes holdup and pressure drop. Mass transfer: Rocha–Bravo–Fair "
                   "+ Tsai effective area (structured), Onda (random). Constants are representative – "
                   "replace with supplier data or calibrate with the mass-transfer multiplier.")

elif page == "Control":
    st.header("Control loops")
    fs_tmp = build(copy.deepcopy(cfg))
    tags = fs_tmp.meta["controllers"]
    rows = []
    for t in tags:
        sec, key = SP_MAP[t]
        tu = cfg["tuning"].setdefault(t, {"Kc": 1.0, "Ti": 600.0, "mode": "auto"})
        pid = fs_tmp[t]
        rows.append({"Tag": t, "PV": LOOP_INFO[t][0], "MV": LOOP_INFO[t][1], "Action": pid.action,
                     "SP": float(cfg[sec][key]), "Kc [-]": float(tu["Kc"]), "Ti [s]": float(tu["Ti"]),
                     "Auto": tu.get("mode", "auto") == "auto"})
    ed = st.data_editor(pd.DataFrame(rows), hide_index=True, width="stretch",
                        disabled=["Tag", "PV", "MV", "Action"], key="ctl_editor",
                        column_config={"Kc [-]": st.column_config.NumberColumn(min_value=0.0, format="%.3f"),
                                       "Ti [s]": st.column_config.NumberColumn(min_value=0.1, format="%.1f")})
    for _, r in ed.iterrows():
        sec, key = SP_MAP[r["Tag"]]
        cfg[sec][key] = float(r["SP"])
        cfg["tuning"][r["Tag"]].update(Kc=float(r["Kc [-]"]), Ti=float(r["Ti [s]"]),
                                       mode="auto" if r["Auto"] else "manual")
    st.caption("Gains are dimensionless on a span basis (error/span → valve fraction). "
               "Tuning changes don't require re-initialisation; integrators are re-seated bumplessly. "
               "Set-point and mode changes here define the *initial* steady state – use the Schedule "
               "for changes during a run.")

elif page == "Schedule":
    st.header("Schedule (disturbances, set-point changes, faults)")
    sched = pd.DataFrame(cfg["schedule"] or [{"t_h": 0.5, "param": "Feed ABV [%]", "value": 6.0}])
    ed = st.data_editor(sched, num_rows="dynamic", width="stretch", key="sched_editor",
                        column_config={
                            "t_h": st.column_config.NumberColumn("Time [h]", min_value=0.0, step=0.05),
                            "param": st.column_config.SelectboxColumn("Parameter", options=list(PARAMS), width="large"),
                            "value": st.column_config.NumberColumn("New value")})
    ed = ed.dropna(subset=["t_h", "param", "value"])
    cfg["schedule"] = [{"t_h": float(r.t_h), "param": r.param, "value": float(r.value)} for r in ed.itertuples()]
    st.caption("Values are in the units shown in the parameter name. Flags (trip/stuck) use 1/0; "
               "controller modes use 1 = auto, 0 = manual (bumpless). Every run starts from the "
               "base-case parameters; events are applied as RESETs at the given times.")

elif page == "Export":
    st.header("Export simulation results")
    if ss.results is None or not len(ss.results["df"]) or len(ss.results["df"]) < 2:
        st.info("Run a simulation first – the export uses the latest run's results.")
    else:
        res = ss.results
        df = res["df"]
        fs_x = build(copy.deepcopy(res["cfg"]))
        cls = classify(fs_x, df.columns)
        cat = unit_catalogue(fs_x)
        cat = cat[cat.unit.isin({u for u, _ in cls.values()})]
        labels = {r.unit: f"{r.unit}  ·  {r.type}" for r in cat.itertuples()}
        major = [u for u in ("FEED", "C-100", "C-200", "T-200", "E-103", "E-201", "D-201") if u in labels]
        groups = {
            "Major equipment": major,
            "Columns only": [u for u in ("C-100", "C-200") if u in labels],
            "Controllers": [u for u in labels if fs_x[u].kind == "controller"],
            "Valves & pumps": [u for u in labels if fs_x[u].kind in ("valve", "pump", "vacuum pump")],
            "Lines & streams": [u for u in labels if fs_x[u].kind in ("liquid line", "vapour line", "air bleed")],
            "All units": list(labels),
        }
        ss.setdefault("exp_units", major)

        def _pick(g):
            ss.exp_units = groups[g]
        st.subheader("1 · Process units")
        bc = st.columns(len(groups))
        for c, g in zip(bc, groups):
            c.button(g, on_click=_pick, args=(g,), width="stretch", key=f"grp_{g}")
        units = st.multiselect("Units to export", list(labels), format_func=lambda u: labels[u], key="exp_units")

        st.subheader("2 · Variable types")
        a, b = st.columns([1.3, 1])
        mode = a.radio("Scope", ["Inputs / outputs only", "Inputs / outputs + internal variables"], index=0,
                       help="Inputs/outputs: interface variables of each unit (stream flows, pressures, temperatures, "
                       "levels, controller PV/SP/OP, valve positions …). Internal: everything else the model computes "
                       "for that unit (segment temperatures, hydraulics, intermediate properties).")
        scopes = ["io"]
        if mode.endswith("internal variables"):
            scopes.append("internal")
            if b.checkbox("Per-segment column profiles", True,
                          help="T, ABV, vapour flow, holdup, efficiency, HETP, flood for every bed segment"):
                scopes.append("profile")
            if b.checkbox("Differential state variables", True,
                          help="Raw integrator states: component holdups, pressures, valve stems, PID integrals …"):
                scopes.append("state")

        st.subheader("3 · Time window & format")
        t_end = float(df.index[-1])
        c1, c2, c3, c4 = st.columns(4)
        tr = c1.slider("Time window [h]", 0.0, t_end / 3600, (0.0, t_end / 3600), 0.05)
        tu = c2.selectbox("Time unit", ["s", "min", "h"], index=2)
        every = c3.number_input("Keep every n-th row", 1, 1000, 1,
                                help=f"Report interval in this run: {res['cfg']['solver']['report_dt_s']:g} s")
        meta = c4.checkbox("Metadata header (# lines)", False)
        csv_text, sub = build_export(fs_x, df, units, scopes, tu, (tr[0] * 3600, tr[1] * 3600), every, meta, res["cfg"])
        n_by = pd.Series([cls[c][1] for c in sub.columns]).value_counts() if len(sub.columns) else pd.Series(dtype=int)
        st.success(f"{len(sub.columns)} variables × {len(sub)} time points  ·  "
                   + "  ·  ".join(f"{SCOPES[k].split(' (')[0]}: {int(n_by.get(k, 0))}" for k in ("io", "internal", "profile", "state") if k in scopes)
                   + f"  ·  ≈{len(csv_text)/1e6:.2f} MB")
        with st.expander("Variable list"):
            vl = pd.DataFrame([{"column": c, "unit": cls[c][0], "type": SCOPES[cls[c][1]].split(" (")[0]} for c in sub.columns])
            st.dataframe(vl, hide_index=True, width="stretch", height=300)
        with st.expander("Data preview"):
            st.dataframe(sub.head(20), width="stretch")

        st.subheader("4 · Write file")
        fname = st.text_input("File name", ss.get("exp_name", "dealc_results.csv"))
        ss.exp_name = fname
        d1, d2 = st.columns([1, 2])
        d1.download_button("⬇ Download CSV", csv_text, fname if fname.endswith(".csv") else fname + ".csv",
                           "text/csv", type="primary", width="stretch", disabled=not len(sub.columns))
        with d2:
            folder = st.text_input("…or write to a folder on this machine", ss.get("exp_dir", "results"))
            ss.exp_dir = folder
            if st.button("💾 Write CSV to folder", disabled=not len(sub.columns)):
                try:
                    os.makedirs(folder, exist_ok=True)
                    path = os.path.abspath(os.path.join(folder, fname if fname.endswith(".csv") else fname + ".csv"))
                    with open(path, "w", newline="") as fh:
                        fh.write(csv_text)
                    st.success(f"Wrote {path}")
                except Exception as exc:
                    st.error(f"Write failed: {exc}")
        st.caption("Columns are named unit.variable; profiles use unit.variable[segment] (1 = top), states "
                   "unit.state.name. The solver log can be downloaded from the Simulation page.")

elif page == "Trust Center":
    st.header("Trust Center")
    st.markdown("The complete set of equations the simulator solves, written as mathematics and traced to the code "
                "that implements them, with the assumptions, data sources and automated verification checks. "
                "Equation numbers (e.g. **PC-9**) are stable identifiers you can cite in reviews.")
    n_eq = sum(len(m["equations"]) for m in MODELS.values())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model components", len(MODELS))
    m2.metric("Numbered equations", n_eq)
    m3.metric("Literature sources", len(REFS))
    m4.metric("Model states (current flowsheet)", build(copy.deepcopy(cfg)).n)
    tl, tu, tv, ta, tr = st.tabs(["📚 Model library", "🧭 Equations by unit", "✅ Verification",
                                  "📋 Assumptions & limitations", "📖 References"])
    with tl:
        c1, c2 = st.columns([1, 3.2])
        with c1:
            whole = st.toggle("Show complete document", False, help="Render every model in sequence")
            keys = [k for g in GROUPS[1:] for k, m in MODELS.items() if m["group"] == g]
            sel = st.radio("Model", keys, format_func=lambda k: f"{MODELS[k]['group']} · {MODELS[k]['title']}",
                           label_visibility="collapsed", disabled=whole)
            st.download_button("⬇ Equation reference (.md)", markdown_document(), "dealc_model_equations.md",
                               "text/markdown", width="stretch",
                               help="Markdown with $$LaTeX$$ blocks – renders in GitHub, Obsidian, VS Code, Jupyter")
        with c2:
            if whole:
                for g in GROUPS[1:]:
                    st.subheader(g)
                    for k in keys:
                        if MODELS[k]["group"] == g:
                            render_model(k, level=3)
                            st.divider()
            else:
                render_model(sel, level=2)
    with tu:
        fs_t = build(copy.deepcopy(cfg))
        tags = ["FEED", "C-100", "C-200", "T-200", "E-101", "E-102", "E-103", "E-201", "K-300", "D-201",
                "V-100", "V-200", "P-101", "P-201", "P-202", "P-203", "FV-100", "LV-101", "HV-102", "HV-103",
                "LV-201", "TV-202", "LV-203"] + (["PV-300"] if cfg["vacuum"]["pic_mv"] == "air bleed valve" else []) \
            + [t for t in fs_t.meta["controllers"]] + (["AT-103"] if "AT-103" in fs_t else []) + ["BAL"]
        tags = [t for t in tags if t in fs_t or t in ("FEED",)]
        st.dataframe(pd.DataFrame([{"Tag": t, "Type": fs_t[t].kind if t in fs_t else "",
                                    "Models": ", ".join(MODELS[k]["title"] for k in models_for_unit(t, cfg))} for t in tags]),
                     hide_index=True, width="stretch", height=220)
        tsel = st.selectbox("Unit", tags, index=tags.index("C-100"))
        render_unit_equations(tsel)
    with tv:
        st.markdown("Automated checks of the property models, numerics and conservation laws. Checks that need the "
                    "plant state use the cached steady state for the current configuration and the latest run.")
        if st.button("Run verification checks", type="primary"):
            fs_v = build(copy.deepcopy(cfg))
            hit = init_cache().get(structural_key(cfg))
            Xv = None
            if hit is not None and hit["n"] == fs_v.n:
                Xv = hit["X"].copy()
                for tag_, op in hit["ops"].items():
                    pid = fs_v[tag_]
                    if pid.mode == "auto":
                        Xv[pid.sl.start] = (op - pid.bias) * pid.Ti / max(pid.Kc, 1e-9)
            res_ok = ss.results if (ss.results is not None and ss.results.get("done")) else None
            with st.spinner("Checking…"):
                ss.checks = run_checks(cfg, fs_v if Xv is not None else None, Xv, res_ok)
            if Xv is None:
                st.info("Initialise or run the current configuration to include the plant-state checks "
                        "(steady state, energy balance, reproducibility).")
        if ss.get("checks"):
            st.dataframe(pd.DataFrame(ss.checks), hide_index=True, width="stretch")
        st.markdown("**Also available:** the solver-accuracy panel (per-step error, conservation closure) and the "
                    "*Tolerance check* button on the Simulation page, which estimates the global integration error.")
    with ta:
        st.markdown("#### Global modelling assumptions")
        st.markdown("""
- Beer = ethanol + water + non-volatile extract; aroma compounds, CO₂ and dissolved gases are not modelled.
- Each packed segment holds well-mixed liquid at its bubble point; vapour is in plug flow with a rate-based Murphree efficiency.
- One lumped vapour space per column; non-condensable air is tracked only in the condenser.
- Packing correlations are used inside their pre-loading range; no flooding, entrainment or maldistribution model.
- Ideal-solution enthalpies and volumes; constant pure-component heat capacities.
- Liquid lines have no holdup; vapour lines are orifice-type restrictions.
- All parameters are deterministic; no plant data reconciliation has been performed. **Calibrate before quantitative use.**
""")
        for g in GROUPS[1:]:
            items = [(MODELS[k]["title"], a) for k in MODELS if MODELS[k]["group"] == g for a in MODELS[k]["assumptions"]]
            if items:
                st.markdown(f"#### {g}")
                st.markdown("\n".join(f"- **{t}:** {a}" for t, a in items))
    with tr:
        for k, v in REFS.items():
            used = [MODELS[m]["title"] for m in MODELS if k in MODELS[m]["refs"]]
            st.markdown(f"- {v}  \n  <span style='color:#888'>used in: {', '.join(used)}</span>", unsafe_allow_html=True)

else:  # ------------------------------------------------------------ Simulation
    st.header("Simulation")
    status = st.container()
    kpi_ph = st.empty()
    prog_ph = st.empty()
    log_ph = st.empty()
    tab_p, tab_c, tab_s = st.tabs(["📈 Process trends", "🏛 Column profiles", "🧮 Solver accuracy"])
    with tab_p:
        proc_ph = st.empty()
    with tab_c:
        prof_ph = st.empty()
        hyd_ph = st.empty()
    with tab_s:
        st.caption(
            "**Local error / tolerance:** a posteriori estimate of each accepted BDF step's local truncation "
            "error (corrector − predictor), weighted RMS relative to atol + rtol·|y|. Values ≤ 1 mean the step met "
            "the tolerance; hover shows the state that dominated. **Conservation closure:** integrated feed − products "
            "− vent versus holdup inventory, for ethanol and total moles. Integration error shows up here as a "
            "tolerance-dependent drift. A floor that doesn't change with rtol is a model approximation "
            "(vapour-space moles neglect dT/dt). Use *Tolerance check* for a global-error estimate.")
        solv_ph = st.empty()
        tol_ph = st.container()
    dl_ph = st.container()

    def render(res, fs=None, live=False, n=[0]):
        n[0] += 1
        df, sdf = res["df"], res["sdf"]
        t_end_h = res["t_end_h"]
        if len(df):
            last, first = df.iloc[-1], df.iloc[0]
            items = [("Product ABV [%]", "C-100.abv_B", "", 3), ("Concentrate ABV [%]", "D-201.abv", "", 1),
                     ("C-100 top P [mbar]", "C-100.P_top_mbar", "", 1), ("Reboiler [kW]", "E-103.Q_kW", "", 0),
                     ("Sump T [°C]", "C-100.T_C", "", 2), ("EtOH vent loss [kg/h]", "E-201.vent_EtOH_kg_h", "", 2),
                     ("Max flood [%]", None, "", 0)]
            with kpi_ph.container():
                cols = st.columns(len(items))
                for c, (lab, key, unit, d) in zip(cols, items):
                    if key is None:
                        v = max(last.get("C-100.flood_max", 0), last.get("C-200.flood_max", 0))
                        c.metric(lab, f"{v:.0f}")
                    else:
                        c.metric(lab, f"{last[key]:.{d}f}", f"{last[key]-first[key]:+.{d}f}")
            prog_ph.progress(min(df.index[-1] / 3600 / t_end_h, 1.0),
                             text=f"t = {df.index[-1]/3600:.2f} / {t_end_h:.2f} h"
                             + (f" · wall {sdf['wall'].iloc[-1]:.1f} s · {len(sdf)} steps" if len(sdf) else ""))
        if res["events"]:
            log_ph.caption("Events: " + " · ".join(f"**{t/3600:.2f} h** {lab}" for t, lab in res["events"]))
        proc_ph.plotly_chart(process_figure(df, res["events"], res["meta"], ss.ref, t_end_h),
                             key=f"proc{n[0]}", width="stretch")
        solv_ph.plotly_chart(solver_figure(sdf, df, res["rtol"], t_end_h), key=f"solv{n[0]}", width="stretch")
        if fs is not None:
            prof_ph.plotly_chart(profile_figure(fs), key=f"prof{n[0]}", width="stretch")
            if not live:
                rows = []
                for col in ("C-100", "C-200"):
                    o, u = fs[col].out, fs[col]
                    rows.append({"Column": col, "Packing": u.pk_name, "HETP mean [m]": o.get("HETP_mean"),
                                 "Max flood [%]": o.get("flood_max"), "Bed ΔP [mbar]": o.get("dP_bed_mbar"),
                                 "ΔP/m [mbar/m]": o.get("dP_bed_mbar") / u.H,
                                 "Liquid load [m³/m²h]": o.get("uL_m3m2h"), "Mean holdup [-]": o.get("hL_mean"),
                                 "Top vapour [mol/s]": o.get("V_top")})
                hyd_ph.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")

    def warnings(df):
        msgs = []
        if len(df):
            if (df.filter(like="flood_max") > 100).any().any():
                msgs.append("A packed bed exceeded 100 % of its capacity F-factor – the model does not represent "
                            "flooding hydraulics beyond this point.")
            if (df.filter(like=".V_min") < 0).any().any():
                msgs.append("Negative vapour flow in a segment (vapour fully condensed by sub-cooled liquid).")
            bleed_mode = ss.results and ss.results["cfg"]["vacuum"]["pic_mv"] == "air bleed valve"
            sat = (df["PV-300.stem"] < 0.005) | (df["PV-300.stem"] > 0.995) if bleed_mode else (df["K-300.speed"] > 0.995)
            if sat.any():
                msgs.append("Vacuum system reached a limit (pump at full speed or bleed shut) – pressure "
                            "control was capacity-limited for part of the run.")
        for m in msgs:
            status.warning(m, icon="⚠️")

    run_req = ss.pop("run_request", False)
    init_req = ss.pop("init_request", False)
    if run_req or init_req:
        run_cfg = copy.deepcopy(cfg)
        fs = build(run_cfg)
        X0, hit = initial_state(fs, run_cfg, status)
        status.success(f"Steady state ready ({fs.n} states; max residual rate {hit['r']:.1e} /h"
                       + (f"; initialised in {hit['wall']:.1f} s" if "wall" in hit else "") + ").", icon="✅")
        if init_req:
            snap = fs.snapshot(0.0, X0)
            res = {"df": pd.DataFrame([snap], index=pd.Index([0.0], name="t")), "sdf": pd.DataFrame(),
                   "events": [], "meta": fs.meta, "t_end_h": run_cfg["sim"]["t_end_h"],
                   "rtol": run_cfg["solver"]["rtol"], "cfg": run_cfg, "X0": X0, "done": True}
            ss.results = res
            render(res, fs)
        else:
            skipped = apply_schedule(fs, run_cfg["schedule"])
            for s_ in skipped:
                status.warning(f"Schedule entry skipped: {s_}")
            if run_cfg["solver"]["jacobian"] == "sparse" and fs.sparsity is None:
                fs.detect_sparsity(X0)
            res = {"df": pd.DataFrame(), "sdf": pd.DataFrame(), "events": [], "meta": fs.meta,
                   "t_end_h": run_cfg["sim"]["t_end_h"], "rtol": run_cfg["solver"]["rtol"], "cfg": run_cfg,
                   "X0": X0, "done": False}
            ss.results = res
            rows, steps = [], []
            try:
                for upd in fs.run(run_cfg["sim"]["t_end_h"] * 3600, X0=X0, report_dt=run_cfg["solver"]["report_dt_s"],
                                  rtol=run_cfg["solver"]["rtol"], jac=run_cfg["solver"]["jacobian"],
                                  yield_every=refresh):
                    rows += upd.rows
                    steps += upd.steps
                    res["events"] += upd.events
                    res["df"] = pd.DataFrame(rows).set_index("t") if rows else pd.DataFrame()
                    res["sdf"] = pd.DataFrame(steps)
                    fs.evaluate(upd.t, upd.X)
                    fs.rhs(upd.t, upd.X)
                    render(res, fs, live=not upd.done)
                res["done"] = True
                status.success("Run complete.", icon="🏁")
            except Exception as exc:          # keep partial results visible
                status.error(f"Simulation stopped: {exc}")
            warnings(res["df"])
    elif ss.results is not None:
        res = ss.results
        if not res.get("done"):
            status.info("Showing partial results of an interrupted run.")
        fs = build(copy.deepcopy(res["cfg"]))
        fs.evaluate(0.0, res["X0"])
        render(res, None)
        prof_ph.info("Column profiles are shown live during a run. Press ▶ Run to see them again.")
        warnings(res["df"])
    else:
        status.info("Press **▶ Run** in the sidebar. The first run initialises the plant to steady state "
                    "(≈10–20 s), then integrates the schedule, streaming results as it goes.")

    # ----------------------------------------------------------- tolerance check & downloads
    if ss.results is not None and ss.results.get("done") and len(ss.results["sdf"]):
        res = ss.results
        with tol_ph:
            if st.button("Tolerance check: re-run at rtol/10 and compare", key="tolchk"):
                with st.spinner("Re-running at tighter tolerance…"):
                    fs2 = build(copy.deepcopy(res["cfg"]))
                    apply_schedule(fs2, res["cfg"]["schedule"])
                    r2 = fs2.simulate(res["t_end_h"] * 3600, X0=res["X0"], verbose=False,
                                      rtol=res["rtol"] / 10, report_dt=res["cfg"]["solver"]["report_dt_s"])
                a, b = res["df"], r2.df
                idx = a.index.intersection(b.index)
                keys = {"Product ABV [%]": "C-100.abv_B", "Concentrate ABV [%]": "D-201.abv",
                        "C-100 top P [mbar]": "C-100.P_top_mbar", "Reboiler duty [kW]": "E-103.Q_kW",
                        "Sump T [°C]": "C-100.T_C", "Drum level [m]": "D-201.level",
                        "Vent EtOH [kg/h]": "E-201.vent_EtOH_kg_h"}
                tab = pd.DataFrame([{"Variable": k, "max |Δ|": float((a.loc[idx, v] - b.loc[idx, v]).abs().max()),
                                     "range in run": float(a[v].max() - a[v].min())} for k, v in keys.items()])
                tab["Δ / range [%]"] = 100 * tab["max |Δ|"] / tab["range in run"].replace(0, np.nan)
                st.dataframe(tab.round(5), hide_index=True, width="stretch")
                st.caption(f"Reference run: rtol = {res['rtol']/10:.0e}, {len(r2.solver)} steps. "
                           "The deviation estimates the global error of the displayed run.")
        with dl_ph:
            c1, c2, c3 = st.columns(3)
            c1.download_button("Results CSV", res["df"].to_csv(), "dealc_results.csv", "text/csv")
            c2.download_button("Solver log CSV", res["sdf"].to_csv(index=False), "dealc_solver.csv", "text/csv")
            c3.download_button("Case JSON", json.dumps(res["cfg"], indent=2), "dealc_case.json", "application/json")
