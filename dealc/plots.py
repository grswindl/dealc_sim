"""Plotly figures for the Streamlit front end (process trends, profiles, solver monitor, PFD)."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTE = ["#2a6fdb", "#e8833a", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf", "#7f7f7f"]


def _add(fig, df, r, c, cols, names, ref=None, dash=None, secondary=False, colors=None):
    if df is None or len(df) == 0:
        return
    th = df.index.values / 3600.0
    for i, (col, nm) in enumerate(zip(cols, names)):
        if col not in df:
            continue
        color = (colors or PALETTE)[i % len(PALETTE)]
        style = dict(color=color, width=1.8, dash=(dash[i] if dash else None))
        fig.add_trace(go.Scatter(x=th, y=df[col], name=nm, line=style, legendgroup=f"{r}{c}",
                                 showlegend=True, hovertemplate=f"{nm}: %{{y:.4g}}<extra></extra>"),
                      row=r, col=c, secondary_y=secondary)
        if ref is not None and col in ref:
            fig.add_trace(go.Scatter(x=ref.index.values / 3600.0, y=ref[col], name=f"{nm} (ref)",
                                     line=dict(color=color, width=1.0, dash="dot"),
                                     legendgroup=f"{r}{c}", showlegend=False, opacity=0.6),
                          row=r, col=c, secondary_y=secondary)


def process_figure(df, events=(), meta=None, ref=None, t_end_h=None):
    meta = meta or {}
    prod = meta.get("product_tag", "AIC-103")
    refl = meta.get("reflux_tag", "TIC-202")
    titles = ("Flows [m³/h]", "Product spec", "Pressures [mbar]", "Temperatures [°C]",
              "Duties [kW]", "Levels [m]", "Vacuum system", "Packing hydraulics")
    specs = [[{}, {}], [{}, {}], [{}, {}], [{"secondary_y": True}, {"secondary_y": True}]]
    fig = make_subplots(rows=4, cols=2, shared_xaxes=True, subplot_titles=titles,
                        vertical_spacing=0.06, horizontal_spacing=0.07, specs=specs)
    _add(fig, df, 1, 1, ["FIC-100.pv", "FIC-100.sp", "L-product.Q", "L-reflux.Q", "L-recycle.Q",
                         "L-distillate.Q"],
         ["Beer feed", "Feed SP", "De-alc product", "Reflux", "Recycle", "Concentrate"], ref,
         dash=[None, "dash", None, None, None, None],
         colors=[PALETTE[0], "#333", PALETTE[1], PALETTE[2], PALETTE[4], PALETTE[3]])
    if prod == "AIC-103":
        _add(fig, df, 1, 2, ["C-100.abv_B", "AT-103.value", "AIC-103.sp"],
             ["Product ABV true [%]", "Analyser AT-103", "ABV SP"], ref, dash=[None, None, "dash"],
             colors=[PALETTE[0], PALETTE[1], "#333"])
    else:
        _add(fig, df, 1, 2, ["TIC-103.pv", "TIC-103.sp", "C-100.abv_B"],
             ["Bed-bottom T [°C]", "T SP", "Product ABV [%]"], ref, dash=[None, "dash", None],
             colors=[PALETTE[0], "#333", PALETTE[1]])
    _add(fig, df, 2, 1, ["C-100.P_top_mbar", "PIC-300.sp", "C-200.P_top_mbar", "E-201.P_mbar"],
         ["C-100 top", "PIC SP", "C-200 top", "Condenser"], ref, dash=[None, "dash", None, None],
         colors=[PALETTE[0], "#333", PALETTE[2], PALETTE[4]])
    _add(fig, df, 2, 2, ["C-100.T_C", "C-100.T_top", f"{refl}.pv" if refl == "TIC-202" else "C-200.T4",
                         "C-200.T_top", "E-102.T_out_C", "E-201.T_c_C"],
         ["C-100 sump", "C-100 top", "C-200 control seg." if refl == "TIC-202" else "C-200 seg 4",
          "C-200 top", "Feed to column", "Condensing T"], ref)
    _add(fig, df, 3, 1, ["E-103.Q_kW", "E-201.Q_kW", "E-101.Q_kW", "E-102.Q_kW"],
         ["Reboiler E-103", "Condenser E-201", "Regen E-101", "Trim E-102"], ref)
    _add(fig, df, 3, 2, ["C-100.level", "T-200.level", "D-201.level"],
         ["C-100 sump", "T-200 (C-200 sump)", "D-201 drum"], ref)
    _add(fig, df, 4, 1, ["K-300.speed", "PV-300.stem", "E-201.y_air"],
         ["Vac. pump speed [-]", "Bleed valve [-]", "Air mole frac. [-]"], ref)
    _add(fig, df, 4, 1, ["E-201.vent_EtOH_kg_h"], ["EtOH vent loss [kg/h] (right)"], ref,
         secondary=True, colors=[PALETTE[3]])
    _add(fig, df, 4, 2, ["C-100.flood_max", "C-200.flood_max"], ["C-100 max flood [%]", "C-200 max flood [%]"],
         ref, colors=[PALETTE[0], PALETTE[2]])
    _add(fig, df, 4, 2, ["C-100.dP_bed_mbar", "C-200.dP_bed_mbar"], ["C-100 bed ΔP [mbar] (right)",
                                                                     "C-200 bed ΔP [mbar] (right)"],
         ref, secondary=True, dash=["dot", "dot"], colors=[PALETTE[0], PALETTE[2]])
    for t, lab in events:
        for r in range(1, 5):
            for c in range(1, 3):
                fig.add_vline(x=t / 3600.0, line=dict(color="#999", width=1, dash="dot"), row=r, col=c)
    fig.update_layout(height=1050, margin=dict(l=40, r=20, t=40, b=30), hovermode="x unified",
                      legend=dict(font=dict(size=10), orientation="v"), template="plotly_white")
    if t_end_h:
        fig.update_xaxes(range=[0, t_end_h])
    fig.update_xaxes(title_text="time [h]", row=4)
    return fig


def profile_figure(fs):
    fig = make_subplots(rows=1, cols=5, shared_yaxes=False,
                        subplot_titles=("Temperature [°C]", "Liquid ABV [%] (log)", "Vapour flow [mol/s]",
                                        "Flood [%]", "HETP [m]"))
    for i, (col, sym) in enumerate((("C-100", "circle"), ("C-200", "square"))):
        o = fs[col].out
        if "T_profile" not in o:
            continue
        N = fs[col].N
        z = np.arange(1, len(o["T_profile"]) + 1)
        zs = np.arange(1, N + 1)
        color = PALETTE[0] if i == 0 else PALETTE[2]
        common = dict(mode="lines+markers", marker=dict(symbol=sym, size=6), line=dict(color=color),
                      legendgroup=col)
        fig.add_trace(go.Scatter(x=o["T_profile"], y=z, name=col, **common), 1, 1)
        fig.add_trace(go.Scatter(x=np.maximum(o["abv_profile"], 1e-3), y=z, showlegend=False, **common), 1, 2)
        if "V_profile" in o:
            fig.add_trace(go.Scatter(x=o["V_profile"], y=z[:len(o["V_profile"])], showlegend=False, **common), 1, 3)
            fig.add_trace(go.Scatter(x=100 * o["flood_profile"], y=zs, showlegend=False, **common), 1, 4)
            fig.add_trace(go.Scatter(x=o["HETP_profile"], y=zs, showlegend=False, **common), 1, 5)
    fig.update_xaxes(type="log", row=1, col=2)
    fig.update_yaxes(autorange="reversed", title_text="segment (1 = top; last C-100 = sump)", row=1, col=1)
    for c in range(2, 6):
        fig.update_yaxes(autorange="reversed", row=1, col=c)
    fig.update_layout(height=380, margin=dict(l=40, r=20, t=40, b=30), template="plotly_white")
    return fig


def solver_figure(sdf, df=None, rtol=None, t_end_h=None):
    fig = make_subplots(rows=2, cols=3, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=("Step size h [s]", "BDF order",
                                        "Local error estimate / tolerance",
                                        "Cumulative work", "Conservation closure |error| (relative)",
                                        "Speed: simulated / wall time"))
    if sdf is not None and len(sdf):
        th = sdf["t"].values / 3600.0
        fig.add_trace(go.Scatter(x=th, y=sdf["h"], mode="lines", name="h", line=dict(color=PALETTE[0])), 1, 1)
        fig.add_trace(go.Scatter(x=th, y=sdf["order"], mode="lines", line=dict(color=PALETTE[1], shape="hv"),
                                 name="order"), 1, 2)
        err = sdf["err"].clip(lower=1e-6)
        fig.add_trace(go.Scatter(x=th, y=err, mode="markers", marker=dict(size=3, color=PALETTE[2]),
                                 name="error norm", text=sdf["err_state"],
                                 hovertemplate="%{y:.3f}  dominant: %{text}<extra></extra>"), 1, 3)
        fig.add_hline(y=1.0, line=dict(color="#d62728", dash="dash"), row=1, col=3)
        for col, nm, c in (("nfev", "RHS evals", PALETTE[0]), ("njev", "Jacobians ×100", PALETTE[3]),
                           ("nsteps", "steps", PALETTE[1])):
            y = sdf[col] * (100 if col == "njev" else 1)
            fig.add_trace(go.Scatter(x=th, y=y, name=nm, line=dict(color=c)), 2, 1)
        w = sdf["wall"].values
        rtf = np.where(w > 0, (sdf["t"].values - sdf["t"].values[0]) / np.maximum(w, 1e-9), np.nan)
        fig.add_trace(go.Scatter(x=th, y=rtf, name="real-time factor", line=dict(color=PALETTE[4])), 2, 3)
    if df is not None and len(df) and "BAL.closure_EtOH" in df:
        th = df.index.values / 3600.0
        fig.add_trace(go.Scatter(x=th, y=df["BAL.closure_EtOH"].abs().clip(lower=1e-14), name="ethanol",
                                 line=dict(color=PALETTE[0])), 2, 2)
        fig.add_trace(go.Scatter(x=th, y=df["BAL.closure_total"].abs().clip(lower=1e-14), name="total moles",
                                 line=dict(color=PALETTE[1])), 2, 2)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=3)
    fig.update_yaxes(type="log", row=2, col=2)
    fig.update_yaxes(type="log", row=2, col=3)
    if t_end_h:
        fig.update_xaxes(range=[0, t_end_h])
    fig.update_xaxes(title_text="time [h]", row=2)
    fig.update_layout(height=560, margin=dict(l=40, r=20, t=40, b=30), template="plotly_white",
                      legend=dict(font=dict(size=10)))
    return fig


def pfd_dot(cfg):
    """Graphviz DOT process flow diagram reflecting the structural options."""
    st, vac, ph = cfg["structure"], cfg["vacuum"], cfg["preheat"]
    c1, c2 = cfg["C100"], cfg["C200"]
    prod_ctl = "AIC-103 (ABV analyser)" if st["product_control"].startswith("analyser") else "TIC-103 (bed T)"
    refl_ctl = "TIC-202" if st["reflux_control"] == "C-200 temperature" else "FIC-202"
    pic_mv = "K-300 speed" if vac["pic_mv"] == "vacuum pump speed" else "PV-300 bleed"
    L = []
    a = L.append
    a('digraph G { rankdir=TB; splines=true; nodesep=0.5; ranksep=0.55; size="11,9"; ratio=compress;')
    a('node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica", fontsize=15, margin="0.2,0.1"];')
    a('edge [fontname="Helvetica", fontsize=12];')
    a('feed [label="Beer feed\\n%.1f m³/h, %.1f %% ABV\\n%.0f °C" , fillcolor="#fff3d6"];' % (
        cfg["feed"]["flow_m3h"], cfg["feed"]["abv"], cfg["feed"]["T_C"]))
    if ph["regen_enabled"]:
        a('E101 [label="E-101\\nregen HX"];')
    a('E102 [label="E-102 trim\\nTIC-102"];')
    a('C100 [label="C-100 stripper\\n%s\\nØ%.2f m × %.1f m (%d seg)", shape=box3d, fillcolor="#dbe8ff"];' % (
        c1["packing"].split(" (")[0], c1["diameter_m"], c1["height_m"], c1["n_seg"]))
    a('E103 [label="E-103 reboiler\\n%s"];' % prod_ctl)
    a('C200 [label="C-200 rectifier\\n%s\\nØ%.2f m × %.1f m (%d seg)", shape=box3d, fillcolor="#dbe8ff"];' % (
        c2["packing"].split(" (")[0], c2["diameter_m"], c2["height_m"], c2["n_seg"]))
    a('T200 [label="T-200 sump\\nLIC-203"];')
    a('E201 [label="E-201 vacuum\\ncondenser"];')
    a('K300 [label="K-300 vacuum pump\\nPIC-300 → %s", fillcolor="#f2e6ff"];' % pic_mv)
    a('D201 [label="D-201 drum\\nLIC-201"];')
    a('prod [label="De-alc beer\\nP-101 / LIC-101", fillcolor="#e3f6e3"];')
    a('conc [label="Ethanol concentrate\\nP-201", fillcolor="#e3f6e3"];')
    a('vent [label="Vent", shape=ellipse, fillcolor="#eeeeee"];')
    a('air [label="Air leak %.1f kg/h", shape=ellipse, fillcolor="#eeeeee"];' % vac["air_leak_kg_h"])
    if ph["regen_enabled"]:
        a('feed -> E101 [label="FV-100\\nFIC-100"]; E101 -> E102;')
        a('C100 -> E101 [label="bottoms"]; E101 -> prod;')
    else:
        a('feed -> E102 [label="FV-100\\nFIC-100"]; C100 -> prod [label="bottoms"];')
    a('E102 -> C100 [label="top"];')
    a('E103 -> C100 [label="duty", style=dashed];')
    a('C100 -> C200 [label="vapour V-100", color="#d62728"];')
    a('C200 -> E201 [label="vapour V-200", color="#d62728"];')
    a('C200 -> T200 [label="liquid"];')
    if st["recycle_to"] == "C-100 top":
        a('T200 -> C100 [label="recycle P-203"];')
    else:
        a('drain [label="Weak water\\nto drain", shape=ellipse, fillcolor="#eeeeee"]; T200 -> drain;')
    a('E201 -> D201 [label="condensate"];')
    a('D201 -> C200 [label="reflux P-202\\n%s"];' % refl_ctl)
    a('D201 -> conc;')
    a('air -> E201 [style=dashed]; E201 -> K300 [label="air + vapour"]; K300 -> vent;')
    if vac["pic_mv"] == "air bleed valve":
        a('bleed [label="PV-300 air bleed", shape=ellipse, fillcolor="#eeeeee"]; bleed -> E201 [style=dashed];')
    a('}')
    return "\n".join(L)
