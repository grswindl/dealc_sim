"""
Interactive P&ID (SVG) for the de-alcoholisation flowsheet.

* ISA-5.1-style symbols: packed columns (hatched beds, elliptical heads), horizontal drums,
  centrifugal pumps, control valves with diaphragm actuators, shell-and-tube / plate
  exchangers, liquid-ring vacuum pump, shared-display (DCS) controller bubbles, dashed
  instrument-signal lines, off-page connectors at battery limits.
* Every unit / line is an SVG group with ``data-tag``; the Streamlit component shows a
  tooltip (design data + latest results) on hover and returns the tag on double-click.
* ``FORMS`` defines the editable parameters for each tag (used by the edit dialog).
"""
import html

import numpy as np

from dynsim.packing import PACKINGS
from .model import SP_MAP, LOOP_INFO

# =============================================================================== editable forms
PK_NAMES = list(PACKINGS)
CHARS = ["equal_pct", "linear", "quick"]


def _valve_form(tag, title):
    return (title, [
        (("valves", tag, "characteristic"), "Inherent characteristic", "select", CHARS, None, None,
         "Equal-% for most flow/level service; linear for utility valves."),
        (("valves", tag, "Kv"), "Kv [m³/h] (0 = auto-size)", "num", 0.0, 200.0, 0.1,
         "0 sizes the valve for ≈50 % opening at the design flow."),
        (("valves", tag, "tau"), "Actuator stroke time constant [s]", "num", 0.2, 120.0, 0.5, None)])


def _pump_form(tag, title):
    return (title, [
        (("pumps", tag, "dp_shutoff"), "Shut-off head ΔP₀ [bar]", "num", 0.5, 20.0, 0.1, None),
        (("pumps", tag, "dp_rated"), "Rated ΔP [bar]", "num", 0.2, 20.0, 0.1, None),
        (("pumps", tag, "Q_rated"), "Rated flow [m³/h]", "num", 0.05, 100.0, 0.05, None),
        (("pumps", tag, "eff"), "Efficiency [-]", "num", 0.1, 0.9, 0.01, None)])


def _col_form(sec, title, sump_label):
    return (title, [
        ((sec, "packing"), "Packing", "select", PK_NAMES, None, None, None),
        ((sec, "diameter_m"), "Diameter [m]", "num", 0.2, 3.0, 0.05, None),
        ((sec, "height_m"), "Packed height [m]", "num", 0.5, 25.0, 0.5, None),
        ((sec, "n_seg"), "Segments (discretisation)", "int", 2, 60, 1, "Each segment = one well-mixed liquid holdup."),
        ((sec, "mt_factor"), "Mass-transfer multiplier", "num", 0.2, 3.0, 0.05, "Calibrate to plant HETP."),
        ((sec, "UA_loss_W_m2K"), "Insulated-shell U [W/m²K]", "num", 0.0, 10.0, 0.1, None),
    ] + ([((sec, "sump_area_m2"), sump_label, "num", 0.05, 5.0, 0.05, None),
          ((sec, "level_sp_m"), "Sump level SP [m]", "num", 0.1, 2.0, 0.05, None)] if sec == "C100" else []))


FORMS = {
    "FEED": ("Beer feed (battery limit)", [
        (("feed", "flow_m3h"), "Feed flow SP [m³/h]", "num", 0.5, 20.0, 0.25, None),
        (("feed", "abv"), "Feed ABV [%]", "num", 0.5, 15.0, 0.1, None),
        (("feed", "extract_gL"), "Extract [g/L]", "num", 0.0, 150.0, 5.0, None),
        (("feed", "T_C"), "Feed temperature [°C]", "num", 0.0, 40.0, 1.0, None),
        (("feed", "P_bar"), "Supply pressure [bar a]", "num", 1.5, 8.0, 0.1, None)]),
    "E-101": ("E-101 regenerative feed/product exchanger", [
        (("preheat", "regen_enabled"), "In service", "bool", None, None, None, None),
        (("preheat", "regen_UA_kW_K"), "UA [kW/K]", "num", 1.0, 200.0, 1.0, None)]),
    "E-102": ("E-102 trim heater (hot water)", [
        (("preheat", "trim_UA_kW_K"), "UA [kW/K]", "num", 1.0, 100.0, 1.0, None),
        (("preheat", "trim_hw_T_C"), "Hot-water supply T [°C]", "num", 50.0, 95.0, 1.0, None),
        (("preheat", "trim_hw_max_kg_s"), "Max hot-water flow [kg/s]", "num", 0.1, 10.0, 0.1, None),
        (("preheat", "T_sp_C"), "Column feed T SP [°C]", "num", 20.0, 60.0, 0.5, None)]),
    "C-100": _col_form("C100", "C-100 stripping column (packed)", "Sump area [m²]"),
    "C-200": _col_form("C200", "C-200 rectifying column (packed)", ""),
    "T-200": ("T-200 C-200 sump", [
        (("C200", "sump_area_m2"), "Sump area [m²]", "num", 0.05, 5.0, 0.05, None),
        (("C200", "level_sp_m"), "Level SP [m]", "num", 0.1, 2.0, 0.05, None),
        (("structure", "recycle_to"), "Bottoms destination", "select", ["C-100 top", "drain"], None, None, None)]),
    "E-103": ("E-103 hot-water reboiler", [
        (("reboiler", "UA_kW_K"), "UA [kW/K]", "num", 5.0, 200.0, 1.0, None),
        (("reboiler", "hw_T_C"), "Hot-water supply T [°C]", "num", 55.0, 95.0, 1.0, None),
        (("reboiler", "hw_max_kg_s"), "Max hot-water flow [kg/s]", "num", 1.0, 40.0, 0.5, None),
        (("reboiler", "tau_s"), "Thermal lag [s]", "num", 5.0, 600.0, 5.0, None)]),
    "E-201": ("E-201 vacuum condenser", [
        (("condenser", "UA_kW_K"), "UA, air-free [kW/K]", "num", 5.0, 300.0, 1.0, None),
        (("condenser", "cw_T_C"), "Coolant inlet T [°C]", "num", -10.0, 30.0, 0.5, None),
        (("condenser", "cw_flow_kg_s"), "Coolant flow [kg/s]", "num", 1.0, 100.0, 1.0, None),
        (("condenser", "volume_m3"), "Gas-side volume [m³]", "num", 0.2, 20.0, 0.1, None),
        (("vacuum", "air_leak_kg_h"), "Air in-leakage (lumped) [kg/h]", "num", 0.0, 30.0, 0.1, None)]),
    "K-300": ("K-300 liquid-ring vacuum pump", [
        (("vacuum", "S_max_m3h"), "Capacity at full speed [m³/h]", "num", 20.0, 2000.0, 10.0, None),
        (("vacuum", "P_ult_mbar"), "Ultimate pressure [mbar]", "num", 5.0, 80.0, 1.0, None),
        (("vacuum", "pic_mv"), "PIC-300 manipulates", "select", ["vacuum pump speed", "air bleed valve"], None, None, None)]),
    "D-201": ("D-201 reflux drum", [
        (("drum", "area_m2"), "Cross-section [m²]", "num", 0.05, 3.0, 0.05, None),
        (("drum", "level_sp_m"), "Level SP [m]", "num", 0.1, 2.0, 0.05, None)]),
    "V-100": ("V-100 vapour line C-100 → C-200", [
        (("vapour_lines", "CdA_100_m2"), "Effective flow area CdA [m²]", "num", 0.005, 0.5, 0.005, None)]),
    "V-200": ("V-200 vapour line C-200 → E-201", [
        (("vapour_lines", "CdA_200_m2"), "Effective flow area CdA [m²]", "num", 0.005, 0.5, 0.005, None)]),
    "AT-103": ("AT-103 online ABV analyser", [
        (("structure", "analyser_tau_s"), "Total measurement lag [s]", "num", 1.0, 900.0, 5.0, "Two first-order lags in series.")]),
    "PV-300": ("PV-300 air bleed valve", [
        (("vacuum", "bleed_Kv"), "Capacity [mol/(s·bar)]", "num", 0.005, 1.0, 0.005, None),
        (("valves", "PV-300", "tau"), "Actuator time constant [s]", "num", 0.2, 60.0, 0.2, None)]),
}
for _t, _ttl in (("FV-100", "FV-100 feed flow valve"), ("LV-101", "LV-101 product level valve"),
                 ("HV-102", "HV-102 trim hot-water valve"), ("HV-103", "HV-103 reboiler hot-water valve"),
                 ("LV-201", "LV-201 concentrate level valve"), ("TV-202", "TV-202 reflux valve"),
                 ("LV-203", "LV-203 recycle level valve")):
    FORMS[_t] = _valve_form(_t, _ttl)
for _t, _ttl in (("P-101", "P-101 product pump"), ("P-201", "P-201 concentrate pump"),
                 ("P-202", "P-202 reflux pump"), ("P-203", "P-203 recycle pump")):
    FORMS[_t] = _pump_form(_t, _ttl)
for _t in SP_MAP:
    sec, key = SP_MAP[_t]
    extra = []
    if _t in ("TIC-202",):
        extra = [(("structure", "tic202_segment"), "Sensor segment (C-200)", "int", 1, 60, 1, None)]
    if _t == "PIC-300":
        extra = [(("vacuum", "pic_pv"), "Measurement", "select", ["C-100 top", "C-200 top", "condenser"], None, None, None)]
    FORMS[_t] = (f"{_t} – {LOOP_INFO[_t][0]} → {LOOP_INFO[_t][1]}", [
        ((sec, key), "Set point (initial)", "num", -1e4, 1e4, 0.05, None),
        (("tuning", _t, "Kc"), "Gain Kc [-]", "num", 0.0, 100.0, 0.05, None),
        (("tuning", _t, "Ti"), "Integral time Ti [s]", "num", 0.1, 1e5, 10.0, None),
        (("tuning", _t, "mode"), "Mode", "select", ["auto", "manual"], None, None, None)] + extra)
LINE_FORM = {"L-feed": "FV-100", "L-product": "LV-101", "L-recycle": "LV-203", "L-reflux": "TV-202",
             "L-distillate": "LV-201", "AT-103*": "AT-103"}


def get_path(cfg, path, default=None):
    d = cfg
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def set_path(cfg, path, value):
    d = cfg
    for k in path[:-1]:
        d = d.setdefault(k, {})
    d[path[-1]] = value


# =============================================================================== SVG primitives
def _e(s):
    return html.escape(str(s), quote=True)


def _pts(p):
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in p)


def line(points, cls="proc", arrow=True):
    m = ' marker-end="url(#arr)"' if arrow else ""
    return f'<polyline class="{cls}" points="{_pts(points)}"{m}/>'


def tagged_line(tag, points, cls="proc", arrow=True, label=None, lx=None, ly=None):
    s = f'<g class="u ln" data-tag="{_e(tag)}">' + line(points, cls, arrow)
    s += f'<polyline class="hit" points="{_pts(points)}"/>'
    if label:
        s += f'<text class="lbl" x="{lx}" y="{ly}">{_e(label)}</text>'
    return s + "</g>"


def text(x, y, s, cls="lbl", anchor="start"):
    return f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{_e(s)}</text>'


def group(tag, inner):
    return f'<g class="u" data-tag="{_e(tag)}">{inner}</g>'


def column(tag, x, y, w, h, bed, sump_top, level_frac, title, sub):
    hd = w * 0.28
    shell = (f'<path class="sym" d="M{x},{y+hd} A{w/2},{hd} 0 0 1 {x+w},{y+hd} L{x+w},{y+h-hd} '
             f'A{w/2},{hd} 0 0 1 {x},{y+h-hd} Z"/>')
    b0, b1 = bed
    bed_s = (f'<rect class="bed" x="{x+3}" y="{b0}" width="{w-6}" height="{b1-b0}"/>'
             f'<line class="thin" x1="{x+3}" y1="{b0}" x2="{x+w-3}" y2="{b1}"/>'
             f'<line class="thin" x1="{x+w-3}" y1="{b0}" x2="{x+3}" y2="{b1}"/>'
             f'<line class="thin" x1="{x+3}" y1="{b0}" x2="{x+w-3}" y2="{b0}"/>'
             f'<line class="thin" x1="{x+3}" y1="{b1}" x2="{x+w-3}" y2="{b1}"/>')
    dist = f'<line class="dist" x1="{x+8}" y1="{b0-10}" x2="{x+w-8}" y2="{b0-10}"/>'
    sup = f'<line class="thin" x1="{x+3}" y1="{b1+6}" x2="{x+w-3}" y2="{b1+6}" stroke-dasharray="3 3"/>'
    lev = ""
    if level_frac is not None and sump_top is not None:
        bot = y + h - 6
        ht = max(min(level_frac, 1.0), 0.0) * (bot - sump_top)
        lev = f'<rect class="liq" x="{x+4}" y="{bot-ht:.1f}" width="{w-8}" height="{ht:.1f}"/>'
    t = (text(x - 8, y + 18, title, "tagt", "end") + text(x + w / 2, b0 + (b1 - b0) / 2, sub, "sub", "middle"))
    return group(tag, shell + lev + bed_s + dist + sup + t)


def drum_h(tag, x, y, w, h, level_frac, title):
    r = h / 2
    shell = (f'<path class="sym" d="M{x+r},{y} L{x+w-r},{y} A{r*0.6},{r} 0 0 1 {x+w-r},{y+h} '
             f'L{x+r},{y+h} A{r*0.6},{r} 0 0 1 {x+r},{y} Z"/>')
    lev = ""
    if level_frac is not None:
        ht = max(min(level_frac, 1.0), 0.0) * (h - 4)
        lev = f'<rect class="liq" x="{x+r*0.5}" y="{y+h-2-ht:.1f}" width="{w-r}" height="{ht:.1f}"/>'
    return group(tag, shell + lev + text(x + w / 2, y - 8, title, "tagt", "middle"))


def pump(tag, cx, cy, out="left", r=13):
    sgn = -1 if out == "left" else 1
    s = (f'<circle class="sym" cx="{cx}" cy="{cy}" r="{r}"/>'
         f'<line class="sym" x1="{cx}" y1="{cy-r}" x2="{cx+sgn*r*1.4}" y2="{cy-r}"/>'
         f'<polygon class="sym" points="{_pts([(cx-r*0.8, cy+r*1.45), (cx+r*0.8, cy+r*1.45), (cx, cy+r*0.55)])}"/>')
    return group(tag, s + text(cx, cy + r + 26, tag, "tag", "middle"))


def valve(tag, cx, cy, vertical=False, s=9, label_dy=-26, left=False):
    body = (f'<polygon class="sym" points="{_pts([(cx-s, cy-s*0.7), (cx+s, cy+s*0.7), (cx+s, cy-s*0.7), (cx-s, cy+s*0.7)])}"/>'
            f'<line class="sym" x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy-s*1.7}"/>'
            f'<path class="sym act" d="M{cx-s*0.9},{cy-s*1.7} A{s*0.9},{s*0.9} 0 0 1 {cx+s*0.9},{cy-s*1.7} Z"/>')
    if vertical:
        body = f'<g transform="rotate(90 {cx} {cy})">{body}</g>'
    lx, ly = ((cx - 22 if left else cx + 24), cy + 4) if vertical else (cx, cy + label_dy)
    anc = ("end" if left else "start") if vertical else "middle"
    return group(tag, body + text(lx, ly, tag, "tag", anc))


def hx_plate(tag, cx, cy, r, title, lpos=None):
    zz = []
    n = 5
    for i in range(n + 1):
        zx = cx - r * 0.8 + i * (1.6 * r / n)
        zy = cy + (r * 0.45 if i % 2 else -r * 0.45)
        zz.append((zx, zy))
    s = f'<circle class="sym" cx="{cx}" cy="{cy}" r="{r}"/><polyline class="thin" points="{_pts(zz)}"/>'
    lx, ly, anc = lpos if lpos else (cx, cy + r + 14, "middle")
    return group(tag, s + text(lx, ly, title, "tag", anc))


def shell_tube(tag, x, y, w, h, title, kettle=False, tpos=None):
    r = h / 2
    if kettle:
        s = (f'<path class="sym" d="M{x},{y+h*0.25} L{x+w*0.25},{y+h*0.25} L{x+w*0.35},{y} L{x+w-r},{y} '
             f'A{r*0.5},{r} 0 0 1 {x+w-r},{y+h} L{x+w*0.35},{y+h} L{x+w*0.25},{y+h*0.75} L{x},{y+h*0.75} Z"/>')
        s += f'<path class="thin" d="M{x+8},{y+h*0.42} L{x+w*0.8},{y+h*0.42} A6,8 0 0 1 {x+w*0.8},{y+h*0.58} L{x+8},{y+h*0.58}"/>'
    else:
        s = (f'<path class="sym" d="M{x+r*0.5},{y} L{x+w-r*0.5},{y} A{r*0.5},{r} 0 0 1 {x+w-r*0.5},{y+h} '
             f'L{x+r*0.5},{y+h} A{r*0.5},{r} 0 0 1 {x+r*0.5},{y} Z"/>'
             f'<line class="thin" x1="{x+r*0.9}" y1="{y}" x2="{x+r*0.9}" y2="{y+h}"/>')
        for k in range(1, 4):
            yy = y + h * k / 4
            s += f'<line class="thin" x1="{x+r*0.9}" y1="{yy}" x2="{x+w-r*0.5}" y2="{yy}"/>'
    tx, ty, anc = tpos if tpos else (x + w / 2, y - 8, "middle")
    return group(tag, s + text(tx, ty, title, "tagt", anc))


def vac_pump(tag, cx, cy, r, title, ly=None):
    s = (f'<circle class="sym" cx="{cx}" cy="{cy}" r="{r}"/>'
         f'<circle class="thin" cx="{cx+r*0.22}" cy="{cy}" r="{r*0.6}"/>'
         f'<line class="thin" x1="{cx-r}" y1="{cy}" x2="{cx-r*0.2}" y2="{cy}"/>')
    return group(tag, s + text(cx, ly if ly else cy + r + 15, title, "tag", "middle"))


def bubble(tag, cx, cy, letters, num, dcs=True, r=17):
    s = ""
    if dcs:
        s += f'<rect class="isq" x="{cx-r}" y="{cy-r}" width="{2*r}" height="{2*r}"/>'
    s += (f'<circle class="ibub" cx="{cx}" cy="{cy}" r="{r}"/>'
          f'<line class="thin" x1="{cx-r}" y1="{cy}" x2="{cx+r}" y2="{cy}"/>'
          f'<text class="itxt" x="{cx}" y="{cy-4}" text-anchor="middle">{_e(letters)}</text>'
          f'<text class="itxt" x="{cx}" y="{cy+12}" text-anchor="middle">{_e(num)}</text>')
    return group(tag, s)


def signal(points):
    return f'<polyline class="sig" points="{_pts(points)}"/>'


def tap(x, y):
    return f'<circle class="tap" cx="{x}" cy="{y}" r="3"/>'


def offpage(tag, x, y, label, direction="right", w=118, h=26):
    if direction == "right":
        p = [(x, y - h / 2), (x + w - 12, y - h / 2), (x + w, y), (x + w - 12, y + h / 2), (x, y + h / 2)]
        tx = x + w / 2 - 5
    else:
        p = [(x + 12, y - h / 2), (x + w, y - h / 2), (x + w, y + h / 2), (x + 12, y + h / 2), (x, y)]
        tx = x + w / 2 + 5
    s = f'<polygon class="opc" points="{_pts(p)}"/>' + text(tx, y + 4, label, "opct", "middle")
    return group(tag, s) if tag else s


def badge(x, y, s, anchor="start"):
    return f'<text class="val" x="{x}" y="{y}" text-anchor="{anchor}">{_e(s)}</text>'


DEFS = """
<defs>
 <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
  <path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker>
 <marker id="arrv" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
  <path d="M0,0 L10,5 L0,10 z" fill="#c0392b"/></marker>
 <pattern id="hatch" patternUnits="userSpaceOnUse" width="7" height="7" patternTransform="rotate(45)">
  <line x1="0" y1="0" x2="0" y2="7" stroke="#9db4d8" stroke-width="1.2"/></pattern>
</defs>"""


# =============================================================================== the drawing
def pid_svg(cfg, row=None):
    """Return the P&ID SVG for the configuration (row = latest results, optional)."""
    st, vac, ph = cfg["structure"], cfg["vacuum"], cfg["preheat"]
    regen = ph["regen_enabled"]
    recycle = st["recycle_to"] == "C-100 top"
    analyser = st["product_control"].startswith("analyser")
    refl_T = st["reflux_control"] == "C-200 temperature"
    bleed = vac["pic_mv"] == "air bleed valve"

    def val(k, fmt="{:.1f}"):
        if row is None or k not in row:
            return None
        return fmt.format(row[k])

    def frac(k, hmax):
        return None if row is None or k not in row else row[k] / hmax

    S = [f'<svg viewBox="0 0 1500 860" xmlns="http://www.w3.org/2000/svg" class="pid">{DEFS}']
    a = S.append
    # ------------------------------------------------------------------ title block / legend
    a('<rect class="frame" x="4" y="4" width="1492" height="852"/>')
    a('<g class="tb"><rect class="frame" x="1180" y="770" width="312" height="82"/>'
      + text(1192, 792, "TWO-COLUMN VACUUM DE-ALCOHOLISATION", "tbt")
      + text(1192, 810, "P&ID – dynsim dynamic model", "tbs")
      + text(1192, 828, "Hover: data & latest results · Double-click: edit", "tbs")
      + text(1192, 845, f"PIC-300 → {'PV-300 air bleed' if bleed else 'K-300 speed (SC)'}", "tbs") + "</g>")
    a('<g class="legend"><rect class="frame" x="1180" y="680" width="312" height="84"/>'
      + line([(1192, 698), (1232, 698)], "proc") + text(1240, 702, "process liquid", "lg")
      + line([(1192, 718), (1232, 718)], "vap") + text(1240, 722, "vapour (pressure-driven)", "lg")
      + signal([(1192, 738), (1232, 738)]) + text(1240, 742, "instrument signal", "lg")
      + line([(1192, 756), (1232, 756)], "util") + text(1240, 760, "utility (HW / CW / air)", "lg")
      + f'<circle class="liq" cx="1420" cy="700" r="6"/>' + text(1430, 704, "liquid level", "lg") + "</g>")

    # ------------------------------------------------------------------ feed train
    a(offpage("FEED", 14, 560, "BEER FEED", "right"))
    a(tagged_line("L-feed", [(132, 560), (195, 560)] if not regen else [(132, 560), (210, 560)], arrow=True))
    a(valve("FV-100", 165, 560, label_dy=26))
    a(tap(148, 560))
    a(bubble("FIC-100", 148, 492, "FIC", "100"))
    a(signal([(148, 509), (148, 553)]) + signal([(165, 509), (165, 520), (165, 545)]))
    if regen:
        a(hx_plate("E-101", 240, 560, 30, "E-101", (276, 600, "start")))
        a(line([(270, 560), (300, 560), (300, 450), (311, 450)]))
        a(tagged_line("L-product", [(420, 730), (240, 730), (240, 591)], label="", lx=0, ly=0))
        a(line([(240, 529), (240, 330), (138, 330)]))
        a(offpage("PRODUCT", 14, 330, "DE-ALC BEER", "left", w=124))
        v = val("E-101.Th_out_C")
        if v:
            a(badge(150, 322, f"{v} °C"))
    else:
        a(line([(195, 560), (300, 560), (300, 450), (311, 450)]))
        a(tagged_line("L-product", [(420, 730), (138, 730)]))
        a(offpage("PRODUCT", 14, 730, "DE-ALC BEER", "left", w=124))
    a(hx_plate("E-102", 340, 450, 28, "E-102"))
    a(line([(340, 300), (340, 375)], "util") + line([(340, 388), (340, 421)], "util"))
    a(text(346, 300, "HWS", "util_t"))
    a(valve("HV-102", 340, 382, vertical=True, left=True))
    a(line([(368, 450), (440, 450), (440, 190), (499, 190)]))
    a(tap(405, 450))
    a(bubble("TIC-102", 405, 360, "TIC", "102"))
    a(signal([(405, 377), (405, 443)]) + signal([(388, 360), (372, 360), (356, 376)]))
    v = val("E-102.T_out_C")
    if v:
        a(badge(446, 444, f"{v} °C"))

    # ------------------------------------------------------------------ C-100
    a(column("C-100", 500, 115, 80, 540, (215, 530), 560, frac("C-100.level", 1.2),
             "C-100", cfg["C100"]["packing"].split(" (")[0]))
    v = val("C-100.P_top_mbar")
    if v:
        a(badge(588, 135, f"{v} mbar · {val('C-100.T_top')} °C"))
        a(badge(494, 668, f"{val('C-100.abv_B', '{:.3f}')} % ABV · {val('C-100.T_C')} °C", "end"))
    # reboiler
    a(shell_tube("E-103", 630, 585, 120, 50, "E-103", kettle=True, tpos=(690, 655, "middle")))
    a(line([(580, 625), (630, 625)]) + line([(690, 585), (690, 570), (581, 570)], "vap"))
    a(line([(840, 612), (790, 612)], "util") + line([(774, 612), (751, 612)], "util"))
    a(text(806, 604, "HWS", "util_t"))
    a(valve("HV-103", 782, 612, label_dy=26))
    # bottoms pump / LV-101
    a(line([(540, 655), (540, 706)]))
    a(pump("P-101", 540, 720, "left"))
    a(line([(527, 707), (500, 707), (500, 730), (433, 730)]))
    a(valve("LV-101", 424, 730, label_dy=26))
    a(tap(580, 610))
    a(bubble("LIC-101", 640, 680, "LIC", "101"))
    a(signal([(580, 610), (610, 610), (640, 650), (640, 663)]) + signal([(623, 680), (610, 690), (424, 690), (424, 718)]))
    # product quality control
    if analyser:
        a(tap(330, 730))
        a(bubble("AT-103", 330, 780, "AT", "103", dcs=False))
        a(signal([(330, 733), (330, 763)]))
        a(bubble("AIC-103", 782, 690, "AIC", "103"))
        a(signal([(347, 780), (800, 780), (800, 707)]) + signal([(782, 673), (782, 625)]))
    else:
        a(tap(580, 520))
        a(bubble("TIC-103", 782, 690, "TIC", "103"))
        a(signal([(580, 520), (830, 520), (830, 690), (799, 690)]) + signal([(782, 673), (782, 625)]))

    # ------------------------------------------------------------------ vapour to C-200
    a(tagged_line("V-100", [(540, 115), (540, 70), (800, 70), (800, 545), (859, 545)], "vap",
                  label="V-100", lx=660, ly=62))
    a(column("C-200", 860, 160, 80, 400, (240, 500), None, None, "C-200",
             cfg["C200"]["packing"].split(" (")[0]))
    a(group("T-200", f'<path class="sym" d="M862,580 L938,580 L938,640 A38,12 0 0 1 862,640 Z"/>'
            + (f'<rect class="liq" x="866" y="{640 - 55 * max(min(frac("T-200.level", 1.0) or 0, 1), 0):.1f}" '
               f'width="68" height="{55 * max(min(frac("T-200.level", 1.0) or 0, 1), 0):.1f}"/>' if row is not None else "")
            + text(900, 616, "T-200", "tag", "middle")))
    a(line([(900, 560), (900, 579)]))
    v = val("C-200.P_top_mbar")
    if v:
        a(badge(948, 182, f"{v} mbar · {val('C-200.T_top')} °C"))
    # recycle / drain
    a(line([(900, 652), (900, 706)]))
    a(pump("P-203", 900, 720, "left"))
    if recycle:
        a(tagged_line("L-recycle", [(887, 707), (870, 707), (870, 800), (766, 800)], arrow=False))
        a(valve("LV-203", 757, 800, label_dy=24))
        a(tagged_line("L-recycle", [(748, 800), (470, 800), (470, 170), (499, 170)]))
    else:
        a(tagged_line("L-recycle", [(887, 707), (870, 707), (870, 800), (766, 800)], arrow=False))
        a(valve("LV-203", 757, 800, label_dy=24))
        a(tagged_line("L-recycle", [(748, 800), (650, 800)]))
        a(offpage(None, 530, 800, "TO DRAIN", "left", w=118))
    a(tap(938, 610))
    a(bubble("LIC-203", 990, 680, "LIC", "203"))
    a(signal([(938, 610), (990, 610), (990, 663)]) + signal([(973, 680), (757, 680), (757, 785)]))

    # ------------------------------------------------------------------ overhead / vacuum
    a(tagged_line("V-200", [(900, 160), (900, 100), (1029, 100)], "vap", label="V-200", lx=930, ly=92))
    a(shell_tube("E-201", 1030, 78, 170, 50, "E-201", tpos=(1030, 70, "start")))
    a(line([(1050, 152), (1050, 129)], "util") + text(1022, 162, "CWS", "util_t"))
    a(line([(1170, 77), (1170, 56)], "util") + text(1176, 60, "CWR", "util_t"))
    a(offpage(None, 1040, 32, "AIR IN-LEAKAGE", "right", w=128) + line([(1104, 45), (1104, 77)], "util"))
    a(line([(1200, 103), (1290, 103)], "vap"))
    a(vac_pump("K-300", 1310, 103, 20, "K-300", ly=76))
    a(line([(1330, 103), (1380, 103), (1380, 60)], "vap") + offpage(None, 1340, 34, "VENT", "right", w=80))
    v = val("E-201.P_mbar")
    if v:
        a(badge(1090, 150, f"{v} mbar · air {val('E-201.y_air', '{:.2f}')} · Tc {val('E-201.T_c_C')} °C"))
        a(badge(1318, 145, f"speed {val('K-300.speed', '{:.0%}')}"))
        a(badge(1390, 120, f"EtOH {val('E-201.vent_EtOH_kg_h')} kg/h"))
    # pressure control
    pv_xy = {"C-100 top": (650, 70), "C-200 top": (900, 130), "condenser": (1200, 90)}[vac["pic_pv"]]
    a(tap(*pv_xy))
    a(bubble("PIC-300", 1310, 220, "PIC", "300"))
    a(signal([pv_xy, (pv_xy[0], 40) if vac["pic_pv"] != "condenser" else (1230, 90),
              (1230, 40) if vac["pic_pv"] != "condenser" else (1230, 90), (1230, 220), (1293, 220)]))
    if bleed:
        a(line([(1265, 190), (1265, 115)], "util") + text(1272, 198, "ATM AIR", "util_t"))
        a(valve("PV-300", 1265, 160, vertical=True, left=True))
        a(signal([(1293, 210), (1275, 168)]))
    else:
        a(text(1316, 180, "SC", "util_t") + signal([(1310, 203), (1310, 123)]))
    # condensate & drum
    a(line([(1080, 128), (1080, 289)]))
    a(drum_h("D-201", 1030, 290, 170, 52, frac("D-201.level", 1.0), "D-201"))
    v = val("D-201.abv")
    if v:
        a(badge(1205, 290, f"{v} % ABV"))
    a(tap(1200, 316))
    a(bubble("LIC-201", 1265, 260, "LIC", "201"))
    a(signal([(1200, 316), (1230, 316), (1265, 290), (1265, 277)]))
    # reflux
    a(line([(1080, 342), (1080, 406)]))
    a(pump("P-202", 1080, 420, "left"))
    a(tagged_line("L-reflux", [(1067, 407), (1030, 407), (1030, 420), (1013, 420)], arrow=False))
    a(valve("TV-202", 1004, 420, label_dy=26))
    a(tagged_line("L-reflux", [(995, 420), (975, 420), (975, 200), (941, 200)]))
    if refl_T:
        seg = int(st.get("tic202_segment", 4))
        n2 = cfg["C200"]["n_seg"]
        ys = 240 + (seg - 0.5) / n2 * 260
        a(tap(940, ys))
        a(bubble("TIC-202", 1030, 485, "TIC", "202"))
        a(signal([(940, ys), (955, ys), (955, 485), (1013, 485)]) + signal([(1030, 468), (1004, 440), (1004, 434)]))
    else:
        a(tap(1045, 420))
        a(bubble("FIC-202", 1030, 485, "FIC", "202"))
        a(signal([(1045, 420), (1045, 468)]) + signal([(1013, 485), (1004, 485), (1004, 434)]))
    # distillate
    a(line([(1160, 342), (1160, 406)]))
    a(pump("P-201", 1160, 420, "right"))
    a(tagged_line("L-distillate", [(1173, 407), (1200, 407), (1200, 420), (1256, 420)], arrow=False))
    a(valve("LV-201", 1265, 420, label_dy=26))
    a(tagged_line("L-distillate", [(1274, 420), (1355, 420)]))
    a(offpage("CONC", 1356, 420, "EtOH CONC.", "right", w=120))
    a(signal([(1282, 260), (1300, 260), (1300, 380), (1265, 380), (1265, 406)]))
    a("</svg>")
    return "".join(S)


# =============================================================================== tooltips
def unit_info(cfg, fs, row=None):
    """Tag -> {title, rows, editable}: design data plus the latest simulated values."""
    def v(k, fmt="{:.3g}", unit=""):
        if row is None or k not in row:
            return "—"
        return fmt.format(row[k]) + (f" {unit}" if unit else "")

    info = {}

    def put(tag, title, rows):
        info[tag] = {"title": _e(title),
                     "rows": [[_e(a), _e(b)] for a, b in rows],
                     "editable": (tag in FORMS) or (tag in LINE_FORM)}

    f = cfg["feed"]
    put("FEED", "Beer feed", [("Flow SP", f"{f['flow_m3h']} m³/h"), ("ABV", f"{f['abv']} %"),
                              ("Extract", f"{f['extract_gL']} g/L"), ("Temperature", f"{f['T_C']} °C"),
                              ("Supply P", f"{f['P_bar']} bar a"), ("Latest flow", v("FIC-100.pv", unit="m³/h")),
                              ("Latest ABV", v("FEED.abv", unit="%"))])
    for tag in ("FV-100", "LV-101", "HV-102", "HV-103", "LV-201", "TV-202", "LV-203", "PV-300"):
        if tag not in fs:
            continue
        u = fs[tag]
        put(tag, f"{tag} control valve", [("Characteristic", u.char), ("Kv", f"{u.Kv:.3g}"),
                                          ("Actuator τ", f"{u.tau:g} s"), ("Stem", v(f"{tag}.stem", "{:.1%}")),
                                          ("Kv·f(stem)", v(f"{tag}.Kv_eff"))])
    lines = {"P-101": "L-product", "P-201": "L-distillate", "P-202": "L-reflux", "P-203": "L-recycle"}
    for tag, ln in lines.items():
        p = cfg["pumps"][tag]
        put(tag, f"{tag} centrifugal pump", [("Shut-off ΔP", f"{p['dp_shutoff']} bar"),
                                             ("Rated", f"{p['dp_rated']} bar @ {p['Q_rated']} m³/h"),
                                             ("Flow", v(f"{ln}.Q", unit="m³/h")), ("ΔP", v(f"{ln}.dp_pump", unit="bar")),
                                             ("Shaft power", v(f"{ln}.power_kW", unit="kW")),
                                             ("Speed", v(f"{tag}.speed", "{:.0%}"))])
    for ln, title in (("L-feed", "Feed line"), ("L-product", "Product line"), ("L-recycle", "Recycle line"),
                      ("L-reflux", "Reflux line"), ("L-distillate", "Concentrate line")):
        put(ln, title, [("Flow", v(f"{ln}.Q", unit="m³/h")), ("Molar flow", v(f"{ln}.F", unit="mol/s")),
                        ("Valve ΔP", v(f"{ln}.dp_valve", unit="bar")), ("(double-click edits", f"{LINE_FORM[ln]})")])
    for col, sec, lab in (("C-100", "C100", "stripping"), ("C-200", "C200", "rectifying")):
        c = cfg[sec]
        rows = [("Packing", c["packing"]), ("Ø × H", f"{c['diameter_m']} m × {c['height_m']} m"),
                ("Segments", f"{c['n_seg']} × {c['height_m']/c['n_seg']:.2f} m"), ("MT multiplier", c["mt_factor"]),
                ("Top P", v(f"{col}.P_top_mbar", "{:.1f}", "mbar")), ("Bed ΔP", v(f"{col}.dP_bed_mbar", "{:.2f}", "mbar")),
                ("Top T", v(f"{col}.T_top", "{:.1f}", "°C")), ("Top vapour", v(f"{col}.V_top", "{:.2f}", "mol/s")),
                ("Mean HETP", v(f"{col}.HETP_mean", "{:.2f}", "m")), ("Max flood", v(f"{col}.flood_max", "{:.0f}", "%")),
                ("Liquid load", v(f"{col}.uL_m3m2h", "{:.1f}", "m³/m²h"))]
        if col == "C-100":
            rows += [("Sump T", v("C-100.T_C", "{:.2f}", "°C")), ("Product ABV", v("C-100.abv_B", "{:.3f}", "%")),
                     ("Sump level", v("C-100.level", "{:.2f}", "m")), ("Boil-up", v("C-100.boilup", "{:.2f}", "mol/s"))]
        put(col, f"{col} {lab} column", rows)
    c2 = cfg["C200"]
    put("T-200", "T-200 C-200 sump", [("Area", f"{c2['sump_area_m2']} m²"), ("Level", v("T-200.level", "{:.2f}", "m")),
                                       ("ABV", v("T-200.abv", "{:.2f}", "%")), ("T", v("T-200.T_C", "{:.1f}", "°C")),
                                       ("Destination", cfg["structure"]["recycle_to"])])
    put("E-101", "E-101 regenerative HX", [("In service", cfg["preheat"]["regen_enabled"]),
                                           ("UA", f"{cfg['preheat']['regen_UA_kW_K']} kW/K"), ("Duty", v("E-101.Q_kW", "{:.0f}", "kW")),
                                           ("Feed out", v("E-101.Tc_out_C", "{:.1f}", "°C")),
                                           ("Product out", v("E-101.Th_out_C", "{:.1f}", "°C"))])
    put("E-102", "E-102 trim heater", [("UA", f"{cfg['preheat']['trim_UA_kW_K']} kW/K"),
                                       ("HW supply", f"{cfg['preheat']['trim_hw_T_C']} °C"),
                                       ("Duty", v("E-102.Q_kW", "{:.0f}", "kW")), ("Outlet T", v("E-102.T_out_C", "{:.2f}", "°C"))])
    r = cfg["reboiler"]
    put("E-103", "E-103 hot-water reboiler", [("UA", f"{r['UA_kW_K']} kW/K"), ("HW supply", f"{r['hw_T_C']} °C"),
                                              ("Max HW flow", f"{r['hw_max_kg_s']} kg/s"), ("Duty", v("E-103.Q_kW", "{:.0f}", "kW"))])
    cd = cfg["condenser"]
    put("E-201", "E-201 vacuum condenser", [("UA (air-free)", f"{cd['UA_kW_K']} kW/K"), ("Coolant", f"{cd['cw_T_C']} °C, {cd['cw_flow_kg_s']} kg/s"),
                                            ("Air leak", f"{cfg['vacuum']['air_leak_kg_h']} kg/h"), ("P", v("E-201.P_mbar", "{:.1f}", "mbar")),
                                            ("Condensing T", v("E-201.T_c_C", "{:.1f}", "°C")), ("Air fraction", v("E-201.y_air", "{:.3f}")),
                                            ("Duty", v("E-201.Q_kW", "{:.0f}", "kW")), ("EtOH vent loss", v("E-201.vent_EtOH_kg_h", "{:.2f}", "kg/h"))])
    vc = cfg["vacuum"]
    put("K-300", "K-300 liquid-ring vacuum pump", [("Capacity", f"{vc['S_max_m3h']} m³/h"), ("Ultimate P", f"{vc['P_ult_mbar']} mbar"),
                                                   ("Speed", v("K-300.speed", "{:.0%}")), ("Suction flow", v("E-201.pump_S_m3h", "{:.0f}", "m³/h")),
                                                   ("Vent", v("E-201.vent_mol_s", "{:.3f}", "mol/s"))])
    put("D-201", "D-201 reflux drum", [("Area", f"{cfg['drum']['area_m2']} m²"), ("Level", v("D-201.level", "{:.2f}", "m")),
                                       ("Concentrate ABV", v("D-201.abv", "{:.1f}", "%")), ("T", v("D-201.T_C", "{:.1f}", "°C"))])
    for vt, ck in (("V-100", "CdA_100_m2"), ("V-200", "CdA_200_m2")):
        put(vt, f"{vt} vapour line", [("CdA", f"{cfg['vapour_lines'][ck]} m²"), ("Flow", v(f"{vt}.F", "{:.2f}", "mol/s")),
                                      ("ΔP", v(f"{vt}.dP_mbar", "{:.2f}", "mbar"))])
    put("AT-103", "AT-103 ABV analyser", [("Lag", f"{cfg['structure'].get('analyser_tau_s', 60)} s"),
                                          ("Reading", v("AT-103.value", "{:.3f}", "%")), ("True value", v("C-100.abv_B", "{:.3f}", "%"))])
    for tag in fs.meta.get("controllers", []):
        tu = cfg["tuning"].get(tag, {})
        sec, key = SP_MAP[tag]
        put(tag, f"{tag} controller", [("PV", LOOP_INFO[tag][0]), ("MV", LOOP_INFO[tag][1]), ("Action", fs[tag].action),
                                       ("SP (initial)", cfg[sec][key]), ("Kc / Ti", f"{tu.get('Kc')} / {tu.get('Ti')} s"),
                                       ("Mode", tu.get("mode", "auto")), ("PV now", v(f"{tag}.pv", "{:.4g}")),
                                       ("Output", v(f"{tag}.op", "{:.1%}"))])
    info["CONC"] = {"title": "Ethanol concentrate (battery limit)", "rows": [["Flow", _e(v("L-distillate.Q", unit="m³/h"))],
                                                                            ["ABV", _e(v("D-201.abv", "{:.1f}", "%"))]], "editable": False}
    info["PRODUCT"] = {"title": "De-alcoholised beer (battery limit)",
                       "rows": [["Flow", _e(v("L-product.Q", unit="m³/h"))], ["ABV", _e(v("C-100.abv_B", "{:.3f}", "%"))]],
                       "editable": False}
    return info


# =============================================================================== component assets
CSS = """
.pid-root { position: relative; width: 100%; }
svg.pid { width: 100%; height: auto; background: #fdfdfb; font-family: Helvetica, Arial, sans-serif; }
.frame { fill: none; stroke: #444; stroke-width: 1.2; }
.sym { fill: #f4f7fc; stroke: #1f2a44; stroke-width: 2; }
.act { fill: #fff; }
.thin { fill: none; stroke: #1f2a44; stroke-width: 1; }
.bed { fill: url(#hatch); stroke: none; }
.dist { stroke: #1f2a44; stroke-width: 2; stroke-dasharray: 6 3; }
.liq { fill: #9fd0f5; opacity: 0.75; }
.proc { fill: none; stroke: #222; stroke-width: 2; }
.vap { fill: none; stroke: #c0392b; stroke-width: 2.4; marker-end: url(#arrv); }
.util { fill: none; stroke: #6b8e23; stroke-width: 1.6; marker-end: url(#arr); }
.sig { fill: none; stroke: #555; stroke-width: 1.1; stroke-dasharray: 7 4; }
.tap { fill: #222; }
.hit { fill: none; stroke: transparent; stroke-width: 14; }
.ibub { fill: #fff; stroke: #1f2a44; stroke-width: 1.4; }
.isq { fill: #fff; stroke: #1f2a44; stroke-width: 1.2; }
.itxt { font-size: 11px; font-weight: 700; fill: #1f2a44; }
.tag { font-size: 11px; fill: #1f2a44; font-weight: 600; }
.tagt { font-size: 14px; fill: #1f2a44; font-weight: 700; }
.sub { font-size: 10px; fill: #2a4a80; font-style: italic; }
.lbl { font-size: 11px; fill: #c0392b; font-weight: 700; }
.lg { font-size: 11px; fill: #333; }
.util_t { font-size: 10px; fill: #6b8e23; font-weight: 700; }
.opc { fill: #fff8e5; stroke: #1f2a44; stroke-width: 1.4; }
.opct { font-size: 11px; font-weight: 700; fill: #1f2a44; }
.val { font-size: 10.5px; fill: #0b6e4f; font-weight: 600; }
.tbt { font-size: 12px; font-weight: 700; fill: #1f2a44; }
.tbs { font-size: 10.5px; fill: #444; }
.u { cursor: pointer; }
.u.hot .sym, .u.hot .ibub, .u.hot .isq, .u.hot .opc { stroke: #e8833a; stroke-width: 3; }
.u.hot polyline.proc, .u.hot polyline.vap { stroke: #e8833a; stroke-width: 3.5; }
.pid-tip { position: absolute; display: none; pointer-events: none; z-index: 20; background: #fff;
  border: 1px solid #9aa5b8; border-radius: 8px; padding: 8px 10px; box-shadow: 0 4px 14px rgba(0,0,0,.15);
  font: 12px Helvetica, Arial, sans-serif; color: #1f2a44; max-width: 360px; }
.pid-tip b { display: block; margin-bottom: 4px; font-size: 13px; }
.pid-tip table { border-collapse: collapse; }
.pid-tip td { padding: 1px 8px 1px 0; vertical-align: top; }
.pid-tip td:first-child { color: #66708a; white-space: nowrap; }
.pid-tip .hint { margin-top: 5px; color: #e8833a; font-size: 11px; }
"""

JS = """
export default function(component) {
  const { data, setTriggerValue, parentElement } = component;
  let root = parentElement.querySelector('.pid-root');
  if (!root) { root = document.createElement('div'); root.className = 'pid-root'; parentElement.appendChild(root); }
  root.innerHTML = data.svg + '<div class="pid-tip"></div>';
  const tip = root.querySelector('.pid-tip');
  const info = data.info || {};
  root.querySelectorAll('[data-tag]').forEach((el) => {
    const tag = el.getAttribute('data-tag');
    el.addEventListener('mousemove', (ev) => {
      ev.stopPropagation();
      const d = info[tag];
      if (!d) return;
      tip.innerHTML = '<b>' + d.title + '</b><table>' +
        d.rows.map((r) => '<tr><td>' + r[0] + '</td><td>' + r[1] + '</td></tr>').join('') + '</table>' +
        (d.editable ? '<div class="hint">Double-click to edit parameters</div>' : '');
      tip.style.display = 'block';
      const rb = root.getBoundingClientRect();
      let x = ev.clientX - rb.left + 16, y = ev.clientY - rb.top + 16;
      if (x + tip.offsetWidth > rb.width) x = ev.clientX - rb.left - tip.offsetWidth - 16;
      if (y + tip.offsetHeight > rb.height) y = Math.max(0, ev.clientY - rb.top - tip.offsetHeight - 16);
      tip.style.left = x + 'px'; tip.style.top = y + 'px';
      el.classList.add('hot');
    });
    el.addEventListener('mouseleave', () => { tip.style.display = 'none'; el.classList.remove('hot'); });
    el.addEventListener('dblclick', (ev) => {
      ev.preventDefault(); ev.stopPropagation();
      tip.style.display = 'none';
      setTriggerValue('edit', tag);
    });
  });
}
"""
