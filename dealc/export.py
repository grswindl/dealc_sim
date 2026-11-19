"""Result export (unit / variable-scope selection) and case-file (JSON) save/restore."""
import datetime as _dt
import io
import json

import numpy as np
import pandas as pd

CASE_FORMAT = "dealc-case"
CASE_VERSION = 3

SCOPES = {
    "io": "Inputs / outputs (interface variables)",
    "internal": "Internal variables (all other outputs)",
    "profile": "Per-segment profiles (columns)",
    "state": "Differential states",
}


def classify(fs, columns):
    """
    Map every recorded column 'unit.key' to (unit, scope) where scope is one of
    io / internal / profile / state, using each unit class's ``io_keys``.
    """
    out = {}
    names = sorted((u.name for u in fs.units), key=len, reverse=True)
    for c in columns:
        unit = next((n for n in names if c.startswith(n + ".")), None)
        if unit is None:
            continue
        rest = c[len(unit) + 1:]
        u = fs[unit]
        if rest.startswith("state."):
            scope = "state"
        elif "[" in rest:
            scope = "profile"
        else:
            scope = "io" if (u.io_keys is None or rest in u.io_keys) else "internal"
        out[c] = (unit, scope)
    return out


def unit_catalogue(fs):
    """DataFrame of units with their type, for the export selector."""
    return pd.DataFrame([{"unit": u.name, "type": u.kind, "states": u.n_states} for u in fs.units])


def select_columns(fs, df, units, scopes):
    cls = classify(fs, df.columns)
    units, scopes = set(units), set(scopes)
    return [c for c in df.columns if c in cls and cls[c][0] in units and cls[c][1] in scopes]


def build_export(fs, df, units, scopes, time_unit="s", t_range=None, every=1, meta_header=False,
                 cfg=None):
    """Return (csv_text, selected DataFrame)."""
    cols = select_columns(fs, df, units, scopes)
    sub = df[cols]
    if t_range is not None:
        sub = sub[(sub.index >= t_range[0]) & (sub.index <= t_range[1] + 1e-9)]
    sub = sub.iloc[::max(int(every), 1)]
    fac = {"s": 1.0, "min": 60.0, "h": 3600.0}[time_unit]
    sub = sub.copy()
    sub.index = pd.Index(np.round(sub.index.values / fac, 9), name=f"time [{time_unit}]")
    buf = io.StringIO()
    if meta_header:
        buf.write(f"# dealc simulation export {_dt.datetime.now().isoformat(timespec='seconds')}\n")
        buf.write(f"# units: {', '.join(sorted(set(units)))}\n")
        buf.write(f"# scopes: {', '.join(SCOPES[s] for s in scopes)}\n")
        if cfg is not None:
            buf.write(f"# schedule: {json.dumps(cfg.get('schedule', []))}\n")
    sub.to_csv(buf)
    return buf.getvalue(), sub


# ----------------------------------------------------------------------------- case files
def case_to_json(cfg, steady=None, results_summary=None):
    doc = {"format": CASE_FORMAT, "version": CASE_VERSION,
           "saved": _dt.datetime.now().isoformat(timespec="seconds"), "config": cfg}
    if steady is not None:
        doc["steady_state"] = {"key": steady["key"], "n": int(steady["n"]),
                               "X": [float(v) for v in steady["X"]], "ops": steady["ops"],
                               "r": float(steady.get("r", np.nan)), "state_names": steady.get("state_names")}
    if results_summary is not None:
        doc["last_results_summary"] = results_summary
    return json.dumps(doc, indent=2)


def case_from_json(text):
    """Returns (config_dict, steady_state_or_None). Accepts case files and bare config dicts."""
    d = json.loads(text)
    if isinstance(d, dict) and d.get("format") == CASE_FORMAT:
        st = d.get("steady_state")
        if st:
            st = dict(st, X=np.asarray(st["X"], dtype=float))
        return d["config"], st
    return d, None
