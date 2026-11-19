"""
Command-line run of the default (or a saved) case without the UI.

    python examples/run_scenario.py                 # default case
    python examples/run_scenario.py my_case.json    # case saved from the app

Writes results/*.csv and results/*.html (interactive Plotly figures).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dealc.model import build, default_config, apply_schedule, kpis
from dealc.plots import process_figure, solver_figure, profile_figure

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

cfg = default_config()
if len(sys.argv) > 1:
    with open(sys.argv[1]) as f:
        for k, v in json.load(f).items():
            if isinstance(v, dict) and k in cfg:
                cfg[k].update(v)
            else:
                cfg[k] = v

fs = build(cfg)
print(f"{fs.name}: {len(fs.units)} units, {fs.n} states")
t0 = time.time()
X0, r = fs.initialize()
print(f"steady state in {time.time()-t0:.1f} s (max residual rate {r:.1e} 1/h)")
for k, v in kpis(fs).items():
    print(f"  {k:28s} {v:10.3f}")
profile_figure(fs).write_html(os.path.join(OUT, "profiles_initial.html"))

apply_schedule(fs, cfg["schedule"])
res = fs.simulate(cfg["sim"]["t_end_h"] * 3600, X0=X0, rtol=cfg["solver"]["rtol"],
                  report_dt=cfg["solver"]["report_dt_s"], jac=cfg["solver"]["jacobian"])
res.df.to_csv(os.path.join(OUT, "results.csv"))
res.solver.to_csv(os.path.join(OUT, "solver_log.csv"), index=False)
process_figure(res.df, res.events, fs.meta).write_html(os.path.join(OUT, "process_trends.html"))
solver_figure(res.solver, res.df).write_html(os.path.join(OUT, "solver_accuracy.html"))
print("final:", {k: round(v, 3) for k, v in kpis(fs).items()})
print(f"max |ethanol balance closure| = {res.df['BAL.closure_EtOH'].abs().max():.2e}")
