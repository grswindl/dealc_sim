# dealc_sim
Python sim engine for de-alcoholization process of beer to make NA beer and spirits (to be denatured or diluted)

# Model - dynsim v4: two-column vacuum de-alcoholization of beer

* **Packed columns** with structured or random packing. Hydraulics and mass transfer come from published correlations.
* A **full dynamic energy balance**, including the heat stored in the liquid and the packing metal, flashing when pressure changes, subcooled feeds and reflux, and heat losses.
* **Pressure dynamics** for every column and the condenser, connected by vapor lines.
* A **vacuum system**: condenser with air accumulation and blanketing, air in-leakage, a liquid-ring pump, and pressure control by pump speed or by air bleed.
* (v3) An **interactive P&ID** with hover data and double-click unit specification, **CSV export** by unit and variable scope, and **JSON case files** that save and restore a simulation.
* A **live UI** where you edit the flowsheet, equipment, controllers and schedule, then watch the run stream in. A dedicated **solver-accuracy** panel shows how well the integration is doing.

```bash
pip install -r requirements.txt
streamlit run app.py                 # UI at http://localhost:8501
python examples/run_scenario.py      # headless run -> results/*.csv, *.html
```

## Using the app

| Page | What you do |
|---|---|
| **Flowsheet** | An interactive **P&ID**. **Hover** over any equipment, line or instrument to see its design data and the latest simulated values. **Double-click** to open a dialog with that unit's specification: columns, exchangers, condenser, vacuum pump, drums, every pump and control valve, vapor lines, the analyser and every controller (setpoint, tuning, mode). Double-clicking a liquid line opens its control valve. The drawing redraws to match the structural options in the expander below it: where the C-200 bottoms go (recycle to C-100 or drain), whether the regenerative heat exchanger is used, the product-quality loop (ABV analyser or bed temperature → reboiler), the reflux loop (temperature or flow), and the vacuum loop (what PIC-300 measures and whether it moves pump speed or the air bleed). |
| **Equipment** | Feed and preheat train, column geometry (packing, diameter, height, number of segments), mass-transfer calibration factor, reboiler, condenser, vacuum pump, air leak, vapor-line sizing. Also the packing library table. |
| **Control** | Editable table of all loops: setpoint, Kc, Ti, auto/manual. |
| **Schedule** | Table of timed RESET events: disturbances (feed ABV, flow, temperature, utilities, air leak), setpoint changes, faults (pump trip, stuck valve) and controller mode switches. |
| **Simulation** | ▶ Run initializes to steady state, which is cached per configuration and warm-started. It then streams KPIs, process trends, live column profiles and the solver monitor. You can keep a run as a reference overlay and run a tolerance check. |
| **Export** | Choose the **process units** to export (or a preset: major equipment, columns, controllers, valves & pumps, lines, all). Choose the **variable scope**: *inputs/outputs only* (each unit's interface variables: stream flows, P, T, levels, PV/SP/OP, valve positions), or *inputs/outputs + internal variables* (all other computed quantities), optionally adding **per-segment profiles** and **differential states**. Then set the time window, time unit, row thinning and an optional metadata header. Download the CSV or write it to a folder on the machine running the app. |

### Trust Center
A dedicated tab documenting **every equation the simulator solves**: 19 model components and 77 numbered equations (e.g. **PC-9**), written as mathematics (KaTeX) and traced to the class that implements them. It has five tabs:
- **Model library:** browse by component, or toggle *Show complete document* to read the whole model in one pass. Download it as Markdown with `$$…$$` LaTeX.
- **Equations by unit:** a map from P&ID tag to models, plus the governing equations and the parameter values used in the current case (packing constants for columns).
- **Verification:** automated checks run on demand. They cover the ethanol–water azeotrope and pure-component boiling points against literature, the exactness of the holdup-correlation inverse, the beer-composition round-trip, RHS reproducibility, steady-state convergence, the C-100 steady-state energy balance (closes to ~1e-12), mass conservation over the last run, and the fraction of solver steps within tolerance.
- **Assumptions & limitations:** global assumptions, then assumptions per model.
- **References:** the literature sources and the models that use each one.

In any unit's double-click dialog, the **📐 Show model equations for this unit** toggle shows the same equations and parameter values in place.

### Case files (sidebar)
A case `.json` holds the complete input configuration: structure, equipment, packing, pumps and valves, controller setpoints, tuning and modes, the schedule, and solver settings. You can optionally **embed the steady-state initial condition**, so a restored case starts without re-initializing. Save it with **⬇ Download** or **💾 Save** (to a path on the server machine). Restore it with **⬆ Upload** or **📂 Load**. Older or partial case files are deep-merged onto the defaults. `cases/default_case_with_steady_state.json` is included as an example.

Changing structure or design triggers re-initialization (about 15 s). Tuning changes don't: the integral terms are re-seated without a bump. Every run starts from the base-case parameters.

## Solver accuracy monitor

| Plot | What it tells you |
|---|---|
| Step size, BDF order | Where the dynamics are fast. Steps shrink at events and fast transients. |
| **Local error / tolerance** | An after-the-fact estimate of each accepted step's local truncation error (C_k·(corrector − predictor)), as a weighted RMS against atol + rtol·\|y\|. Values ≤ 1 mean the step met the tolerance. Hover shows the state that dominated the error. |
| **Conservation closure** | Integrated feed − products − vent, compared with the holdup inventory, for ethanol and for total moles. Integration error shows up as drift that depends on tolerance. A floor that stays the same as you tighten tolerance (≈1e-5 here) is a model approximation instead: the vapor-space moles neglect dT/dt. |
| Work, real-time factor | RHS evaluations, Jacobians, steps; simulated time per second of wall-clock time. |
| **Tolerance check** button | Re-runs at rtol/10 and reports the maximum deviation of the key variables, which estimates the global error. Default case at rtol 1e-4: product ABV within ±0.0002 %ABV, pressure within ±0.02 mbar. |

## Model summary

**Packed bed (`dynsim/columns.py`).** N segments. Each holds a well-mixed liquid holdup at its bubble point at the local pressure. Vapor moves up in plug flow, with segment Murphree efficiency E = 1 − exp(−dz/H_OG), where H_OG = H_G + λ·H_L. Liquid outflow comes from inverting the Billet–Schultes holdup correlation (trickle flow, no weirs).

**Energy balance (index-reduced).** Differentiating H = Σnᵢhᵢ(T_bub) + C_metal·T gives, for each stage, an equation linear in the vapor rate V_k and in dP/dt. A bottom-up sweep gives V_k = a_k + b_k·dP/dt. The top vapor-space balance (V_void/RT)·dP/dt = V_1 − V_exit then closes for dP/dt. The large negative b_k from flashing liquid is why vacuum columns respond slowly to pressure changes.

**Packing (`dynsim/packing.py`).**
- Holdup and pressure drop: Billet–Schultes, plus a loading-region rise.
- Structured packing: Rocha–Bravo–Fair k_G and k_L, Tsai effective area.
- Random packing: Onda k_G, k_L and wetted area.
- Capacity: F-factor against the packing's F_max.

| Packing | HETP | ΔP | Flood |
|---|---|---|---|
| Mellapak 250Y | ≈0.37 m | ≈1.3 mbar/m | ≈53% |
| Pall 50 mm | ≈0.95 m | ≈1.6 mbar/m | ≈61% |

(Values at the design point: 100 mbar, F ≈ 1.6.)

**Vacuum condenser (`VacuumCondenser`).** States are gas moles, air moles and condensable composition. The condensing temperature is the bubble point at the condensable partial pressure. Duty uses an ε-NTU model with blanketing factor (1 − y_air)². The vent gas, at the vacuum pump capacity S(P), carries air plus ethanol: this is the vent-loss output.

**Equipment.**
- Pump + valve lines: closed-form pressure balance with pump curve and affinity laws, valve characteristic and actuator lag, non-return valve and loss of suction.
- Vapor lines: m = CdA·√(2ρΔP).
- Heat exchangers: counterflow ε-NTU with thermal lag.
- Hot-water reboiler: Q = C(1 − e^(−UA/C))·ΔT.
- PID: ISA form, anti-windup, bumpless auto/manual.

**Thermodynamics.** NRTL + Antoine for ethanol–water; it reproduces the 1 atm azeotrope at 0.897 / 78.17 °C. Beer extract is a non-volatile pseudo-component. Ideal liquid enthalpies, with latent heats that vary with temperature. Viscosity, surface tension and diffusivity correlations for the transfer coefficients.

## Known limitations and next steps

- Packing constants are representative. Replace them with vendor data, or fit the mass-transfer multiplier and C_h / C_p to plant data.
- No flooding hydraulics beyond F_max. The app warns when a bed goes above 100%.
- Air is lumped in the condenser; leaks into the columns themselves aren't tracked. No vent condenser or scrubber yet: an obvious addition, because vent ethanol loss is significant.
- Heat of mixing is neglected (≤0.8 kJ/mol for ethanol–water).
- Start-up from an empty, cold plant needs liquid-free segments and a dry-packing model; not included.
- For an equation-oriented formulation with analytic Jacobians, sensitivities and parameter estimation, port the same equations to CasADi/IDAS or Pyomo.DAE. The `Unit.outputs/derivatives` split maps directly.

## Layout

```
dynsim/   thermo.py  packing.py  columns.py  equipment.py  control.py  core.py
dealc/    model.py (config, flowsheet builder, schedule parameters)   plots.py (Plotly)
          pid.py (P&ID SVG, tooltips, edit forms, component JS/CSS)   export.py (CSV export, case files)
          trust.py (equation library in LaTeX, unit→model map, verification checks)
app.py    Streamlit UI (requires Streamlit ≥ 1.50 for st.components.v2)
cases/    example case file
examples/run_scenario.py
```
