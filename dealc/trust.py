r"""
Trust Center: the model equations actually solved by dynsim, in mathematical notation,
with nomenclature, assumptions, source-code traceability, references and verification checks.

Every equation here is transcribed from the implementation (module named in ``code``);
if you change a model, update its entry so the documentation stays truthful.
"""
import numpy as np

from dynsim.packing import PACKINGS

# =============================================================================== model library
# Each model: title, code, purpose, equations [(heading, latex, note)], nomenclature
# [(symbol, meaning, unit)], assumptions [..], refs [ref keys]
MODELS = {}


def _m(key, **kw):
    MODELS[key] = kw


_m("thermo", title="Thermodynamics & physical properties", code="dynsim/thermo.py", group="Properties",
   purpose="Phase equilibrium, enthalpies and transport properties of beer modelled as ethanol (E), "
           "water (W) and a non-volatile extract pseudo-component (X, maltose-equivalent, M = 342.3 g/mol).",
   equations=[
       ("Vapour pressure (Antoine)",
        r"\log_{10}\!\left(\frac{P_i^{\mathrm{sat}}}{\mathrm{mmHg}}\right)=A_i-\frac{B_i}{\theta+C_i},\qquad "
        r"\begin{array}{l}E:\;A=8.20417,\;B=1642.89,\;C=230.300\\ W:\;A=8.07131,\;B=1730.63,\;C=233.426\end{array}",
        r"$\theta$ = temperature in °C."),
       ("Activity coefficients (NRTL, extract-free basis)",
        r"\begin{aligned}\ln\gamma_E&=x_W'^2\left[\tau_{WE}\left(\frac{G_{WE}}{x_E'+x_W'G_{WE}}\right)^2+\frac{\tau_{EW}G_{EW}}{(x_W'+x_E'G_{EW})^2}\right]\\"
        r"\ln\gamma_W&=x_E'^2\left[\tau_{EW}\left(\frac{G_{EW}}{x_W'+x_E'G_{EW}}\right)^2+\frac{\tau_{WE}G_{WE}}{(x_E'+x_W'G_{WE})^2}\right]\end{aligned}",
        None),
       ("NRTL parameters",
        r"\begin{gathered}\tau_{ij}=a_{ij}+\frac{b_{ij}}{T},\qquad G_{ij}=e^{-\alpha\tau_{ij}}\\"
        r"a_{EW}=-0.8009,\;\;a_{WE}=3.4578,\;\;b_{EW}=246.18\,\mathrm{K},\;\;b_{WE}=-586.08\,\mathrm{K},\;\;\alpha=0.3\end{gathered}",
        r"$x_E' = x_E/(x_E+x_W)$; the extract enters only by dilution of the true mole fractions."),
       ("Bubble point (solved by Newton iteration)",
        r"\sum_{i\in\{E,W\}} x_i\,\gamma_i(x',T)\,P_i^{\mathrm{sat}}(T)=P,\qquad y_i^*=\frac{x_i\gamma_iP_i^{\mathrm{sat}}}{P},\quad y_X^*=0",
        r"Converged to $|\Delta T|<10^{-9}$ K; $y^*$ renormalised so that $\sum_i y_i^*=1$."),
       ("Latent heats",
        r"\lambda_E=42300-70\,(T-298.15),\qquad \lambda_W=43990-44.5\,(T-298.15)\quad[\mathrm{J\,mol^{-1}}]", None),
       ("Enthalpies (ideal mixing, reference: liquid at 25 °C)",
        r"h_L=\sum_i x_i\,c_{p,i}\,(T-T_{\mathrm{ref}}),\qquad H_V=\sum_{i\in\{E,W\}} y_i\left[c_{p,i}(T-T_{\mathrm{ref}})+\lambda_i(T)\right]",
        r"$c_p$ = 112.4 / 75.3 / 428.0 J mol⁻¹ K⁻¹ (E/W/X). Heat of mixing neglected."),
       ("Liquid density and ABV (ideal volumes)",
        r"\frac{1}{\rho_m}=\sum_i\frac{x_iM_i}{\rho_i},\qquad \mathrm{ABV}=100\,\frac{x_EM_E/\rho_E}{\sum_ix_iM_i/\rho_i}",
        r"$\rho_i$ = 789 / 997 / 1590 kg m⁻³."),
       ("Liquid viscosity (Grunberg–Nissan + extract term)",
        r"\ln\mu_L=x_E'\ln\mu_E+x_W'\ln\mu_W+6.0\,x_E'x_W'\frac{298.15}{T}+170\,x_X",
        r"$\mu_W=10^{-3}\exp\!\left(-3.7188+\frac{578.919}{T-137.546}\right)$, "
        r"$\mu_E=1.2\times10^{-3}\exp\!\left[1900\left(\frac1T-\frac1{293.15}\right)\right]$ Pa s."),
       ("Surface tension",
        r"\sigma=\sigma_E+(\sigma_W-\sigma_E)\,e^{-10x_E'},\quad \sigma_W=0.0757-1.7\times10^{-4}\theta,\quad \sigma_E=0.0240-8.3\times10^{-5}\theta",
        None),
       ("Diffusivities and vapour properties",
        r"\begin{gathered}D_L=1.24\times10^{-9}\,\frac{T}{298.15}\,\frac{0.89\times10^{-3}}{\mu_L},\qquad "
        r"D_V=\frac{1.3\times10^{-5}}{P/\mathrm{bar}}\left(\frac{T}{298.15}\right)^{1.75}\\"
        r"\rho_V=\frac{P\,M_V}{RT},\qquad \mu_V=10^{-5}\,\mathrm{Pa\,s}\end{gathered}", None),
   ],
   nomenclature=[(r"$x_i, y_i$", "liquid / vapour mole fraction", "–"), (r"$\gamma_i$", "activity coefficient", "–"),
                 (r"$P_i^{sat}$", "vapour pressure", "bar"), (r"$h_L, H_V$", "molar enthalpy of liquid / vapour", "J/mol"),
                 (r"$\lambda_i$", "latent heat", "J/mol"), (r"$\rho_m$", "molar density", "mol/m³")],
   assumptions=["Ideal vapour phase (vacuum).", "Extract is non-volatile and thermodynamically inert except by dilution.",
                "Ideal liquid volumes and enthalpies (no excess volume, no heat of mixing ≤ 0.8 kJ/mol).",
                "Aroma compounds not modelled."],
   refs=["renon1968", "antoine"])

_m("feed", title="Beer feed (battery limit)", code="dynsim/equipment.py · FeedSource", group="Equipment",
   purpose="Converts beer specification (ABV, extract) to mole fractions at a fixed supply pressure and temperature.",
   equations=[("Composition per litre of beer",
               r"\begin{gathered}m_E=0.789\cdot10\,\mathrm{ABV},\qquad V_X=\frac{c_X}{1.59},\qquad m_W=0.997\,(1000-10\,\mathrm{ABV}-V_X)\\"
               r"x_i=\frac{m_i/M_i}{\sum_j m_j/M_j}\end{gathered}",
               r"$m$ in g, $V$ in mL, $c_X$ = extract in g/L."),
              ("Boundary conditions", r"P_{\mathrm{out}}=P_{\mathrm{supply}},\qquad T=T_{\mathrm{feed}}", None)],
   nomenclature=[(r"$c_X$", "extract concentration", "g/L")], assumptions=["Ideal volumes."], refs=[])

_m("packed_column", title="Packed column – balances, energy & pressure", code="dynsim/columns.py · PackedColumn",
   group="Separation",
   purpose="Dynamic packed bed discretised into N segments (1 = top), each a well-mixed liquid holdup in equilibrium "
           "with the vapour leaving it at the local pressure; vapour in plug flow; one lumped vapour space per column.",
   equations=[
       ("States", r"\mathbf{s}=\left\{n_{i,k}\;(i\in E,W,X;\;k=1\dots N[+\mathrm{sump}]),\;P_{\mathrm{top}},\;\Delta P_{\mathrm{bed}},\;y_v\right\}", None),
       ("Composition, local pressure and temperature",
        r"M_k=\sum_in_{i,k},\quad x_{i,k}=\frac{n_{i,k}}{M_k},\quad P_k=P_{\mathrm{top}}+\Delta P_{\mathrm{bed}}\frac{k-\tfrac12}{N},\quad T_k=T_{\mathrm{bub}}(\mathbf{x}_k,P_k)",
        r"The reboiler sump (if present) sits at $P_{\mathrm{top}}+\Delta P_{\mathrm{bed}}$."),
       ("Liquid hydraulics (trickle flow)",
        r"h_{L,k}=\frac{M_k}{\rho_{m,k}\,A\,\Delta z},\qquad u_{L,k}=h_L^{-1}(h_{L,k}),\qquad L_k=u_{L,k}\,A\,\rho_{m,k}",
        r"$h_L^{-1}$ = analytical inverse of the Billet–Schultes holdup correlation (PH-2)."),
       ("Component balance (segment k)",
        r"\frac{dn_{i,k}}{dt}=L_{k-1}x_{i,k-1}-L_kx_{i,k}+V_{k+1}y_{i,k+1}-V_ky_{i,k}+\sum_fF_fz_{i,f}",
        r"Reboiler sump: inflow $L_N$, outflows $B\,x_i$ and $V_s\,y^*_i$."),
       ("Rate-based segment efficiency",
        r"y_{i,k}=y_{i,k+1}+E_k\left(y^*_{i,k}-y_{i,k+1}\right),\qquad E_k=1-\exp\!\left(-\frac{\Delta z}{H_{OG,k}}\right)",
        r"Vapour plug flow across a well-mixed liquid; the sump is an equilibrium stage ($E=1$)."),
       ("Overall transfer unit",
        r"H_{OG}=H_G+\Lambda H_L,\qquad \Lambda=m\frac{V}{L},\qquad m=\frac{\partial y^*_E}{\partial x_E}\Big|_{P}",
        r"$m$ from a numerical perturbation of the bubble-point calculation; $V$ from the state-based estimate (PC-14)."),
       ("Stage enthalpy (liquid + packing/shell metal)",
        r"H_k=\sum_in_{i,k}\,h_i(T_k)+C_{m,k}T_k,\qquad C_{m,k}=c_{p,\mathrm{st}}\left(\rho_{\mathrm{bulk}}A\Delta z+\pi D\Delta z\,t_{\mathrm{sh}}\rho_{\mathrm{st}}\right)",
        None),
       ("Energy balance",
        r"\frac{dH_k}{dt}=\underbrace{L_{k-1}h_{L,k-1}+V_{k+1}H_{V,k+1}+\sum_fF_fh_{L,f}+Q_k-UA_k(T_k-T_{\mathrm{amb}})}_{\textstyle E_{\mathrm{in},k}-\,L_kh_{L,k}}-V_kH_{V,k}",
        r"$Q_k$ = reboiler duty in the sump, otherwise $0$."),
       (r"Index reduction ($T$ is algebraic: $T=T_{\mathrm{bub}}$)",
        r"\begin{aligned}\frac{dT_k}{dt}&=\sum_i\frac{\partial T_k}{\partial n_{i,k}}\frac{dn_{i,k}}{dt}+\frac{\partial T_k}{\partial P}\frac{dP}{dt}\\"
        r"\Rightarrow\quad V_k\Big(H_{V,k}-\sum_ig_{i,k}y_{i,k}\Big)&=E_{\mathrm{in},k}-L_kh_{L,k}-\sum_ig_{i,k}A_{i,k}-C_k\frac{\partial T_k}{\partial P}\frac{dP}{dt}\end{aligned}",
        r"$g_{i,k}=h_i(T_k)+C_k\,\partial T_k/\partial n_{i,k}$, $C_k=\sum_in_{i,k}c_{p,i}+C_{m,k}$, $A_{i,k}$ = all component flows except $V_ky_{i,k}$. "
        "Sensitivities from numerical perturbation of the bubble-point solve."),
       ("Bottom-up sweep", r"V_k=a_k+b_k\,\frac{dP_{\mathrm{top}}}{dt}\qquad(k=N_s,\dots,1)",
        r"Each stage equation is linear in its own $V_k$ and in $dP/dt$; the coefficients $a_k,b_k$ cascade upward."),
       ("Pressure dynamics (lumped vapour space)",
        r"\begin{gathered}n_v=\frac{P_{\mathrm{top}}V_{\mathrm{void}}}{RT_1},\qquad \frac{V_{\mathrm{void}}}{RT_1}\frac{dP_{\mathrm{top}}}{dt}=V_1-V_{\mathrm{exit}}\\"
        r"\Rightarrow\quad\frac{dP_{\mathrm{top}}}{dt}=\frac{a_1-V_{\mathrm{exit}}}{V_{\mathrm{void}}/(RT_1)-b_1}\end{gathered}",
        r"$V_{void}=\varepsilon AH+A\,h_{\mathrm{head}}+$ sump vapour space. $b_1\ll0$: flashing liquid makes the effective capacitance large."),
       ("Vapour-space composition", r"n_v\frac{dy_v}{dt}=\max(V_1,0)\,\left(y_{E,1}-y_v\right)", None),
       ("Bed pressure drop (fast lag)",
        r"\frac{d\Delta P_{\mathrm{bed}}}{dt}=\frac{1}{\tau_{\Delta P}}\left[\sum_k\left(\frac{\Delta P}{H}\right)_k\Delta z-\Delta P_{\mathrm{bed}}\right],\quad \tau_{\Delta P}=1\,\mathrm{s}",
        r"$(\Delta P/H)_k$ from Billet–Schultes with the actual $V_k$."),
       (r"Hydraulic vapour estimate (for $E_k$)",
        r"V_{\mathrm{est},k}=V_{\mathrm{exit}}+\left(V_{\mathrm{bot}}-V_{\mathrm{exit}}\right)\frac{k-\tfrac12}{N},\qquad V_{\mathrm{bot}}=\frac{Q_{\mathrm{reb}}}{\lambda(T_s,y^*_s)}\;\text{or}\;F_{\mathrm{vap,in}}",
        r"Keeps the system explicit (no algebraic loop between $E_k$ and $V_k$)."),
   ],
   nomenclature=[(r"$n_{i,k}$", "component holdup of stage k", "mol"), (r"$L_k, V_k$", "liquid / vapour molar flow leaving stage k", "mol/s"),
                 (r"$\Delta z, A, D$", "segment height, cross-section, diameter", "m, m², m"), (r"$E_k$", "Murphree vapour efficiency", "–"),
                 (r"$H_G, H_L, H_{OG}$", "heights of transfer units", "m"), (r"$C_{m,k}$", "metal heat capacity", "J/K"),
                 (r"$V_{void}$", "lumped vapour volume", "m³"), (r"$y_v$", "ethanol fraction in vapour space", "–")],
   assumptions=[r"Liquid on each segment well mixed and at its bubble point; vapour approaches equilibrium through $E_k$.",
                r"Vapour holdup lumped at the column top; segment pressures follow the top pressure ($dP_k/dt\approx dP_{\mathrm{top}}/dt$).",
                r"$dT_1/dt$ neglected in the vapour-space mole balance (source of the $\sim10^{-5}$ closure floor).",
                "No entrainment, weeping or flooding hydraulics; capacity reported as % flood.",
                "Feeds mix into the top segment; sub-cooled or superheated feeds condense or flash through the energy balance."],
   refs=["billet1999", "taylor1993"])

_m("packing_hydraulics", title="Packing hydraulics (holdup, pressure drop, capacity)", code="dynsim/packing.py",
   group="Separation", purpose="Billet–Schultes pre-loading holdup and irrigated pressure drop; F-factor capacity.",
   equations=[
       ("Liquid holdup (pre-loading)",
        r"h_L=\left(\frac{12\,\mu_La^2u_L}{g\rho_L}\right)^{1/3}\left(\frac{a_h}{a}\right)^{2/3},\qquad "
        r"\frac{a_h}{a}=\begin{cases}C_h\,\mathrm{Re}_L^{0.15}\mathrm{Fr}_L^{0.1}&\mathrm{Re}_L<5\\0.85\,C_h\,\mathrm{Re}_L^{0.25}\mathrm{Fr}_L^{0.1}&\mathrm{Re}_L\ge5\end{cases}",
        r"$\mathrm{Re}_L=u_L\rho_L/(a\mu_L)$, $\mathrm{Fr}_L=u_L^2a/g$."),
       ("Analytical inverse used for L(holdup)",
        r"h_L=K\,u_L^{\,p}\;\Rightarrow\;u_L=\left(\frac{h_L}{K}\right)^{1/p},\qquad p=0.6333\;(\mathrm{Re}_L\ge5),\;\;0.5667\;(\mathrm{Re}_L<5)",
        r"Continuous at $\mathrm{Re}_L=5$."),
       ("Dry pressure drop",
        r"\begin{gathered}\frac{\Delta P_0}{H}=\psi_0\frac{a}{\varepsilon^3}\frac{\rho_Vu_V^2}{2}\frac1K\\"
        r"\psi_0=C_p\left(\frac{64}{\mathrm{Re}_V}+\frac{1.8}{\mathrm{Re}_V^{0.08}}\right),\qquad "
        r"\mathrm{Re}_V=\frac{u_Vd_p\rho_VK}{(1-\varepsilon)\mu_V}\end{gathered}",
        r"$d_p=6(1-\varepsilon)/a$, wall factor $K^{-1}=1+\tfrac23\frac{1}{1-\varepsilon}\frac{d_p}{D}$."),
       ("Irrigated pressure drop",
        r"\frac{\Delta P}{H}=\frac{\Delta P_0}{H}\left(\frac{\varepsilon}{\varepsilon-h_L}\right)^{1.5}\exp\!\left(\frac{13300}{a^{1.5}}\sqrt{\mathrm{Fr}_L}\right)\left(1+2\varphi^8\right)",
        r"Last factor $(1+2\varphi^8)$: empirical loading-region rise (not part of Billet–Schultes)."),
       ("Capacity (fraction of flood)", r"\varphi=\frac{F_V}{F_{\max}},\qquad F_V=u_V\sqrt{\rho_V}", r"$F_{\max}$ is a packing constant."),
   ],
   nomenclature=[(r"$a,\varepsilon$", "specific area, void fraction", "m²/m³, –"), (r"$C_h, C_p$", "packing constants", "–"),
                 (r"$u_L, u_V$", "superficial velocities", "m/s"), (r"$h_L$", "fractional liquid holdup", "–"),
                 (r"$F_V$", "vapour F-factor", "Pa^0.5")],
   assumptions=["Pre-loading holdup correlation used at all loads.", "Constants are representative; replace with vendor data."],
   refs=["billet1999"])

_m("mt_structured", title="Mass transfer – structured packing", code="dynsim/packing.py · transfer_units (structured)",
   group="Separation", purpose="Rocha–Bravo–Fair (SRP) film coefficients with Tsai et al. effective area.",
   equations=[
       ("Effective velocities", r"u_{Ge}=\frac{u_V}{\varepsilon(1-h_L)\sin\theta},\qquad u_{Le}=\frac{u_L}{\varepsilon h_L\sin\theta}", None),
       ("Gas film", r"k_G=0.054\,\frac{D_V}{S}\left[\frac{(u_{Ge}+u_{Le})\rho_VS}{\mu_V}\right]^{0.8}\mathrm{Sc}_V^{0.33}", None),
       ("Liquid film (penetration)", r"k_L=2\sqrt{\frac{D_L\,C_E\,u_{Le}}{\pi S}},\qquad C_E=0.9", None),
       ("Effective area (Tsai)", r"\frac{a_e}{a}=\min\!\left\{1,\;1.34\left[\frac{\rho_L}{\sigma}g^{1/3}\left(\frac{u_L}{a}\right)^{4/3}\right]^{0.116}\right\}", None),
       ("Transfer units", r"H_G=\frac{u_V}{f_{MT}\,k_G\,a_e},\qquad H_L=\frac{u_L}{f_{MT}\,k_L\,a_e},\qquad \mathrm{HETP}=H_{OG}\frac{\ln\Lambda}{\Lambda-1}",
        r"$f_{MT}$ = user mass-transfer multiplier (calibration)."),
   ],
   nomenclature=[(r"$S,\theta$", "corrugation side, inclination", "m, °"), (r"$k_G,k_L$", "film coefficients", "m/s"),
                 (r"$a_e$", "effective interfacial area", "m²/m³"), (r"$\mathrm{Sc}_V$", r"$\mu_V/(\rho_VD_V)$", "–")],
   assumptions=["Binary-like transfer of ethanol represents the mixture.", r"Same $f_{MT}$ applied to $k_G$ and $k_L$."],
   refs=["rocha1996", "tsai2011"])

_m("mt_random", title="Mass transfer – random packing", code="dynsim/packing.py · transfer_units (random)",
   group="Separation", purpose="Onda–Takeuchi–Okumoto correlations for wetted area and film coefficients.",
   equations=[
       ("Wetted area",
        r"\frac{a_w}{a}=1-\exp\!\left[-1.45\left(\frac{\sigma_c}{\sigma}\right)^{0.75}\mathrm{Re}_L^{0.1}\mathrm{Fr}_L^{-0.05}\mathrm{We}_L^{0.2}\right]",
        r"$\mathrm{Re}_L=\frac{L_m}{a\mu_L}$, $\mathrm{Fr}_L=\frac{L_m^2a}{\rho_L^2g}$, $\mathrm{We}_L=\frac{L_m^2}{\rho_L\sigma a}$, $L_m=u_L\rho_L$; $a_e=\max(a_w,0.05a)$."),
       ("Liquid film", r"k_L\left(\frac{\rho_L}{\mu_Lg}\right)^{1/3}=0.0051\left(\frac{L_m}{a_w\mu_L}\right)^{2/3}\mathrm{Sc}_L^{-1/2}(a\,d_p)^{0.4}", None),
       ("Gas film", r"\frac{k_G}{a\,D_V}=C\left(\frac{G_m}{a\mu_V}\right)^{0.7}\mathrm{Sc}_V^{1/3}(a\,d_p)^{-2},\qquad C=5.23", r"$G_m=u_V\rho_V$."),
       ("Transfer units", r"H_G=\frac{u_V}{f_{MT}\,k_G\,a_e},\qquad H_L=\frac{u_L}{f_{MT}\,k_L\,a_e}", None),
   ],
   nomenclature=[(r"$\sigma_c$", "critical surface tension of packing", "N/m"), (r"$d_p$", "nominal packing size", "m")],
   assumptions=["Onda constants for rings > 15 mm."], refs=["onda1968"])

_m("vapour_line", title="Vapour line / nozzle", code="dynsim/equipment.py · VaporLine", group="Equipment",
   purpose="Pressure-driven vapour flow between column vapour spaces and the condenser.",
   equations=[("Regularised orifice equation",
               r"\dot m=C_dA\,\sqrt{2\rho_V}\,\frac{\Delta P}{\sqrt{|\Delta P|+1\,\mathrm{Pa}}}\;\approx\;C_dA\sqrt{2\rho_V\Delta P},\qquad F=\frac{\dot m}{M_V}",
               r"Bidirectional; $y$, $\rho_V$ and $M_V$ from the upstream vapour space.")],
   nomenclature=[(r"$C_dA$", "effective flow area", "m²")], assumptions=["Incompressible orifice form (ΔP ≪ P)."], refs=[])

_m("condenser", title="Vacuum condenser with non-condensables", code="dynsim/equipment.py · VacuumCondenser",
   group="Equipment", purpose="Gas-side mole balances with air accumulation, blanketing, and the vacuum-pump vent.",
   equations=[
       ("States and pressure", r"\{n_g,\;n_{\mathrm{air}},\;y_c\},\qquad P=\frac{n_gRT_g}{V_g},\quad T_g=T_{cw}+5\,\mathrm{K},\quad y_{\mathrm{air}}=\frac{n_{\mathrm{air}}}{n_g}", None),
       ("Condensing temperature", r"p_v=\max\{P(1-y_{\mathrm{air}}),\,2\,\mathrm{mbar}\},\qquad T_c=T_{\mathrm{bub}}\!\left([y_c,1-y_c,0],\,p_v\right)", None),
       ("Duty (ε-NTU, condensing side isothermal, air blanketing)",
        r"Q=C_{cw}\left[1-\exp\!\left(-\frac{UA\,(1-y_{\mathrm{air}})^2}{C_{cw}}\right)\right]\,\mathrm{softplus}(T_c-T_{cw}),\qquad N_c=\frac{Q}{\lambda(T_c,y_c)}",
        r"$C_{cw}=\dot m_{cw}c_{p,w}$."),
       ("Vent (vacuum pump)", r"\dot n_{\mathrm{pump}}=S(P)\frac{P}{RT_g},\qquad \dot n_{\mathrm{EtOH,vent}}=\dot n_{\mathrm{pump}}(1-y_{\mathrm{air}})\,y_c", None),
       ("Balances",
        r"\begin{aligned}\frac{dn_g}{dt}&=V_{\mathrm{in}}+\dot n_{\mathrm{air,in}}-N_c-\dot n_{\mathrm{pump}}\\"
        r"\frac{dn_{\mathrm{air}}}{dt}&=\dot n_{\mathrm{air,in}}-\dot n_{\mathrm{pump}}\,y_{\mathrm{air}}\\"
        r"(n_g-n_{\mathrm{air}})\frac{dy_c}{dt}&=\max(V_{\mathrm{in}},0)\,(y_{\mathrm{in}}-y_c)\end{aligned}",
        r"$\dot n_{\mathrm{air,in}}$ = lumped leakage (kg/h ÷ $M_{\mathrm{air}}$) + bleed flow."),
   ],
   nomenclature=[(r"$y_c$", "ethanol fraction of condensable gas", "–"), (r"$S$", "pump suction capacity", "m³/s"),
                 (r"$N_c$", "condensation rate", "mol/s")],
   assumptions=["Total condensation at the condensate bubble point (vent gas has condensate composition).",
                "All air leakage lumped at the condenser.", r"Empirical blanketing factor $(1-y_{\mathrm{air}})^2$."],
   refs=[])

_m("vacuum_pump", title="Liquid-ring vacuum pump & air bleed", code="dynsim/equipment.py · VacuumPump, AirBleed",
   group="Equipment", purpose="Suction capacity curve with VFD speed dynamics; atmospheric air bleed for pressure control.",
   equations=[("Capacity", r"S=S_{\max}\,n\;\mathrm{softplus}\!\left(1-\left(\frac{P_{\mathrm{ult}}}{P}\right)^2\right),\qquad \frac{dn}{dt}=\frac{n_{\mathrm{cmd}}-n}{\tau}", r"$n_{\mathrm{cmd}}=0$ when tripped."),
              ("Air bleed (choked gas valve)", r"\dot n_{\mathrm{bleed}}=K_v\,f(u)\,P_{\mathrm{atm}}\sqrt{1-\left(\frac{P}{P_{\mathrm{atm}}}\right)^2}", None)],
   nomenclature=[(r"$P_{ult}$", "ultimate pressure (seal-liquid vapour pressure)", "bar")], assumptions=[], refs=[])

_m("tank", title="Liquid vessel (sump, reflux drum)", code="dynsim/equipment.py · LiquidTank", group="Equipment",
   purpose="Well-mixed liquid holdup with component and energy balances; liquid may be sub-cooled.",
   equations=[("Component balance", r"\frac{dn_i}{dt}=\sum_{\mathrm{in}}F\,z_i-F_{\mathrm{out}}\,x_i", None),
              ("Energy balance", r"\left(\sum_in_ic_{p,i}+C_m\right)\frac{dT}{dt}=\sum_{\mathrm{in}}F\,(\mathbf z\cdot\mathbf c_p)\,(T_{\mathrm{in}}-T)-UA\,(T-T_{\mathrm{amb}})", r"Follows from $\frac{d}{dt}\sum_i n_ih_i$ with ideal enthalpies."),
              ("Level and outlet pressure", r"\ell=\frac{\sum_in_i}{\rho_mA},\qquad P_{\mathrm{out}}=P+\rho g\ell", None)],
   nomenclature=[(r"$\ell$", "liquid level", "m")], assumptions=["Constant cross-section (vertical vessel equivalent)."], refs=[])

_m("pump_line", title="Liquid line: pump + valve hydraulics", code="dynsim/equipment.py · LiquidLine, CentrifugalPump",
   group="Equipment", purpose="Closed-form pressure balance source → pump → control valve → destination.",
   equations=[("Pressure balance",
               r"P_{\mathrm{src}}+\Delta P_0n^2-k_pQ^2-\mathrm{SG}\left(\frac{Q}{K_vf(u)}\right)^2-k_{\mathrm{line}}Q^2-\frac{\rho g\Delta z}{10^5}=P_{\mathrm{dest}}", "Q in m³/h, pressures in bar."),
              ("Solution", r"Q=\sqrt{\frac{\mathrm{softplus}(\mathrm{drive})}{k_p+\mathrm{SG}/(K_vf)^2+k_{\mathrm{line}}}}\;\cdot\;s\!\left(\frac{\ell_{\mathrm{src}}}{\ell_{\min}}-1\right)",
               r"Non-return valve (no back-flow); $s$ = smoothstep loss of suction near $\ell_{\min}=0.05$ m."),
              ("Pump curve (affinity laws)", r"\Delta P_{\mathrm{pump}}=\Delta P_0n^2-k_pQ^2,\quad k_p=\frac{\Delta P_0-\Delta P_{\mathrm{rated}}}{Q_{\mathrm{rated}}^2},\quad \frac{dn}{dt}=\frac{n_{\mathrm{cmd}}-n}{\tau}", None),
              ("Molar flow and power", r"F=\frac{Q}{3600}\rho_m,\qquad W=\frac{Q\,\Delta P_{\mathrm{pump}}}{\eta}", None)],
   nomenclature=[(r"$Q$", "volumetric flow", "m³/h"), (r"$n$", "pump speed fraction", "–")], assumptions=["No line holdup (instantaneous hydraulics)."], refs=[])

_m("valve", title="Control valve", code="dynsim/equipment.py · Valve", group="Equipment",
   purpose="Inherent characteristic and first-order actuator.",
   equations=[("Characteristic", r"f(u)=\begin{cases}u&\text{linear}\\ \dfrac{R^u-1}{R-1},\;R=30&\text{equal \% (modified)}\\ \sqrt u&\text{quick opening}\end{cases}", None),
              ("Actuator", r"\frac{du}{dt}=\frac{\mathrm{clip}(u_{\mathrm{cmd}},0,1)-u}{\tau}", r"$du/dt=0$ when the stem is set stuck."),
              ("Auto-sizing (Kv = 0)", r"K_v=\frac{Q_{\mathrm{design}}}{f(u_{\mathrm{design}})\sqrt{\Delta P_{\mathrm{design}}/\mathrm{SG}}}", None)],
   nomenclature=[(r"$u$", "stem position", "–")], assumptions=[], refs=[])

_m("hx_regen", title="Regenerative plate exchanger", code="dynsim/equipment.py · CounterflowHX", group="Equipment",
   purpose="Feed/product counter-current exchanger with outlet-temperature lags.",
   equations=[("Effectiveness", r"\varepsilon=\frac{1-e^{-\mathrm{NTU}(1-C_r)}}{1-C_re^{-\mathrm{NTU}(1-C_r)}}\;(C_r<1),\qquad \varepsilon=\frac{\mathrm{NTU}}{1+\mathrm{NTU}}\;(C_r\to1)", r"$\mathrm{NTU}=UA/C_{min}$, $C_r=C_{min}/C_{max}$, $C=F\,\mathbf z\cdot\mathbf c_p$."),
              ("Duty and outlet dynamics", r"Q=\varepsilon\,C_{\min}(T_{h,\mathrm{in}}-T_{c,\mathrm{in}}),\qquad \tau\frac{dT_{c,\mathrm{out}}}{dt}=T_{c,\mathrm{in}}+\frac{Q}{C_c}-T_{c,\mathrm{out}}", r"Analogous for the hot side; $\tau=30$ s.")],
   nomenclature=[], assumptions=["Plate/fluid inertia lumped in a first-order lag."], refs=[])

_m("hx_trim", title="Hot-water trim heater", code="dynsim/equipment.py · HotWaterHeater", group="Equipment",
   purpose="Process stream heated by hot water whose flow is set by HV-102.",
   equations=[("Hot-water capacity", r"C_{hw}=f(u)\,\dot m_{hw,\max}\,c_{p,w}", None),
              ("Duty and outlet", r"Q=\varepsilon(\mathrm{NTU},C_r)\,C_{\min}(T_{hw}-T_{\mathrm{in}}),\qquad \tau\frac{dT_{\mathrm{out}}}{dt}=T_{\mathrm{in}}+\frac{Q}{C_p}-T_{\mathrm{out}}", r"Counter-flow $\varepsilon$ as for the regenerative exchanger; $\tau=20$ s.")],
   nomenclature=[], assumptions=[], refs=[])

_m("reboiler", title="Hot-water reboiler", code="dynsim/equipment.py · HotWaterReboiler", group="Equipment",
   purpose="Kettle reboiler duty from a hot-water circuit (boiling side isothermal).",
   equations=[("Duty", r"Q^*=C_{hw}\left(1-e^{-UA/C_{hw}}\right)\max(T_{hw}-T_B,0),\qquad C_{hw}=f(u)\,\dot m_{hw,\max}c_{p,w}", None),
              ("Thermal lag", r"\tau\frac{dQ}{dt}=Q^*-Q", r"$Q$ enters the C-100 sump energy balance as $Q_k$ (PC-8).")],
   nomenclature=[(r"$T_B$", "sump (boiling) temperature", "K")], assumptions=["Lag lumps tube-wall and hot-water inertia."], refs=[])

_m("pid", title="PID controller", code="dynsim/control.py · PID", group="Control",
   purpose="ISA-form PID with span-normalised error, derivative on filtered PV, anti-windup and AUTO/MAN.",
   equations=[("Error", r"e=s\,\frac{\mathrm{SP}-\mathrm{PV}}{\mathrm{span}},\qquad s=\begin{cases}+1&\text{reverse}\\-1&\text{direct}\end{cases}", None),
              ("Control law", r"u=b+K_c\left(e+\frac{I}{T_i}\right)-s\,K_cT_d\,\frac{1}{\mathrm{span}}\frac{\mathrm{PV}-\mathrm{PV}_f}{T_d/N},\qquad \mathrm{OP}=\mathrm{clip}(u,0,1)", None),
              ("Integral with anti-windup", r"\frac{dI}{dt}=\begin{cases}0&\text{MAN, or }(u>1\wedge e>0)\text{ or }(u<0\wedge e<0)\\ e&\text{otherwise}\end{cases}", None),
              ("Derivative filter", r"\frac{d\mathrm{PV}_f}{dt}=\frac{\mathrm{PV}-\mathrm{PV}_f}{T_d/N},\quad N=10", None),
              ("Bumpless transfer", r"\text{MAN}\to\text{AUTO}:\;b\leftarrow\mathrm{OP},\;I\leftarrow0", None)],
   nomenclature=[(r"$K_c, T_i, T_d$", "gain, integral, derivative time", "–, s, s"), (r"$b$", "bias", "–")], assumptions=[], refs=[])

_m("analyser", title="Online analyser", code="dynsim/control.py · Analyzer", group="Control",
   purpose="Measurement dynamics as n first-order lags (default n = 2).",
   equations=[("Lags in series", r"\frac{\tau}{n}\frac{dm_1}{dt}=y-m_1,\qquad \frac{\tau}{n}\frac{dm_j}{dt}=m_{j-1}-m_j,\qquad \text{reading}=m_n", None)],
   nomenclature=[], assumptions=["No sample transport dead time (approximated by the lags)."], refs=[])

_m("balance", title="Conservation monitor", code="dynsim/control.py · BalanceMonitor", group="Numerics",
   purpose="Independent check of mass conservation and integration accuracy.",
   equations=[("Integrated boundary flows", r"\frac{d\mathcal I_i}{dt}=\sum_{\mathrm{feeds}}Fz_i,\qquad \frac{d\mathcal O_i}{dt}=\sum_{\mathrm{products+vent}}Fz_i", None),
              ("Closure", r"\epsilon_i(t)=\frac{\mathrm{inv}_i(t)-\mathrm{inv}_i(0)-\left(\mathcal I_i-\mathcal O_i\right)}{\max\{\mathrm{inv}_i(0),\mathcal I_i\}}", r"$\mathrm{inv}_i$ = all liquid holdups + vapour-space and condenser condensables.")],
   nomenclature=[], assumptions=[], refs=[])

_m("numerics", title="Solution method", code="dynsim/core.py · Flowsheet", group="Numerics",
   purpose="How the coupled model is assembled and integrated.",
   equations=[("Semi-explicit DAE reduced to ODE", r"\dot{\mathbf s}=\mathbf f\big(t,\mathbf s,\mathbf a(\mathbf s)\big),\qquad \mathbf a=\text{algebraic outputs evaluated in causal order}",
               r"Flows leaving holdups depend only on states $\mathbf s$; bubble points and hydraulics are solved inside each evaluation of $\mathbf f$."),
              ("Variable-order BDF/NDF (orders 1–5)", r"\sum_{j=0}^{k}\alpha_j\mathbf s_{n-j}=h\,\gamma_k\,\mathbf f(t_n,\mathbf s_n)\quad\text{(Newton with finite-difference Jacobian)}", None),
              ("Error control", r"\left\|\frac{\mathbf e_n}{\mathrm{atol}+\mathrm{rtol}\,|\mathbf s_n|}\right\|_{\mathrm{RMS}}\le1,\qquad \mathbf e_n\approx C_k\left(\mathbf s_n-\mathbf s_n^{\mathrm{pred}}\right)", "Reported per step in the solver-accuracy panel."),
              ("Steady-state initialisation", r"\text{pseudo-transient } \dot{\mathbf s}=\mathbf f \text{ over a horizon, then Newton (trust-region) on } \mathbf f(\mathbf s)=0", r"Converged when $\max_j\,|f_j|/\max(|s_j|,\,10^3\,\mathrm{atol}_j)<0.05\ \mathrm{h^{-1}}$."),
              ("Schedule", r"\mathbf p(t^+)=\mathbf p(t^-)+\Delta\mathbf p\;\text{at event times; integration restarted}", "Parameters restored after each run.")],
   nomenclature=[], assumptions=[], refs=["shampine1997"])

REFS = {
    "renon1968": "Renon, H. & Prausnitz, J.M. (1968). Local compositions in thermodynamic excess functions for liquid mixtures. AIChE J. 14, 135–144.",
    "antoine": "Antoine-equation constants for ethanol and water (mmHg, °C form) as tabulated in standard property handbooks.",
    "billet1999": "Billet, R. & Schultes, M. (1999). Prediction of mass transfer columns with dumped and arranged packings. Trans IChemE 77A, 498–504.",
    "rocha1996": "Rocha, J.A., Bravo, J.L. & Fair, J.R. (1996). Distillation columns containing structured packings: a comprehensive model for their performance. 2. Mass-transfer model. Ind. Eng. Chem. Res. 35, 1660–1667.",
    "tsai2011": "Tsai, R.E., Seibert, A.F., Eldridge, R.B. & Rochelle, G.T. (2011). A dimensionless model for predicting the mass-transfer area of structured packing. AIChE J. 57, 1173–1184.",
    "onda1968": "Onda, K., Takeuchi, H. & Okumoto, Y. (1968). Mass transfer coefficients between gas and liquid phases in packed columns. J. Chem. Eng. Japan 1, 56–62.",
    "taylor1993": "Taylor, R. & Krishna, R. (1993). Multicomponent Mass Transfer. Wiley (HTU/NTU and segment-efficiency concepts).",
    "shampine1997": "Shampine, L.F. & Reichelt, M.W. (1997). The MATLAB ODE suite. SIAM J. Sci. Comput. 18, 1–22 (NDF/BDF basis of SciPy's BDF).",
}

GROUPS = ["Overview", "Properties", "Separation", "Equipment", "Control", "Numerics"]


def models_for_unit(tag, cfg):
    """Model keys that define the behaviour of a P&ID tag."""
    def mt(sec):
        return "mt_structured" if PACKINGS[cfg[sec]["packing"]]["kind"] == "structured" else "mt_random"
    if tag == "C-100":
        return ["packed_column", "packing_hydraulics", mt("C100"), "thermo"]
    if tag == "C-200":
        return ["packed_column", "packing_hydraulics", mt("C200"), "thermo"]
    table = {"FEED": ["feed", "thermo"], "T-200": ["tank"], "D-201": ["tank"], "E-101": ["hx_regen"],
             "E-102": ["hx_trim", "valve"], "E-103": ["reboiler"], "E-201": ["condenser", "thermo"],
             "K-300": ["vacuum_pump"], "PV-300": ["vacuum_pump", "valve"], "V-100": ["vapour_line"],
             "V-200": ["vapour_line"], "AT-103": ["analyser"], "BAL": ["balance"]}
    if tag in table:
        return table[tag]
    if tag.startswith(("FV", "LV", "TV", "HV")):
        return ["valve", "pump_line"] if not tag.startswith("HV") else ["valve"]
    if tag.startswith("P-") or tag.startswith("L-"):
        return ["pump_line", "valve"]
    if tag[:3] in ("FIC", "TIC", "LIC", "AIC", "PIC"):
        return ["pid"]
    return []


def packing_table(cfg, sec):
    pk = PACKINGS[cfg[sec]["packing"]]
    keys = [("a", "a [m²/m³]"), ("eps", "ε [–]"), ("Ch", "Cₕ [–]"), ("Cp", "Cₚ [–]"), ("Fmax", "F_max [Pa½]"),
            ("S", "S, corrugation side [m]"), ("theta", "θ [°]"), ("dp", "dₚ [m]"), ("sigma_c", "σ꜀ critical [N/m]"),
            ("rho_bulk", "ρ bulk [kg/m³]")]
    return [(lab, pk[k]) for k, lab in keys if k in pk]


# =============================================================================== verification
def run_checks(cfg=None, fs=None, X=None, results=None):
    """Return a list of verification checks: name, result, criterion, status."""
    from dynsim import thermo as th, packing as pkg
    out = []

    def add(name, value, crit, ok, what):
        out.append({"Check": name, "What it verifies": what, "Result": value, "Criterion": crit,
                    "Status": "✅ pass" if ok else "⚠️ review"})

    xs = np.linspace(0.80, 0.97, 1701)
    T, y, _ = th.bubble_point(np.column_stack([xs, 1 - xs, 0 * xs]), 1.01325)
    i = int(np.argmin(np.abs(y[:, 0] - xs)))
    add("Ethanol–water azeotrope @ 1 atm", f"x = {xs[i]:.3f}, T = {T[i]-273.15:.2f} °C", "lit. 0.894 / 78.15 °C (±0.01, ±0.3 K)",
        abs(xs[i] - 0.894) < 0.01 and abs(T[i] - 273.15 - 78.15) < 0.3, "NRTL + Antoine VLE")
    Tw = th.bubble_point(np.array([[0.0, 1.0, 0.0]]), 1.01325)[0][0] - 273.15
    add("Pure water boiling point @ 1 atm", f"{Tw:.2f} °C", "100.00 ± 0.2 °C", abs(Tw - 100) < 0.2, "Antoine constants")
    Te = th.bubble_point(np.array([[1.0, 0.0, 0.0]]), 1.01325)[0][0] - 273.15
    add("Pure ethanol boiling point @ 1 atm", f"{Te:.2f} °C", "78.29 ± 0.3 °C", abs(Te - 78.29) < 0.3, "Antoine constants")
    err = 0.0
    for pk in PACKINGS.values():
        u = np.geomspace(1e-5, 2e-2, 60)
        h = pkg.holdup(pk, u, 1000.0, 6e-4)
        err = max(err, float(np.max(np.abs(pkg.uL_from_holdup(pk, h, 1000.0, 6e-4) / u - 1))))
    add("Holdup correlation inverse", f"max rel. error {err:.1e}", "< 1e-10", err < 1e-10, "Liquid outflow from holdup is the exact inverse of the holdup correlation")
    xb = th.beer_composition(5.0, 50.0)
    add("Beer composition round-trip", f"{th.abv(xb)[0]:.6f} % ABV", "5.000000", abs(th.abv(xb)[0] - 5) < 1e-9, "ABV ↔ mole fractions")
    if fs is not None and X is not None:
        f1 = fs.rhs(0.0, X.copy())
        f2 = fs.rhs(0.0, X.copy())
        rel = float(np.max(np.abs(f1 - f2) / (np.abs(f1) + fs.atol())))
        add("Reproducible right-hand side", f"max rel. |Δf| = {rel:.1e}", "< 1e-6",
            rel < 1e-6, "Repeated RHS calls agree (bubble-point Newton is warm-started and converged to 1e-9 K, "
            "so results agree to round-off rather than bit-for-bit)")
        mask = np.ones(fs.n, bool)
        for u in fs.units:
            if u.steady_exclude:
                mask[u.sl] = False
        r = float(np.max((np.abs(f1) / np.maximum(np.abs(X), fs.atol() * 1e3))[mask]) * 3600)
        add("Initial state is a steady state", f"max rel. rate {r:.1e} 1/h", "< 0.05 1/h", r < 0.05, "Initialisation converged")
        # C-100 overall energy balance at steady state (exercises the index-reduced energy equation)
        from dynsim.thermo import h_liq, h_vap
        c1 = fs["C-100"]
        fs.rhs(0.0, X)
        e_in = sum(ref()["F"] * float(h_liq(ref()["T"], ref()["z"])) for _, ref in c1.feeds) + fs["E-103"].out["Q"]
        prod = fs["L-product"].out["stream"]
        vap = fs["V-100"].out["stream"]
        e_out = prod["F"] * float(h_liq(prod["T"], prod["z"])) + vap["F"] * float(h_vap(vap["T"], vap["z"]))
        st = c1._st
        e_out += float(np.sum(c1.UAl * (st["T"] - c1.T_amb)))
        rel = abs(e_in - e_out) / max(abs(e_in), 1.0)
        add("C-100 steady-state energy balance", f"in {e_in/1e3:.1f} kW, out {e_out/1e3:.1f} kW, rel. {rel:.1e}",
            "< 1e-3", rel < 1e-3, "Index-reduced dynamic energy balance closes at steady state")
    if results is not None and len(results.get("df", [])) and "BAL.closure_EtOH" in results["df"]:
        df = results["df"]
        ce, ct = df["BAL.closure_EtOH"].abs().max(), df["BAL.closure_total"].abs().max()
        add("Mass conservation over last run", f"EtOH {ce:.1e}, total {ct:.1e}", "< 1e-4", max(ce, ct) < 1e-4,
            "Integrated in − out = Δ inventory")
        sdf = results.get("sdf")
        if sdf is not None and len(sdf):
            frac = float((sdf["err"].dropna() <= 1.0 + 1e-6).mean())
            add("Accepted steps within tolerance", f"{frac:.1%} of {len(sdf)} steps", "≥ 99 %", frac >= 0.99,
                "A-posteriori local error estimate ≤ 1")
    return out


def _eq_prefix(key):
    return "".join(w[0] for w in key.split("_")).upper()


def markdown_document():
    """Whole model documentation as Markdown with $$ LaTeX $$ blocks."""
    L = ["# dynsim de-alcoholisation model – equation reference", ""]
    for g in GROUPS[1:]:
        L += [f"## {g}", ""]
        for key, m in MODELS.items():
            if m["group"] != g:
                continue
            L += [f"### {m['title']}", f"*Source:* `{m['code']}`", "", m["purpose"], ""]
            for i, (h, tex, note) in enumerate(m["equations"], 1):
                L += [f"**({_eq_prefix(key)}-{i}) {h}**", "", "$$", tex, "$$", ""]
                if note:
                    L += [f"> {note}", ""]
            if m["nomenclature"]:
                L += ["| Symbol | Meaning | Unit |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in m["nomenclature"]] + [""]
            if m["assumptions"]:
                L += ["*Assumptions:*"] + [f"- {a}" for a in m["assumptions"]] + [""]
            if m["refs"]:
                L += ["*References:* " + "; ".join(REFS[r] for r in m["refs"]), ""]
    L += ["## References", ""] + [f"- {v}" for v in REFS.values()]
    return "\n".join(L)
