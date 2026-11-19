"""
Thermodynamics for beer modelled as ethanol / water / extract.

* Ethanol-water non-ideality: NRTL (Aspen-style parameters, tau_ij = a_ij + b_ij/T)
* Vapour pressures: Antoine (mmHg, degC form), valid ~0-100 degC
* Extract (sugars, dextrins, proteins) is a non-volatile pseudo-component with the
  molar mass of maltose. It dilutes the liquid (lowers water/ethanol activity via
  mole fraction) but never enters the vapour.

Units used throughout the package: T [K], P [bar], flows [mol/s], energy [J].
"""
import numpy as np

COMPONENTS = ("ethanol", "water", "extract")
E, W, X = 0, 1, 2
MW = np.array([46.068, 18.015, 342.3])      # g/mol
RHO = np.array([789.0, 997.0, 1590.0])      # kg/m3 pure liquid (extract = apparent)
G = 9.81

# Antoine: log10(P/mmHg) = A - B/(T/degC + C)
_ANT = np.array([[8.20417, 1642.89, 230.300],     # ethanol
                 [8.07131, 1730.63, 233.426]])    # water
_MMHG = 1.0 / 750.0616                           # bar per mmHg

# NRTL ethanol(1) - water(2)
NRTL_A12, NRTL_A21 = -0.8009, 3.4578
NRTL_B12, NRTL_B21 = 246.18, -586.0809
NRTL_ALPHA = 0.3


def psat(T):
    """Vapour pressure [bar] of ethanol, water and dP/dT [bar/K]. T: array (n,)."""
    Tc = np.asarray(T, dtype=float)[..., None] - 273.15
    A, B, C = _ANT[:, 0], _ANT[:, 1], _ANT[:, 2]
    P = 10.0 ** (A - B / (Tc + C)) * _MMHG
    dP = P * np.log(10.0) * B / (Tc + C) ** 2
    return P, dP


def gamma(x1, T):
    """NRTL activity coefficients for ethanol(1)/water(2) on an extract-free basis."""
    g1, g2 = _gamma2(x1, T)
    return np.stack([g1, g2], axis=-1)


def _gamma2(x1, T):
    x2 = 1.0 - x1
    t12 = NRTL_A12 + NRTL_B12 / T
    t21 = NRTL_A21 + NRTL_B21 / T
    G12 = np.exp(-NRTL_ALPHA * t12)
    G21 = np.exp(-NRTL_ALPHA * t21)
    d1 = x1 + x2 * G21
    d2 = x2 + x1 * G12
    r1, r2 = G21 / d1, G12 / d2
    ln1 = x2 * x2 * (t21 * r1 * r1 + t12 * G12 / (d2 * d2))
    ln2 = x1 * x1 * (t12 * r2 * r2 + t21 * G21 / (d1 * d1))
    return np.exp(ln1), np.exp(ln2)


_LN10 = np.log(10.0)


def _psat2(T):
    Tc = T - 273.15
    d1, d2 = Tc + 230.300, Tc + 233.426
    p1 = 10.0 ** (8.20417 - 1642.89 / d1) * _MMHG
    p2 = 10.0 ** (8.07131 - 1730.63 / d2) * _MMHG
    return p1, p2, p1 * _LN10 * 1642.89 / (d1 * d1), p2 * _LN10 * 1730.63 / (d2 * d2)


def latent_heat(T):
    """Heat of vaporisation [J/mol] of ethanol, water (linear fits 25-100 degC)."""
    T = np.asarray(T, dtype=float)
    return np.stack([42300.0 - 70.0 * (T - 298.15),
                     43990.0 - 44.5 * (T - 298.15)], axis=-1)


def bubble_point(x, P, T0=None, tol=1e-9, maxit=50):
    """
    Bubble temperature and equilibrium vapour for liquid mole fractions x (n,3) at P [bar].
    Vectorised Newton iteration. Returns T (n,), y* (n,3), gamma (n,2).
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    n = x.shape[0]
    T = np.full(n, 330.0) if T0 is None else np.array(T0, dtype=float, copy=True)
    xe, xw = x[:, E], x[:, W]
    x1sf = np.clip(xe / np.maximum(xe + xw, 1e-12), 0.0, 1.0)   # extract-free basis
    for _ in range(maxit):
        g1, g2 = _gamma2(x1sf, T)
        p1, p2, dp1, dp2 = _psat2(T)
        a1, a2 = xe * g1, xw * g2
        f = a1 * p1 + a2 * p2 - P
        dT = -f / np.maximum(a1 * dp1 + a2 * dp2, 1e-12)
        np.clip(dT, -15.0, 15.0, out=dT)
        T += dT
        if np.max(np.abs(dT)) < tol:
            break
    g1, g2 = _gamma2(x1sf, T)
    p1, p2, _, _ = _psat2(T)
    g = np.stack([g1, g2], axis=-1)
    y = np.zeros_like(x)
    y[:, E] = xe * g1 * p1 / P
    y[:, W] = xw * g2 * p2 / P
    s = y[:, E] + y[:, W]
    y[:, :2] /= s[:, None]                     # remove residual Newton error
    return T, y, g


def liquid_props(x):
    """Molar density [mol/m3], mass density [kg/m3], molar mass [kg/mol] (ideal volume)."""
    x = np.atleast_2d(x)
    mw = x @ MW / 1000.0                       # kg/mol
    vm = x @ (MW / 1000.0 / RHO)               # m3/mol
    return 1.0 / vm, mw / vm, mw


def abv(x):
    """Alcohol by volume [% v/v] from mole fractions (ideal mixing volume)."""
    x = np.atleast_2d(x)
    vol = x * MW / RHO
    return 100.0 * vol[:, E] / vol.sum(axis=1)


def beer_composition(abv_pct, extract_gL):
    """Mole fractions of a beer with given ABV [%] and extract content [g/L]."""
    V_e = abv_pct / 100.0 * 1000.0             # mL ethanol per litre
    m_e = V_e * RHO[E] / 1000.0                # g
    V_x = extract_gL / (RHO[X] / 1000.0)       # mL
    m_w = (1000.0 - V_e - V_x) * RHO[W] / 1000.0
    n = np.array([m_e, m_w, extract_gL]) / MW
    return n / n.sum()


# =============================================================================
# Caloric properties (ideal mixing, reference: liquid at 25 degC)
# =============================================================================
TREF = 298.15
R_GAS = 8.314
CP_L = np.array([112.4, 75.3, 428.0])     # J/mol/K liquid (extract ~ maltose solution)
MW_AIR = 28.96


def h_liq(T, x):
    """Molar liquid enthalpy [J/mol]; T (n,), x (n,3) or scalar/1-D."""
    return (np.asarray(x) @ CP_L) * (np.asarray(T) - TREF)


def hv_comp(T):
    """Per-component saturated-vapour enthalpies [J/mol] (n,3); extract -> 0 (never vaporises)."""
    T = np.asarray(T, dtype=float)
    lam = latent_heat(T)
    hl = CP_L[None, :] * (T[..., None] - TREF) if T.ndim else CP_L * (T - TREF)
    out = np.array(hl, dtype=float, copy=True)
    out[..., :2] += lam
    out[..., 2] = 0.0
    return out


def h_vap(T, y):
    return np.sum(hv_comp(T) * np.asarray(y), axis=-1)


# =============================================================================
# Transport properties (engineering correlations, ~0-100 degC)
# =============================================================================
def mu_liq(T, x):
    """Liquid viscosity [Pa s]: Grunberg-Nissan ethanol/water + extract factor."""
    x = np.atleast_2d(x)
    T = np.asarray(T, dtype=float)
    muW = 1e-3 * np.exp(-3.7188 + 578.919 / (T - 137.546))        # Vogel, water
    muE = 1.2e-3 * np.exp(1900.0 * (1.0 / T - 1.0 / 293.15))      # ethanol
    s = np.maximum(x[:, E] + x[:, W], 1e-12)
    e, w = x[:, E] / s, x[:, W] / s
    lnmu = e * np.log(muE) + w * np.log(muW) + e * w * 6.0 * (298.15 / T) + 170.0 * x[:, X]
    return np.exp(lnmu)


def sigma_liq(T, x):
    """Surface tension [N/m] (strongly non-linear in ethanol)."""
    x = np.atleast_2d(x)
    Tc = np.asarray(T, dtype=float) - 273.15
    sW = 0.0757 - 1.7e-4 * Tc
    sE = 0.0240 - 8.3e-5 * Tc
    s = np.maximum(x[:, E] + x[:, W], 1e-12)
    return sE + (sW - sE) * np.exp(-10.0 * x[:, E] / s)


def D_liq(T, mu):
    """Liquid diffusivity of ethanol [m2/s] (Stokes-Einstein scaled)."""
    return 1.24e-9 * (np.asarray(T) / 298.15) * (0.89e-3 / mu)


def D_vap(T, P):
    """Ethanol-water vapour diffusivity [m2/s], P in bar."""
    return 1.3e-5 * (np.asarray(T) / 298.15) ** 1.75 / np.asarray(P)


MU_VAP = 1.0e-5   # Pa s


def rho_vap(T, P, y):
    """Vapour density [kg/m3] and molar mass [kg/mol] (ideal gas)."""
    mw = np.asarray(y) @ MW / 1000.0
    return np.asarray(P) * 1e5 * mw / (R_GAS * np.asarray(T)), mw
