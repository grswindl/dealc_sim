"""
Packing library and packed-bed correlations (vectorised over bed segments).

Hydraulics (both packing families) - Billet & Schultes (1999):
    * pre-loading liquid holdup  h_L(u_L)  and its analytical inverse u_L(h_L)
    * dry + irrigated pressure drop, with an empirical loading-region enhancement
Mass transfer
    * structured: Rocha-Bravo-Fair (SRP, 1996) k_G, k_L; Tsai et al. (2011) effective area
    * random    : Onda-Takeuchi-Okumoto (1968) k_G, k_L and wetted area
Capacity
    * flooding approximated by the vapour F-factor relative to a packing F_max

Constants are representative literature/vendor-style values. Replace them with your
supplier's data (or fit `mt_factor` / Ch / Cp to plant data) before using the model
for design.
"""
import numpy as np

G = 9.81

PACKINGS = {
    "Mellapak 250Y (structured)": dict(
        kind="structured", a=250.0, eps=0.970, S=0.017, theta=45.0, Ch=0.554, Cp=0.292,
        Fmax=3.0, rho_bulk=210.0),
    "Mellapak 500Y (structured)": dict(
        kind="structured", a=500.0, eps=0.975, S=0.0081, theta=45.0, Ch=0.482, Cp=0.295,
        Fmax=2.4, rho_bulk=330.0),
    "Wire gauze BX 500 (structured)": dict(
        kind="structured", a=492.0, eps=0.900, S=0.0089, theta=60.0, Ch=0.650, Cp=0.390,
        Fmax=2.1, rho_bulk=210.0),
    "Pall ring 25 mm metal (random)": dict(
        kind="random", a=223.5, eps=0.954, dp=0.025, Ch=0.719, Cp=0.957, Fmax=2.0,
        sigma_c=0.075, onda_C=5.23, rho_bulk=360.0),
    "Pall ring 50 mm metal (random)": dict(
        kind="random", a=112.6, eps=0.951, dp=0.050, Ch=0.784, Cp=0.763, Fmax=2.6,
        sigma_c=0.075, onda_C=5.23, rho_bulk=390.0),
    "Hiflow ring 50 mm metal (random)": dict(
        kind="random", a=92.3, eps=0.977, dp=0.050, Ch=0.876, Cp=0.421, Fmax=2.9,
        sigma_c=0.075, onda_C=5.23, rho_bulk=180.0),
}


def get(name):
    return PACKINGS[name]


# -----------------------------------------------------------------------------
# Liquid holdup (Billet-Schultes pre-loading)
# -----------------------------------------------------------------------------
def _holdup_coeffs(pk, rhoL, muL):
    a, Ch = pk["a"], pk["Ch"]
    base = (12.0 * muL * a ** 2 / (G * rhoL)) ** (1 / 3)
    # Re_L >= 5:  ah/a = 0.85 Ch Re^0.25 Fr^0.1 ;  Re_L < 5: ah/a = Ch Re^0.15 Fr^0.1
    k_hi = base * (0.85 * Ch * (rhoL / (a * muL)) ** 0.25 * (a / G) ** 0.1) ** (2 / 3)
    k_lo = base * (Ch * (rhoL / (a * muL)) ** 0.15 * (a / G) ** 0.1) ** (2 / 3)
    return k_hi, k_lo


def holdup(pk, uL, rhoL, muL):
    """Fractional liquid holdup [m3/m3] for superficial liquid velocity uL [m/s]."""
    k_hi, k_lo = _holdup_coeffs(pk, rhoL, muL)
    Re = uL * rhoL / (pk["a"] * muL)
    return np.where(Re >= 5.0, k_hi * uL ** 0.6333, k_lo * uL ** 0.5667)


def uL_from_holdup(pk, hL, rhoL, muL):
    """Analytical inverse of `holdup` (continuous at Re_L = 5)."""
    k_hi, k_lo = _holdup_coeffs(pk, rhoL, muL)
    h = np.maximum(hL, 0.0)
    u_hi = (h / k_hi) ** (1 / 0.6333)
    u_lo = (h / k_lo) ** (1 / 0.5667)
    Re_hi = u_hi * rhoL / (pk["a"] * muL)
    return np.where(Re_hi >= 5.0, u_hi, u_lo)


# -----------------------------------------------------------------------------
# Pressure drop and capacity
# -----------------------------------------------------------------------------
def pressure_drop(pk, D, uV, rhoV, muV, uL, hL):
    """Irrigated pressure drop [Pa/m] and fraction of flood (F / Fmax)."""
    a, eps, Cp = pk["a"], pk["eps"], pk["Cp"]
    dp = 6.0 * (1.0 - eps) / a
    K = 1.0 / (1.0 + (2.0 / 3.0) / (1.0 - eps) * dp / D)
    ReV = np.maximum(np.abs(uV) * dp * rhoV * K / ((1.0 - eps) * muV), 1e-3)
    psi0 = Cp * (64.0 / ReV + 1.8 / ReV ** 0.08)
    dP0 = psi0 * a / eps ** 3 * rhoV * uV ** 2 / 2.0 / K
    FrL = uL ** 2 * a / G
    wet = (eps / np.maximum(eps - hL, 0.05)) ** 1.5 * np.exp(13300.0 / a ** 1.5 * np.sqrt(FrL))
    ff = np.abs(uV) * np.sqrt(rhoV) / pk["Fmax"]
    loading = 1.0 + 2.0 * np.minimum(ff, 1.2) ** 8          # empirical loading-region rise
    return dP0 * wet * loading, ff


# -----------------------------------------------------------------------------
# Mass transfer
# -----------------------------------------------------------------------------
def transfer_units(pk, uV, uL, hL, rhoL, muL, sigma, DL, rhoV, muV, DV, mt_factor=1.0):
    """
    Height of gas- and liquid-film transfer units H_G, H_L [m] and effective area [m2/m3].
    """
    a, eps = pk["a"], pk["eps"]
    uV = np.maximum(np.abs(uV), 1e-4)
    uL = np.maximum(uL, 1e-7)
    if pk["kind"] == "structured":
        S, sn = pk["S"], np.sin(np.radians(pk["theta"]))
        uGe = uV / (eps * np.maximum(1.0 - hL, 0.5) * sn)
        uLe = uL / (eps * np.maximum(hL, 1e-3) * sn)
        ScV = muV / (rhoV * DV)
        kG = 0.054 * DV / S * ((uGe + uLe) * rhoV * S / muV) ** 0.8 * ScV ** 0.33
        kL = 2.0 * np.sqrt(DL * 0.9 * uLe / (np.pi * S))
        ae = a * np.minimum(1.34 * ((rhoL / sigma) * G ** (1 / 3) * (uL / a) ** (4 / 3)) ** 0.116, 1.0)
    else:
        dp, Ld, Gm = pk["dp"], uL * rhoL, uV * rhoV
        ReL = Ld / (a * muL)
        FrL = Ld ** 2 * a / (rhoL ** 2 * G)
        WeL = Ld ** 2 / (rhoL * sigma * a)
        aw = a * (1.0 - np.exp(-1.45 * (pk["sigma_c"] / sigma) ** 0.75 * ReL ** 0.1
                               * FrL ** -0.05 * WeL ** 0.2))
        ae = np.maximum(aw, 0.05 * a)
        ScL = muL / (rhoL * DL)
        kL = 0.0051 * (Ld / (ae * muL)) ** (2 / 3) * ScL ** -0.5 * (a * dp) ** 0.4 \
            * (muL * G / rhoL) ** (1 / 3)
        ScV = muV / (rhoV * DV)
        kG = pk["onda_C"] * a * DV * (Gm / (a * muV)) ** 0.7 * ScV ** (1 / 3) * (a * dp) ** -2.0
    kG, kL = kG * mt_factor, kL * mt_factor
    return uV / (kG * ae), uL / (kL * ae), ae


def hetp(HOG, lam):
    """HETP from H_OG and stripping factor lambda = m V / L."""
    lam = np.asarray(lam, dtype=float)
    f = np.where(np.abs(lam - 1.0) < 1e-3, 1.0, np.log(np.maximum(lam, 1e-6)) / (lam - 1.0))
    return HOG / np.maximum(f, 1e-6)
