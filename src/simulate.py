"""
Data simulation for the NMD rate model.

Two DGPs:
  1. Static pass-through (original, for notebook 04 baseline)
  2. ECM with Markov regime-switching betas (new, for notebook 04_ecm)
"""

import numpy as np
import pandas as pd

# ── Fixed model parameters ────────────────────────────────────────────────────

LAMBDA     = 0.0609
MATURITIES = np.array([3, 6, 12, 24, 36, 60, 84, 120], dtype=float)
SEGMENTS   = ["Retail Current", "Retail Savings", "SME Operational", "Corporate"]

# ── VAR(1) DGP (used when no real data is available) ─────────────────────────

TRUE_MU  = np.array([5.0, -1.5, 1.0])
TRUE_PHI = np.diag([0.95, 0.90, 0.85])
TRUE_Q   = np.array([
    [0.10, 0.03, 0.01],
    [0.03, 0.08, 0.02],
    [0.01, 0.02, 0.06],
])
TRUE_SIGMA_OBS = 0.10

# ── Static pass-through DGP (original) ───────────────────────────────────────

TRUE_ALPHA   = np.array([0.50, 0.80, 1.00, 1.20])
TRUE_BETA_L  = np.array([0.20, 0.35, 0.50, 0.75])
TRUE_BETA_S  = np.array([0.05, 0.10, 0.15, 0.25])
TRUE_BETA_C  = np.array([0.02, 0.05, 0.08, 0.10])
TRUE_SIGMA_D = np.array([0.10, 0.12, 0.15, 0.18])

# ── ECM regime-switching DGP ──────────────────────────────────────────────────

# Regime 0: low-rate environment (2014–2021 in the ECB data)
TRUE_BETA_L_R0 = np.array([0.03, 0.05, 0.08, 0.15])
TRUE_BETA_S_R0 = np.array([0.01, 0.02, 0.03, 0.05])
TRUE_BETA_C_R0 = np.array([0.01, 0.02, 0.02, 0.03])

# Regime 1: hiking cycle (2022–2024)
TRUE_BETA_L_R1 = np.array([0.20, 0.35, 0.50, 0.75])
TRUE_BETA_S_R1 = np.array([0.05, 0.10, 0.15, 0.25])
TRUE_BETA_C_R1 = np.array([0.02, 0.05, 0.08, 0.10])

# Regime-invariant ECM parameters
TRUE_GAMMA   = np.array([-0.20, -0.25, -0.30, -0.40])   # adjustment speeds
TRUE_SIGMA_ECM = np.array([0.08, 0.10, 0.12, 0.15])      # ECM residual noise


# ── Core functions ────────────────────────────────────────────────────────────

def ns_loadings(maturities=MATURITIES, lam=LAMBDA):
    """Nelson-Siegel factor loadings (len(maturities), 3)."""
    tau = np.asarray(maturities, dtype=float)
    x   = lam * tau
    l1  = (1.0 - np.exp(-x)) / x
    l2  = l1 - np.exp(-x)
    return np.column_stack([np.ones_like(tau), l1, l2])


def simulate_factors(T, rng):
    """Simulate T periods of VAR(1) factors (used as fallback without real data)."""
    L       = np.linalg.cholesky(TRUE_Q)
    factors = np.zeros((T, 3))
    factors[0] = TRUE_MU
    for t in range(1, T):
        factors[t] = TRUE_MU + TRUE_PHI @ (factors[t - 1] - TRUE_MU) + L @ rng.standard_normal(3)
    return factors


def simulate_yields(factors, rng):
    """Generate yield observations from factors."""
    Lambda      = ns_loadings()
    yields_true = factors @ Lambda.T
    noise       = rng.normal(0.0, TRUE_SIGMA_OBS, size=yields_true.shape)
    return yields_true + noise, yields_true


def simulate_deposit_rates(factors, rng):
    """Original static pass-through DGP."""
    T = len(factors)
    S = len(SEGMENTS)
    L, Sl, C = factors[:, 0], factors[:, 1], factors[:, 2]
    rates = np.zeros((T, S))
    for s in range(S):
        mean = (TRUE_ALPHA[s]
                + TRUE_BETA_L[s] * L
                + TRUE_BETA_S[s] * Sl
                + TRUE_BETA_C[s] * C)
        rates[:, s] = mean + rng.normal(0.0, TRUE_SIGMA_D[s], size=T)
    return rates


def make_regime_sequence(T, dates=None):
    """
    Build the true regime sequence.
    Regime 0 (low rate) until 2021-12, Regime 1 (hiking) from 2022-01 onwards.
    If dates is None, assume data starts at 2014-01 and use index positions.
    """
    z = np.zeros(T, dtype=int)
    if dates is not None:
        for i, d in enumerate(dates):
            z[i] = 0 if d < pd.Timestamp("2022-01-01") else 1
    else:
        # Assume monthly from 2014-01: regime 1 starts at month 96 (2022-01)
        z[96:] = 1
    return z


def simulate_deposit_rates_ecm(factors, rng, dates=None):
    """
    ECM + regime-switching DGP.

    Regime 0 (low rate):  very low pass-through.
    Regime 1 (hiking):    normal pass-through.
    Adjustment speed gamma is regime-invariant.

    Parameters
    ----------
    factors : (T, 3) array [Level, Slope, Curvature]
    rng     : numpy default_rng
    dates   : DatetimeIndex for regime assignment (optional)

    Returns
    -------
    deposit_rates : (T, S)
    regime_seq    : (T,) int array of true regimes
    """
    T = len(factors)
    S = len(SEGMENTS)
    z = make_regime_sequence(T, dates)

    BETA_L = np.stack([TRUE_BETA_L_R0, TRUE_BETA_L_R1])  # (2, S)
    BETA_S = np.stack([TRUE_BETA_S_R0, TRUE_BETA_S_R1])
    BETA_C = np.stack([TRUE_BETA_C_R0, TRUE_BETA_C_R1])

    L, Sl, C = factors[:, 0], factors[:, 1], factors[:, 2]

    # Initialise deposit rates at long-run equilibrium under regime z[0]
    r = np.zeros((T, S))
    r[0] = (TRUE_ALPHA
            + BETA_L[z[0]] * L[0]
            + BETA_S[z[0]] * Sl[0]
            + BETA_C[z[0]] * C[0])

    for t in range(1, T):
        k = z[t]
        r_eq_prev = (TRUE_ALPHA
                     + BETA_L[k] * L[t - 1]
                     + BETA_S[k] * Sl[t - 1]
                     + BETA_C[k] * C[t - 1])
        ect = r[t - 1] - r_eq_prev
        dr  = TRUE_GAMMA * ect + rng.normal(0.0, TRUE_SIGMA_ECM, size=S)
        r[t] = r[t - 1] + dr

    return r, z


def simulate_all(T=120, seed=42):
    """Simulate full dataset using synthetic VAR(1) factors (no real data)."""
    rng    = np.random.default_rng(seed)
    factors = simulate_factors(T, rng)
    yields_obs, yields_true = simulate_yields(factors, rng)
    deposit_rates = simulate_deposit_rates(factors, rng)

    dates = pd.date_range("2015-01", periods=T, freq="MS")
    mat_labels = [f"{int(m)}m" for m in MATURITIES]

    return {
        "factors":     pd.DataFrame(factors,       index=dates, columns=["Level", "Slope", "Curvature"]),
        "yields":      pd.DataFrame(yields_obs,    index=dates, columns=mat_labels),
        "yields_true": pd.DataFrame(yields_true,   index=dates, columns=mat_labels),
        "deposits":    pd.DataFrame(deposit_rates, index=dates, columns=SEGMENTS),
    }
