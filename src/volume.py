"""
NMD volume model.

Log-volume follows a partial-adjustment AR(1) with spread sensitivity:

    log V_{s,t} = α_s + ρ_s · log V_{s,t-1}
                       + β_s(p_{t-1}) · spread_{s,t-1}
                       + ε_{s,t}

where
    spread_{s,t}   = yield_5y_t − r_{s,t}   (opportunity cost)
    β_s(p)         = β⁰_s + Δβ_s · p        (regime-varying sensitivity)
    p_{t-1}        = P(z_{t-1}=1 | y_{1:t-1})  (filtered regime prob from ECM)

Regime-varying sensitivity: in the hiking cycle (p→1), spread sensitivity
increases — depositors are more likely to shift funds to higher-yield products.
Constraint: β⁰_s < 0 and Δβ_s < 0  (both regimes have negative sensitivity).

Hierarchical priors across segments for α, ρ, β⁰, Δβ, σ.

NII contribution: V_{s,t} · spread_{s,t} / 12  (monthly, annualised rates).
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from simulate import SEGMENTS

N_SEGMENTS = len(SEGMENTS)

# ── True DGP parameters ───────────────────────────────────────────────────────

# Initial volumes (arbitrary units, e.g. EUR bn)
TRUE_V0 = np.array([100.0, 50.0, 30.0, 20.0])

# Volume persistence  ρ_s ∈ (0,1): retail more persistent
TRUE_RHO = np.array([0.98, 0.95, 0.92, 0.88])

# Spread sensitivity in regime 0 (low rate): mild negative
TRUE_BETA_V_R0 = np.array([-0.02, -0.03, -0.05, -0.10])

# Spread sensitivity in regime 1 (hiking): stronger negative
TRUE_BETA_V_R1 = np.array([-0.05, -0.08, -0.12, -0.20])

# Implied Δβ = β_R1 − β_R0 (negative: more sensitive in hiking cycle)
TRUE_DELTA_BETA_V = TRUE_BETA_V_R1 - TRUE_BETA_V_R0

# Volume noise (log scale)
TRUE_SIGMA_V = np.array([0.010, 0.015, 0.020, 0.025])


# ── Simulation ────────────────────────────────────────────────────────────────

def compute_spread(deposit_rates, yields, maturity_col="60m"):
    """
    Opportunity cost spread: 5y yield − deposit rate.

    Parameters
    ----------
    deposit_rates : (T, S) array or DataFrame
    yields        : DataFrame with maturity columns (e.g. '60m')

    Returns
    -------
    spread : (T, S) array
    """
    y5 = yields[maturity_col].values[:, None]          # (T, 1)
    r  = deposit_rates.values if hasattr(deposit_rates, "values") else deposit_rates
    return y5 - r                                       # (T, S)


def simulate_volumes(deposit_rates, yields, regime_seq, rng, maturity_col="60m"):
    """
    Simulate NMD volumes using the true DGP.

    Parameters
    ----------
    deposit_rates : (T, S) array
    yields        : DataFrame with column `maturity_col`
    regime_seq    : (T,) int array of true regimes {0, 1}
    rng           : numpy default_rng

    Returns
    -------
    volumes : (T, S) array (level, not log)
    """
    T = len(deposit_rates)
    S = N_SEGMENTS
    spread = compute_spread(deposit_rates, yields, maturity_col)  # (T, S)

    log_v = np.zeros((T, S))
    log_v[0] = np.log(TRUE_V0)

    for t in range(1, T):
        k       = regime_seq[t - 1]                           # lagged regime
        beta_vt = np.where(k == 0, TRUE_BETA_V_R0, TRUE_BETA_V_R1)  # (S,)
        log_v[t] = (TRUE_RHO * log_v[t - 1]
                    + beta_vt * spread[t - 1]
                    + rng.normal(0.0, TRUE_SIGMA_V))
        # α_s absorbed into initial level (long-run mean = log V0 / (1-ρ))

    return np.exp(log_v)


# ── PyMC model ────────────────────────────────────────────────────────────────

def build_volume_model(volumes, deposit_rates, yields, regime_probs, maturity_col="60m"):
    """
    Build the hierarchical volume PyMC model.

    Parameters
    ----------
    volumes       : (T, S) numpy array — observed volumes (level)
    deposit_rates : (T, S) numpy array — deposit rates (%)
    yields        : DataFrame         — yield curve (contains maturity_col)
    regime_probs  : (T-1, 2) numpy array — filtered regime probs from ECM
                    (regime_probs[:, 1] = P(z_t=1 | y_{1:t}))
    maturity_col  : str — yield maturity used for spread computation
    """
    T, S = volumes.shape
    assert S == N_SEGMENTS

    spread = compute_spread(deposit_rates, yields, maturity_col)  # (T, S)

    # Align: model uses observations t=1,...,T-1 given t-1 info
    log_v_obs  = np.log(volumes[1:]).astype(np.float64)     # (T-1, S)
    log_v_lag  = np.log(volumes[:-1]).astype(np.float64)    # (T-1, S)
    spread_lag = spread[:-1].astype(np.float64)             # (T-1, S)
    # Lagged regime probability: use t-1 filtered probs; regime_probs has T-1 rows (t=1..T-1)
    p_lag = regime_probs[:T - 1, 1].astype(np.float64)     # (T-1,) P(z=1|past)

    # Regime-weighted spread: p_lag * spread_lag
    p_lag_spread = p_lag[:, None] * spread_lag              # (T-1, S)

    coords = {"segment": SEGMENTS}

    with pm.Model(coords=coords) as model:

        # ── Intercepts ────────────────────────────────────────────────────────
        mu_alpha    = pm.Normal("mu_alpha",    mu=0.0, sigma=1.0)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=0.5)
        alpha_z     = pm.Normal("alpha_z", mu=0.0, sigma=1.0, dims="segment")
        alpha       = pm.Deterministic(
            "alpha", mu_alpha + sigma_alpha * alpha_z, dims="segment"
        )

        # ── Persistence ρ_s ∈ (0,1) ──────────────────────────────────────────
        # Logit-normal hierarchical: logit(ρ) ~ Normal(μ_logit_rho, σ_logit_rho)
        # LogNormal prior on σ keeps density at 0 zero — avoids hyper-funnel
        # divergences that HalfNormal creates with only S=4 segments.
        mu_logit_rho    = pm.Normal("mu_logit_rho",    mu=3.5, sigma=1.0)  # ~sigmoid(3.5)≈0.97
        sigma_logit_rho = pm.LogNormal("sigma_logit_rho", mu=np.log(0.5), sigma=0.4)
        rho_z           = pm.Normal("rho_z", mu=0.0, sigma=1.0, dims="segment")
        rho             = pm.Deterministic(
            "rho",
            pm.math.sigmoid(mu_logit_rho + sigma_logit_rho * rho_z),
            dims="segment",
        )

        # ── Spread sensitivity regime 0: β⁰_s < 0 ────────────────────────────
        mu_log_beta0    = pm.Normal("mu_log_beta0",    mu=np.log(0.04), sigma=0.8)
        sigma_log_beta0 = pm.LogNormal("sigma_log_beta0", mu=np.log(0.5), sigma=0.4)
        beta0_z         = pm.Normal("beta0_z", mu=0.0, sigma=1.0, dims="segment")
        beta0           = pm.Deterministic(
            "beta0",
            -pt.exp(mu_log_beta0 + sigma_log_beta0 * beta0_z),
            dims="segment",
        )  # strictly negative

        # ── Incremental sensitivity in regime 1: Δβ_s < 0 ────────────────────
        mu_log_dbeta    = pm.Normal("mu_log_dbeta",    mu=np.log(0.06), sigma=0.8)
        sigma_log_dbeta = pm.LogNormal("sigma_log_dbeta", mu=np.log(0.5), sigma=0.4)
        dbeta_z         = pm.Normal("dbeta_z", mu=0.0, sigma=1.0, dims="segment")
        delta_beta      = pm.Deterministic(
            "delta_beta",
            -pt.exp(mu_log_dbeta + sigma_log_dbeta * dbeta_z),
            dims="segment",
        )  # strictly negative → regime 1 more sensitive

        # ── Observation noise ─────────────────────────────────────────────────
        sigma_v = pm.HalfNormal("sigma_v", sigma=0.05, dims="segment")

        # ── Linear predictor ──────────────────────────────────────────────────
        # β_s(p) = β⁰_s + Δβ_s · p
        # mu_log_v[t,s] = α_s + ρ_s · log V_{s,t-1}
        #                     + β⁰_s · spread_{s,t-1}
        #                     + Δβ_s · p_{t-1} · spread_{s,t-1}
        mu_log_v = (
            alpha[None, :]
            + rho[None, :] * log_v_lag
            + beta0[None, :] * spread_lag
            + delta_beta[None, :] * p_lag_spread
        )  # (T-1, S)

        # ── Likelihood ────────────────────────────────────────────────────────
        pm.Normal(
            "log_volumes",
            mu=mu_log_v,
            sigma=sigma_v[None, :],
            observed=log_v_obs,
        )

    return model


# ── Numpy forward pass for prediction ────────────────────────────────────────

def predict_volumes(idata, volumes_hist, deposit_rates_new, yields_new, p_new,
                    maturity_col="60m"):
    """
    Posterior predictive volume simulation for scenario paths.

    Parameters
    ----------
    idata              : ArviZ InferenceData
    volumes_hist       : (T, S) — historical volumes (last row used as V_T)
    deposit_rates_new  : (H, S) — scenario deposit rates
    yields_new         : (H, J) DataFrame or array with maturity_col column
    p_new              : (H,) — assumed P(z=1) for each future period
    maturity_col       : str

    Returns
    -------
    pred_volumes : (n_samples, H, S) — level volumes
    """
    post = idata.posterior

    def pm(name):
        return post[name].values.reshape(-1, N_SEGMENTS)

    alpha_s  = pm("alpha")
    rho_s    = pm("rho")
    beta0_s  = pm("beta0")
    dbeta_s  = pm("delta_beta")
    sigma_s  = pm("sigma_v")

    N = alpha_s.shape[0]
    H = len(deposit_rates_new)

    # Compute scenario spread
    if hasattr(yields_new, maturity_col):
        y5_new = yields_new[maturity_col].values
    elif hasattr(yields_new, "__getitem__"):
        y5_new = yields_new[maturity_col].values if hasattr(yields_new, "columns") else yields_new[:, 4]
    else:
        y5_new = yields_new[:, 4]  # fallback: 5th maturity = 60m

    dr_new = deposit_rates_new.values if hasattr(deposit_rates_new, "values") else deposit_rates_new
    spread_new = y5_new[:, None] - dr_new   # (H, S)

    rng  = np.random.default_rng(0)
    v0   = volumes_hist[-1]                  # (S,) last observed volumes
    pred = np.zeros((N, H, N_SEGMENTS))

    for n in range(N):
        log_v = np.log(v0).copy()
        for h in range(H):
            beta_t = beta0_s[n] + dbeta_s[n] * p_new[h]   # (S,)
            mu     = (alpha_s[n]
                      + rho_s[n] * log_v
                      + beta_t * spread_new[h])
            log_v  = mu + rng.normal(0.0, sigma_s[n])
            pred[n, h] = np.exp(log_v)

    return pred


# ── NII ───────────────────────────────────────────────────────────────────────

def compute_nii(volumes, deposit_rates, yields, maturity_col="60m"):
    """
    Monthly NII contribution per segment.

    NII_{s,t} = V_{s,t} · spread_{s,t} / 12

    Parameters
    ----------
    volumes       : (T, S) array or DataFrame
    deposit_rates : (T, S) array or DataFrame
    yields        : DataFrame with maturity_col

    Returns
    -------
    nii : (T, S) array (same units as volumes × %)
    """
    V = volumes.values if hasattr(volumes, "values") else volumes
    spread = compute_spread(deposit_rates, yields, maturity_col)
    return V * spread / 12.0
