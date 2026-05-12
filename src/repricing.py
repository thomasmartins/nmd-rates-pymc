"""
Bayesian hierarchical deposit repricing model.

For each segment s and time t:
    r_{s,t} = αₛ + β_L_s · Lₜ + β_S_s · Sₜ + β_C_s · Cₜ + εₜ,   εₜ ~ N(0, σₛ²)

Segment-level parameters are drawn from shared hyperpriors (partial pooling):
    αₛ     ~ Normal(μ_α,   σ_α)
    β_L_s  ~ Normal(μ_βL,  σ_βL)
    β_S_s  ~ Normal(μ_βS,  σ_βS)
    β_C_s  ~ Normal(μ_βC,  σ_βC)

Non-centred parameterisation is used throughout to improve MCMC geometry.
"""

import numpy as np
import pymc as pm

from simulate import SEGMENTS

N_SEGMENTS = len(SEGMENTS)


def build_repricing_model(factors, deposit_rates):
    """Build and return the hierarchical repricing PyMC model.

    Parameters
    ----------
    factors       : (T, 3) array — filtered [Level, Slope, Curvature] (%)
    deposit_rates : (T, S) array — observed deposit rates per segment (%)
    """
    T, S = deposit_rates.shape
    assert S == N_SEGMENTS, f"Expected {N_SEGMENTS} segments, got {S}"

    L  = factors[:, 0]
    Sl = factors[:, 1]
    C  = factors[:, 2]

    coords = {"segment": SEGMENTS, "obs": np.arange(T)}

    with pm.Model(coords=coords) as model:

        # ── Hyperpriors ───────────────────────────────────────────────────────
        mu_alpha    = pm.Normal("mu_alpha",   mu=0.8,  sigma=1.0)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=0.5)

        mu_beta_L    = pm.Normal("mu_beta_L",   mu=0.40, sigma=0.30)
        sigma_beta_L = pm.HalfNormal("sigma_beta_L", sigma=0.20)

        mu_beta_S    = pm.Normal("mu_beta_S",   mu=0.10, sigma=0.20)
        sigma_beta_S = pm.HalfNormal("sigma_beta_S", sigma=0.15)

        mu_beta_C    = pm.Normal("mu_beta_C",   mu=0.05, sigma=0.15)
        sigma_beta_C = pm.HalfNormal("sigma_beta_C", sigma=0.10)

        # ── Segment-level parameters (non-centred) ────────────────────────────
        alpha_z = pm.Normal("alpha_z", mu=0.0, sigma=1.0, dims="segment")
        alpha   = pm.Deterministic("alpha", mu_alpha + sigma_alpha * alpha_z, dims="segment")

        beta_L_z = pm.Normal("beta_L_z", mu=0.0, sigma=1.0, dims="segment")
        beta_L   = pm.Deterministic("beta_L", mu_beta_L + sigma_beta_L * beta_L_z, dims="segment")

        beta_S_z = pm.Normal("beta_S_z", mu=0.0, sigma=1.0, dims="segment")
        beta_S   = pm.Deterministic("beta_S", mu_beta_S + sigma_beta_S * beta_S_z, dims="segment")

        beta_C_z = pm.Normal("beta_C_z", mu=0.0, sigma=1.0, dims="segment")
        beta_C   = pm.Deterministic("beta_C", mu_beta_C + sigma_beta_C * beta_C_z, dims="segment")

        # ── Observation noise (per segment) ───────────────────────────────────
        sigma_dep = pm.HalfNormal("sigma_dep", sigma=0.2, dims="segment")

        # ── Linear predictor  (T, S) ──────────────────────────────────────────
        mu_dep = (
            alpha[None, :]
            + beta_L[None, :] * L[:, None]
            + beta_S[None, :] * Sl[:, None]
            + beta_C[None, :] * C[:, None]
        )

        # ── Likelihood ────────────────────────────────────────────────────────
        pm.Normal(
            "deposit_rates",
            mu=mu_dep,
            sigma=sigma_dep[None, :],
            observed=deposit_rates,
            dims=("obs", "segment"),
        )

    return model


def predict_deposit_rates(idata, factors_new):
    """Compute posterior predictive deposit rates for new factor paths.

    Parameters
    ----------
    idata       : ArviZ InferenceData with posterior samples
    factors_new : (T_new, 3) array — [Level, Slope, Curvature] scenarios

    Returns
    -------
    pred : (n_samples, T_new, S) array of posterior predictive rates
    """
    post = idata.posterior
    alpha  = post["alpha"].values    # (chains, draws, S)
    beta_L = post["beta_L"].values
    beta_S = post["beta_S"].values
    beta_C = post["beta_C"].values
    sigma  = post["sigma_dep"].values

    # Flatten chains and draws
    n_chains, n_draws, S = alpha.shape
    n_samples = n_chains * n_draws
    alpha  = alpha.reshape(n_samples, S)
    beta_L = beta_L.reshape(n_samples, S)
    beta_S = beta_S.reshape(n_samples, S)
    beta_C = beta_C.reshape(n_samples, S)
    sigma  = sigma.reshape(n_samples, S)

    T_new = len(factors_new)
    L  = factors_new[:, 0]   # (T_new,)
    Sl = factors_new[:, 1]
    C  = factors_new[:, 2]

    # Mean prediction: (n_samples, T_new, S)
    mu = (
        alpha[:, None, :]
        + beta_L[:, None, :] * L[None, :, None]
        + beta_S[:, None, :] * Sl[None, :, None]
        + beta_C[:, None, :] * C[None, :, None]
    )
    # Add observation noise
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, sigma[:, None, :], size=(n_samples, T_new, S))
    return mu + noise
