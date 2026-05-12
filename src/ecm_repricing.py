"""
Hierarchical ECM with Markov regime-switching betas.

Model:
    Δr_{s,t} = γ_s · ECT_{s,t-1}^{(z_t)} + ε_{s,t},   ε_{s,t} ~ N(0, σ_s²)

    ECT_{s,t-1}^{(k)} = r_{s,t-1} − α_s − β_L_s^{(k)} L_{t-1}
                                          − β_S_s^{(k)} S_{t-1}
                                          − β_C_s       C_{t-1}

    z_t | z_{t-1} ~ Markov(P)  with P = [[p00, 1-p00],[1-p11, p11]]

Regime identification: μ_βL^{(0)} < μ_βL^{(1)}  (δ > 0 constraint).

The Hamilton filter marginalises over z_t exactly, giving a differentiable
marginal log-likelihood used as a pm.Potential for NUTS/numpyro.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor import scan

from simulate import SEGMENTS

N_SEGMENTS = len(SEGMENTS)
N_REGIMES  = 2


# ── Hamilton filter (pytensor) ────────────────────────────────────────────────

def _hamilton_step(
    delta_r_t, L_tm1, S_tm1, C_tm1, r_tm1,
    xi_prev,
    P, alpha, beta_L, beta_S, beta_C, gamma, sigma_dep,
):
    """
    One Hamilton filter step.

    Parameters
    ----------
    delta_r_t        : (S,)   deposit rate changes Δr_{s,t}
    L_tm1,S_tm1,C_tm1: scalar lagged NS factors
    r_tm1            : (S,)   lagged deposit rates
    xi_prev          : (2,)   filtered regime probabilities ξ_{t-1|t-1}
    P                : (2,2)  transition matrix, P[i,j] = P(z_t=j | z_{t-1}=i)
    alpha            : (S,)   intercepts
    beta_L           : (2,S)  level betas [regime, segment]
    beta_S           : (2,S)  slope betas
    beta_C           : (S,)   curvature betas (regime-invariant, broadcast below)
    gamma            : (S,)   adjustment speeds (strictly negative)
    sigma_dep        : (S,)   observation noise

    Returns
    -------
    xi_upd : (2,)   updated filtered probabilities
    ll_t   : scalar log-likelihood contribution
    """
    # ── Predict ──────────────────────────────────────────────────────────────
    xi_pred = P.T @ xi_prev  # (2,)

    # ── Long-run equilibrium under each regime: (2, S) ───────────────────────
    r_eq = (alpha[None, :]
            + beta_L * L_tm1
            + beta_S * S_tm1
            + beta_C[None, :] * C_tm1)

    # ── ECT: (2, S) ──────────────────────────────────────────────────────────
    ect = r_tm1[None, :] - r_eq

    # ── Conditional mean of Δr under each regime: (2, S) ─────────────────────
    mu = gamma[None, :] * ect

    # ── Log conditional density (sum log-normal across segments): (2,) ────────
    log_f = pt.sum(
        -0.5 * pt.log(2.0 * np.pi)
        - pt.log(sigma_dep[None, :])
        - 0.5 * ((delta_r_t[None, :] - mu) / sigma_dep[None, :]) ** 2,
        axis=1,
    )

    # ── Hamilton update (log domain for numerical stability) ─────────────────
    log_joint = pt.log(xi_pred + 1e-300) + log_f   # (2,)
    log_c     = pt.max(log_joint)
    log_norm  = log_c + pt.log(pt.sum(pt.exp(log_joint - log_c)) + 1e-300)

    xi_upd = pt.exp(log_joint - log_norm)
    ll_t   = log_norm

    return xi_upd, ll_t


def hamilton_filter(
    delta_r, L_lag, S_lag, C_lag, r_lag,
    P, alpha, beta_L, beta_S, beta_C, gamma, sigma_dep,
):
    """
    Run Hamilton filter over T-1 observations.

    Parameters
    ----------
    delta_r      : (T-1, S) tensor
    L_lag,S_lag,C_lag : (T-1,) tensors
    r_lag        : (T-1, S) tensor
    (remaining)  : model parameters (pytensor variables)

    Returns
    -------
    xi_filtered : (T-1, 2)
    log_lik     : scalar
    """
    # Stationary distribution as initial state
    p00   = P[0, 0]
    p11   = P[1, 1]
    denom = 2.0 - p00 - p11
    xi0   = pt.stack([(1.0 - p11) / denom, (1.0 - p00) / denom])  # (2,)

    [xi_filtered, log_liks], _ = scan(
        fn=_hamilton_step,
        sequences=[delta_r, L_lag, S_lag, C_lag, r_lag],
        outputs_info=[xi0, None],
        non_sequences=[P, alpha, beta_L, beta_S, beta_C, gamma, sigma_dep],
    )

    return xi_filtered, pt.sum(log_liks)


# ── PyMC model ────────────────────────────────────────────────────────────────

def build_ecm_model(factors, deposit_rates):
    """
    Build the ECM + Markov regime-switching PyMC model.

    Parameters
    ----------
    factors       : (T, 3) numpy array — filtered [Level, Slope, Curvature]
    deposit_rates : (T, S) numpy array — observed deposit rates
    """
    T, S = deposit_rates.shape
    assert S == N_SEGMENTS

    # Precompute lagged sequences (T-1 observations)
    delta_r = (deposit_rates[1:] - deposit_rates[:-1]).astype(np.float64)  # (T-1, S)
    r_lag   = deposit_rates[:-1].astype(np.float64)                        # (T-1, S)
    L_lag   = factors[:-1, 0].astype(np.float64)                           # (T-1,)
    S_lag   = factors[:-1, 1].astype(np.float64)
    C_lag   = factors[:-1, 2].astype(np.float64)

    delta_r_t = pt.as_tensor_variable(delta_r)
    r_lag_t   = pt.as_tensor_variable(r_lag)
    L_lag_t   = pt.as_tensor_variable(L_lag)
    S_lag_t   = pt.as_tensor_variable(S_lag)
    C_lag_t   = pt.as_tensor_variable(C_lag)

    coords = {
        "segment": SEGMENTS,
        "regime":  ["Low rate", "Normal rate"],
    }

    with pm.Model(coords=coords) as model:

        # ── Transition matrix ─────────────────────────────────────────────────
        p00 = pm.Beta("p00", alpha=18.0, beta=2.0)   # P(stay in regime 0)
        p11 = pm.Beta("p11", alpha=18.0, beta=2.0)   # P(stay in regime 1)
        P   = pt.stack([[p00, 1.0 - p00], [1.0 - p11, p11]])  # (2, 2)

        # ── Intercepts (regime-invariant) ─────────────────────────────────────
        mu_alpha    = pm.Normal("mu_alpha", mu=0.8, sigma=1.0)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=0.5)
        alpha_z     = pm.Normal("alpha_z", mu=0.0, sigma=1.0, dims="segment")
        alpha       = pm.Deterministic(
            "alpha", mu_alpha + sigma_alpha * alpha_z, dims="segment"
        )  # (S,)

        # ── Level betas — REGIME-SPECIFIC ────────────────────────────────────
        # Identification: regime 0 has strictly lower pass-through (delta > 0)
        mu_beta_L_0  = pm.Normal("mu_beta_L_0",  mu=0.05, sigma=0.10)
        delta_beta_L = pm.HalfNormal("delta_beta_L", sigma=0.30)
        mu_beta_L_1  = pm.Deterministic("mu_beta_L_1", mu_beta_L_0 + delta_beta_L)
        sigma_beta_L = pm.HalfNormal("sigma_beta_L", sigma=0.15)
        beta_L_z     = pm.Normal("beta_L_z", mu=0.0, sigma=1.0, shape=(2, S))
        mu_beta_L    = pt.stack([mu_beta_L_0, mu_beta_L_1])          # (2,)
        beta_L       = pm.Deterministic(
            "beta_L", mu_beta_L[:, None] + sigma_beta_L * beta_L_z   # (2, S)
        )

        # ── Slope betas — REGIME-SPECIFIC ─────────────────────────────────────
        mu_beta_S_0  = pm.Normal("mu_beta_S_0",  mu=0.02, sigma=0.08)
        delta_beta_S = pm.HalfNormal("delta_beta_S", sigma=0.15)
        mu_beta_S_1  = pm.Deterministic("mu_beta_S_1", mu_beta_S_0 + delta_beta_S)
        sigma_beta_S = pm.HalfNormal("sigma_beta_S", sigma=0.10)
        beta_S_z     = pm.Normal("beta_S_z", mu=0.0, sigma=1.0, shape=(2, S))
        mu_beta_S    = pt.stack([mu_beta_S_0, mu_beta_S_1])
        beta_S       = pm.Deterministic(
            "beta_S", mu_beta_S[:, None] + sigma_beta_S * beta_S_z   # (2, S)
        )

        # ── Curvature betas — regime-invariant ────────────────────────────────
        mu_beta_C    = pm.Normal("mu_beta_C",    mu=0.03, sigma=0.10)
        sigma_beta_C = pm.HalfNormal("sigma_beta_C", sigma=0.08)
        beta_C_z     = pm.Normal("beta_C_z", mu=0.0, sigma=1.0, dims="segment")
        beta_C       = pm.Deterministic(
            "beta_C", mu_beta_C + sigma_beta_C * beta_C_z, dims="segment"
        )  # (S,)

        # ── ECM adjustment speed (hierarchical, strictly negative) ─────────────
        mu_log_adj    = pm.Normal("mu_log_adj",    mu=np.log(0.25), sigma=0.5)
        sigma_log_adj = pm.HalfNormal("sigma_log_adj", sigma=0.30)
        adj_z         = pm.Normal("adj_z", mu=0.0, sigma=1.0, dims="segment")
        gamma         = pm.Deterministic(
            "gamma",
            -pt.exp(mu_log_adj + sigma_log_adj * adj_z),
            dims="segment",
        )  # (S,), strictly negative

        # ── Observation noise ─────────────────────────────────────────────────
        sigma_dep = pm.HalfNormal("sigma_dep", sigma=0.15, dims="segment")

        # ── Hamilton filter marginal likelihood ───────────────────────────────
        _, log_lik = hamilton_filter(
            delta_r_t, L_lag_t, S_lag_t, C_lag_t, r_lag_t,
            P, alpha, beta_L, beta_S, beta_C, gamma, sigma_dep,
        )
        pm.Potential("hamilton_loglik", log_lik)

    return model


# ── Numpy forward pass ────────────────────────────────────────────────────────

def extract_regime_probs_numpy(idata, factors, deposit_rates):
    """
    Run the Hamilton filter in numpy with posterior mean parameters.

    Returns
    -------
    xi_filtered : (T-1, 2) array — P(z_t = k | y_{1:t}) for each t
    """
    post = idata.posterior

    def pm(name):
        return post[name].mean(("chain", "draw")).values

    p00_v = float(pm("p00"))
    p11_v = float(pm("p11"))
    P_np  = np.array([[p00_v, 1 - p00_v], [1 - p11_v, p11_v]])

    alpha_np  = pm("alpha")         # (S,)
    beta_L_np = pm("beta_L")        # (2, S)
    beta_S_np = pm("beta_S")        # (2, S)
    beta_C_np = pm("beta_C")        # (S,)
    gamma_np  = pm("gamma")         # (S,)
    sigma_np  = pm("sigma_dep")     # (S,)

    T, S = deposit_rates.shape
    delta_r = deposit_rates[1:] - deposit_rates[:-1]
    r_lag   = deposit_rates[:-1]
    L_lag   = factors[:-1, 0]
    S_lag   = factors[:-1, 1]
    C_lag   = factors[:-1, 2]

    denom = 2.0 - p00_v - p11_v
    xi    = np.array([(1.0 - p11_v) / denom, (1.0 - p00_v) / denom])
    xi_all = np.zeros((T - 1, N_REGIMES))

    for t in range(T - 1):
        xi_pred = P_np.T @ xi
        r_eq    = (alpha_np[None, :]
                   + beta_L_np * L_lag[t]
                   + beta_S_np * S_lag[t]
                   + beta_C_np[None, :] * C_lag[t])           # (2, S)
        ect     = r_lag[t][None, :] - r_eq                    # (2, S)
        mu      = gamma_np[None, :] * ect                     # (2, S)
        log_f   = np.sum(
            -0.5 * np.log(2 * np.pi)
            - np.log(sigma_np[None, :])
            - 0.5 * ((delta_r[t][None, :] - mu) / sigma_np[None, :]) ** 2,
            axis=1,
        )                                                      # (2,)
        log_joint = np.log(xi_pred + 1e-300) + log_f
        log_norm  = np.log(np.sum(np.exp(log_joint - log_joint.max()))) + log_joint.max()
        xi        = np.exp(log_joint - log_norm)
        xi_all[t] = xi

    return xi_all


def predict_deposit_rates_ecm(idata, factors_hist, deposits_hist, factors_new, regime_new):
    """
    Posterior predictive ECM simulation for scenario factor paths.

    Parameters
    ----------
    idata         : ArviZ InferenceData with posterior
    factors_hist  : (T, 3) historical factors (for initialisation)
    deposits_hist : (T, S) historical deposit rates
    factors_new   : (H, 3) scenario factor path
    regime_new    : (H,) int — assumed regime at each future period (0 or 1)

    Returns
    -------
    pred : (n_samples, H, S)
    """
    post = idata.posterior
    alpha_s  = post["alpha"].values.reshape(-1, N_SEGMENTS)    # (N, S)
    beta_L_s = post["beta_L"].values.reshape(-1, 2, N_SEGMENTS)# (N, 2, S)
    beta_S_s = post["beta_S"].values.reshape(-1, 2, N_SEGMENTS)
    beta_C_s = post["beta_C"].values.reshape(-1, N_SEGMENTS)
    gamma_s  = post["gamma"].values.reshape(-1, N_SEGMENTS)
    sigma_s  = post["sigma_dep"].values.reshape(-1, N_SEGMENTS)

    N = alpha_s.shape[0]
    H = len(factors_new)
    S = N_SEGMENTS

    rng   = np.random.default_rng(0)
    r0    = deposits_hist[-1]           # (S,) last observed rate
    pred  = np.zeros((N, H, S))

    for n in range(N):
        r = r0.copy()
        for h in range(H):
            k   = regime_new[h]
            L_t = factors_new[h, 0]
            S_t = factors_new[h, 1]
            C_t = factors_new[h, 2]
            r_eq = (alpha_s[n]
                    + beta_L_s[n, k] * L_t
                    + beta_S_s[n, k] * S_t
                    + beta_C_s[n] * C_t)
            ect  = r - r_eq
            dr   = gamma_s[n] * ect + rng.normal(0.0, sigma_s[n])
            r    = r + dr
            pred[n, h] = r

    return pred
