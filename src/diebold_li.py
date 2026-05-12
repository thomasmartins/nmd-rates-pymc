"""
Diebold-Li state-space model.

State equation (VAR(1)):
    fₜ = μ + Φ(fₜ₋₁ − μ) + ηₜ,    ηₜ ~ N(0, Q)

Observation equation:
    yₜ = Λ·fₜ + εₜ,                εₜ ~ N(0, H)

where Λ is the (J×3) Nelson-Siegel loading matrix evaluated at fixed λ,
and H = σ²_obs · Iⱼ (homoskedastic measurement error).

The Kalman filter marginal likelihood is computed via pytensor.scan so that
PyMC/NUTS can differentiate through it.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor import scan

from simulate import LAMBDA, MATURITIES, ns_loadings

# Pre-compute fixed loading matrix (float64)
LAMBDA_MAT = ns_loadings(MATURITIES, LAMBDA).astype(np.float64)   # (J, 3)
J = LAMBDA_MAT.shape[0]
N_STATES = 3


# ── Kalman filter (pytensor) ──────────────────────────────────────────────────

def _kalman_step(y_t, f_prev, P_prev, mu, Phi, Q, sigma_obs_sq):
    """One Kalman filter prediction-update step (all pytensor ops).

    Parameters
    ----------
    y_t          : (J,)  yield observations at time t
    f_prev       : (3,)  filtered state mean at t−1
    P_prev       : (3,3) filtered state covariance at t−1
    mu           : (3,)  long-run state mean
    Phi          : (3,3) VAR(1) transition matrix
    Q            : (3,3) state noise covariance
    sigma_obs_sq : scalar measurement noise variance

    Returns
    -------
    f_upd, P_upd, log_lik_t
    """
    Lambda = pt.as_tensor_variable(LAMBDA_MAT)   # (J, 3)
    H = pt.eye(J) * sigma_obs_sq                 # (J, J)

    # ── Predict ──────────────────────────────────────────────────────────────
    f_pred = mu + Phi @ (f_prev - mu)            # (3,)
    P_pred = Phi @ P_prev @ Phi.T + Q            # (3,3)

    # ── Innovation ───────────────────────────────────────────────────────────
    v = y_t - Lambda @ f_pred                    # (J,)
    S = Lambda @ P_pred @ Lambda.T + H           # (J,J)  innovation covariance

    # ── Log-likelihood contribution ──────────────────────────────────────────
    S_chol = pt.linalg.cholesky(S)
    log_det_S = 2.0 * pt.sum(pt.log(pt.diag(S_chol)))
    S_inv_v = pt.linalg.solve(S, v)
    ll_t = -0.5 * (J * pt.log(2.0 * np.pi) + log_det_S + pt.dot(v, S_inv_v))

    # ── Update ───────────────────────────────────────────────────────────────
    # Kalman gain: K = P_pred Λᵀ S⁻¹   shape (3,J)
    K = pt.linalg.solve(S, Lambda @ P_pred).T
    f_upd = f_pred + K @ v                       # (3,)
    I_KL = pt.eye(N_STATES) - K @ Lambda
    P_upd = I_KL @ P_pred                        # (3,3)
    P_upd = 0.5 * (P_upd + P_upd.T)             # symmetrize

    return f_upd, P_upd, ll_t


def kalman_filter(y_obs_tensor, mu, Phi, Q, sigma_obs_sq):
    """Run Kalman filter over all T observations via pytensor.scan.

    Returns
    -------
    f_filtered : (T, 3)  filtered state means
    log_lik    : scalar  total log-likelihood
    """
    f0 = mu                          # initialise at long-run mean
    P0 = pt.eye(N_STATES) * 2.0     # diffuse initial covariance

    [f_filtered, _, log_liks], _ = scan(
        fn=_kalman_step,
        sequences=[y_obs_tensor],
        outputs_info=[f0, P0, None],
        non_sequences=[mu, Phi, Q, sigma_obs_sq],
    )
    return f_filtered, pt.sum(log_liks)


# ── PyMC model ────────────────────────────────────────────────────────────────

def build_dl_model(y_obs):
    """Build and return the Diebold-Li PyMC model.

    Parameters
    ----------
    y_obs : (T, J) numpy array of yield observations (%)
    """
    y_data = pt.as_tensor_variable(y_obs.astype(np.float64))

    with pm.Model() as model:

        # ── Long-run state mean ───────────────────────────────────────────────
        mu = pm.Normal("mu", mu=np.array([5.0, -1.5, 1.0]), sigma=2.0, shape=3)

        # ── VAR(1) transition matrix (diagonal) ──────────────────────────────
        # Beta prior concentrated near 1 (persistent but stationary factors)
        phi_diag = pm.Beta("phi_diag", alpha=18.0, beta=2.0, shape=3)
        Phi = pt.diag(phi_diag)

        # ── State noise covariance (diagonal) ────────────────────────────────
        q_stds = pm.HalfNormal("q_stds", sigma=0.3, shape=3)
        Q = pt.diag(q_stds ** 2)

        # ── Measurement noise ─────────────────────────────────────────────────
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.2)
        sigma_obs_sq = sigma_obs ** 2

        # ── Kalman filter marginal likelihood ─────────────────────────────────
        _, log_lik = kalman_filter(y_data, mu, Phi, Q, sigma_obs_sq)
        pm.Potential("kalman_loglik", log_lik)

    return model


# ── Numpy forward pass (post-sampling factor extraction) ──────────────────────

def extract_factors_numpy(idata, y_obs):
    """Run the Kalman filter in numpy using posterior mean parameters.

    Returns
    -------
    factors : (T, 3) array of filtered [Level, Slope, Curvature]
    """
    post = idata.posterior

    mu       = post["mu"].mean(("chain", "draw")).values          # (3,)
    phi_diag = post["phi_diag"].mean(("chain", "draw")).values    # (3,)
    q_stds   = post["q_stds"].mean(("chain", "draw")).values      # (3,)
    sigma_obs = float(post["sigma_obs"].mean(("chain", "draw")).values)

    Phi = np.diag(phi_diag)
    Q   = np.diag(q_stds ** 2)
    H   = np.eye(J) * sigma_obs ** 2
    Lambda = LAMBDA_MAT

    T = len(y_obs)
    f = mu.copy()
    P = np.eye(N_STATES) * 2.0
    factors = np.zeros((T, N_STATES))

    for t in range(T):
        # Predict
        f_pred = mu + Phi @ (f - mu)
        P_pred = Phi @ P @ Phi.T + Q
        # Update
        v = y_obs[t] - Lambda @ f_pred
        S = Lambda @ P_pred @ Lambda.T + H
        K = np.linalg.solve(S, Lambda @ P_pred).T
        f = f_pred + K @ v
        P = (np.eye(N_STATES) - K @ Lambda) @ P_pred
        P = 0.5 * (P + P.T)
        factors[t] = f

    return factors
