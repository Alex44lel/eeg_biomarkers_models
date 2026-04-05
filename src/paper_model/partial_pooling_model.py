import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az


def build_model(t, lzc, subject_idx, n_subjects):
    """Build the PyMC partial pooling two-compartment model."""
    with pm.Model() as model:
        # Data containers
        t_data = pm.Data("t_data", t)
        subj_idx = pm.Data("subj_idx", subject_idx)

        # Hyperpriors on scale parameters
        k0_sigma = pm.Exponential("k0_sigma", lam=1.0)
        k1_sigma = pm.Exponential("k1_sigma", lam=1.0)
        k2_sigma = pm.Exponential("k2_sigma", lam=1.0)
        y_init_sigma = pm.Exponential("y_init_sigma", lam=1.0)

        # Per-subject rate parameters
        k0 = pm.HalfCauchy("k0", beta=k0_sigma, shape=n_subjects)
        k1 = pm.HalfCauchy("k1", beta=k1_sigma, shape=n_subjects)
        k2 = pm.HalfCauchy("k2", beta=k2_sigma, shape=n_subjects)
        y_init = pm.HalfNormal("y_init", sigma=y_init_sigma, shape=n_subjects)

        # Quadratic formula for alpha and beta
        s = k0 + k1 + k2
        p = k0 * k2
        disc = pt.maximum(s ** 2 - 4.0 * p, 1e-12)
        sqrt_disc = pt.sqrt(disc)
        alpha = (s - sqrt_disc) / 2.0
        beta = (s + sqrt_disc) / 2.0

        # Index into per-subject parameters
        k1_obs = k1[subj_idx]
        y_init_obs = y_init[subj_idx]
        alpha_obs = alpha[subj_idx]
        beta_obs = beta[subj_idx]

        # Clamp time: before injection (t<0) there is no drug, so B(t)=0
        t_eff = t_data

        # Brain compartment analytical solution (eq. 17)
        # No baseline shift needed — data is already baseline-normalized
        y2 = (
            k1_obs * y_init_obs / (beta_obs - alpha_obs)
            * (pt.exp(-alpha_obs * t_eff) - pt.exp(-beta_obs * t_eff))
        )

        # Observation noise
        lz_sigma = pm.HalfCauchy("lz_sigma", beta=1.0)

        # Likelihood
        pm.Normal("lz_obs", mu=y2, sigma=lz_sigma, observed=lzc)

    return model


def fit_model(model, draws=1000, chains=2):
    """Run MCMC and return InferenceData."""
    with model:
        trace = pm.sample(draws=draws, chains=chains, return_inferencedata=True)
    return trace


def compute_posterior_predictive(trace, t_grid, subject_idx_grid, n_subjects):
    """Compute y2 predictions on a fine time grid from posterior samples.

    Returns:
        predictions: np.array of shape (n_samples, len(t_grid))
    """
    # Extract posterior samples
    k0 = trace.posterior["k0"].values  # (chains, draws, n_subjects)
    k1 = trace.posterior["k1"].values
    k2 = trace.posterior["k2"].values
    y_init = trace.posterior["y_init"].values

    # Flatten chains and draws
    k0 = k0.reshape(-1, n_subjects)
    k1 = k1.reshape(-1, n_subjects)
    k2 = k2.reshape(-1, n_subjects)
    y_init = y_init.reshape(-1, n_subjects)

    # Compute alpha, beta per sample
    s = k0 + k1 + k2
    p = k0 * k2
    disc = np.maximum(s ** 2 - 4.0 * p, 1e-12)
    sqrt_disc = np.sqrt(disc)
    alpha = (s - sqrt_disc) / 2.0
    beta = (s + sqrt_disc) / 2.0

    # Compute predictions: shape (n_samples, n_time_points)
    subj = subject_idx_grid
    k1_g = k1[:, subj]
    y_init_g = y_init[:, subj]
    alpha_g = alpha[:, subj]
    beta_g = beta[:, subj]

    t_g = t_grid[np.newaxis, :]

    predictions = (
        k1_g * y_init_g / (beta_g - alpha_g)
        * (np.exp(-alpha_g * t_g) - np.exp(-beta_g * t_g))
    )

    return predictions
