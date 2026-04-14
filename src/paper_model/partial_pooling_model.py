import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model(t, lzc, subject_idx, n_subjects, plasma_data=None,
                observe_lzc=True):
    """Build the PyMC partial pooling two-compartment model with Hill equation.

    Args:
        plasma_data: optional dict with keys 'plasma_t', 'plasma_conc',
            'plasma_subj_idx'. When provided, y1 is computed at the plasma
            time points and fitted to the observed plasma concentrations.
            When None, plasma_obs is an unobserved latent variable.
        observe_lzc: if True (default), y2 is fitted to the LZc data.
            Set to False for plasma-only fitting.
    """
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

        # Index into per-subject parameters (for LZc time grid)
        k1_obs = k1[subj_idx]
        k2_obs = k2[subj_idx]
        y_init_obs = y_init[subj_idx]
        alpha_obs = alpha[subj_idx]
        beta_obs = beta[subj_idx]

        # t_data is already time-since-injection (t=0 at injection)
        t_eff = t_data

        # eq 16: plasma compartment (computed at LZc time points)
        y1 = (y_init_obs / (beta_obs - alpha_obs)
              * ((k2_obs - alpha_obs) * pt.exp(-alpha_obs * t_eff)
                 - (k2_obs - beta_obs) * pt.exp(-beta_obs * t_eff)))
        # Brain compartment analytical solution (eq. 17)
        y2 = (
            k1_obs * y_init_obs / (beta_obs - alpha_obs)
            * (pt.exp(-alpha_obs * t_eff) - pt.exp(-beta_obs * t_eff))
        )

        # Hill tissue response equation: E/E_max = [A]^n / (EC50^n + [A]^n)
        # E_max = 1, E = lz, [A] = y2
        EC50 = pm.HalfCauchy("EC50", beta=1.0)
        n_hill = pm.HalfNormal("n_hill", sigma=2.0)
        y2_safe = pt.maximum(y2, 0.0)
        lz_predicted = y2_safe**n_hill / (EC50**n_hill + y2_safe**n_hill)

        # Observation noise
        lz_sigma = pm.HalfCauchy("lz_sigma", beta=1.0)
        plasma_sigma = pm.HalfCauchy("plasma_sigma", beta=1.0)

        # Likelihood — brain (LZc) via Hill equation
        if observe_lzc:
            pm.Normal("lz_obs", mu=lz_predicted, sigma=lz_sigma, observed=lzc)
        else:
            pm.Normal("lz_obs", mu=lz_predicted, sigma=lz_sigma)

        # Likelihood — plasma
        if plasma_data is not None:
            # Compute y1 at the plasma-specific time points
            plasma_t_data = pm.Data("plasma_t_data", plasma_data["plasma_t"])
            plasma_subj = pm.Data("plasma_subj_idx", plasma_data["plasma_subj_idx"])

            k2_p = k2[plasma_subj]
            y_init_p = y_init[plasma_subj]
            alpha_p = alpha[plasma_subj]
            beta_p = beta[plasma_subj]

            y1_plasma = (
                y_init_p / (beta_p - alpha_p)
                * ((k2_p - alpha_p) * pt.exp(-alpha_p * plasma_t_data)
                   - (k2_p - beta_p) * pt.exp(-beta_p * plasma_t_data))
            )
            # plasma_scale converts model units to ng/mL
            plasma_scale = pm.HalfNormal("plasma_scale", sigma=500)
            pm.Normal("plasma_obs", mu=y1_plasma * plasma_scale,
                      sigma=plasma_sigma,
                      observed=plasma_data["plasma_conc"])
        else:
            pm.Normal("plasma_obs", mu=y1, sigma=plasma_sigma)

    return model


def fit_model(model, draws=2000, chains=2):
    """Run MCMC and return InferenceData."""
    with model:
        trace = pm.sample(tune=2000, draws=draws, chains=chains, target_accept=0.9, cores=2, return_inferencedata=True)
    return trace


def _apply_hill(y2_samples, trace, ndim):
    """Apply Hill equation to y2 samples if parameters exist in trace."""
    if "EC50" not in trace.posterior or "n_hill" not in trace.posterior:
        return y2_samples

    EC50 = trace.posterior["EC50"].values.reshape(-1)
    n_hill = trace.posterior["n_hill"].values.reshape(-1)
    y2_safe = np.maximum(y2_samples, 0.0)

    if ndim == 3:
        EC50_exp = EC50[:, np.newaxis, np.newaxis]
        n_exp = n_hill[:, np.newaxis, np.newaxis]
    else:
        EC50_exp = EC50[:, np.newaxis]
        n_exp = n_hill[:, np.newaxis]

    return y2_safe**n_exp / (EC50_exp**n_exp + y2_safe**n_exp)


def compute_posterior_predictive(trace, t_grid, subject_idx_grid, n_subjects):
    """Compute LZc predictions on a fine time grid from posterior samples.

    When the trace contains Hill equation parameters (EC50, n_hill), the
    raw y2 (brain concentration) is transformed through the Hill equation.

    Returns:
        (posterior_curve_samples, posterior_predictive_samples)
    """
    # Extract posterior samples
    k0 = trace.posterior["k0"].values.reshape(-1, n_subjects)
    k1 = trace.posterior["k1"].values.reshape(-1, n_subjects)
    k2 = trace.posterior["k2"].values.reshape(-1, n_subjects)
    y_init = trace.posterior["y_init"].values.reshape(-1, n_subjects)
    lz_sigma = trace.posterior["lz_sigma"].values.reshape(-1)

    # Compute alpha, beta per sample
    s = k0 + k1 + k2
    p = k0 * k2
    disc = np.maximum(s ** 2 - 4.0 * p, 1e-12)
    sqrt_disc = np.sqrt(disc)
    alpha = (s - sqrt_disc) / 2.0
    beta = (s + sqrt_disc) / 2.0

    subj = np.atleast_1d(subject_idx_grid)
    k1_g = k1[:, subj]
    y_init_g = y_init[:, subj]
    alpha_g = alpha[:, subj]
    beta_g = beta[:, subj]

    # Handle broadcasting
    if len(subj) != len(t_grid):
        ndim = 3
        alpha_g = alpha_g[:, :, np.newaxis]
        beta_g = beta_g[:, :, np.newaxis]
        k1_g = k1_g[:, :, np.newaxis]
        y_init_g = y_init_g[:, :, np.newaxis]
        t_g = t_grid[np.newaxis, np.newaxis, :]
        lz_sigma_expanded = lz_sigma[:, np.newaxis, np.newaxis]
    else:
        ndim = 2
        t_g = t_grid[np.newaxis, :]
        lz_sigma_expanded = lz_sigma[:, np.newaxis]

    posterior_curve_samples = (
        k1_g * y_init_g / (beta_g - alpha_g)
        * (np.exp(-alpha_g * t_g) - np.exp(-beta_g * t_g))
    )

    # Apply Hill equation if parameters exist
    posterior_curve_samples = _apply_hill(posterior_curve_samples, trace, ndim)

    posterior_predictive_samples = np.random.normal(
        loc=posterior_curve_samples, scale=lz_sigma_expanded
    )

    return posterior_curve_samples, posterior_predictive_samples


def compute_posterior_predictive_y2_raw(trace, t_grid, subject_idx_grid, n_subjects):
    """Compute raw y2 (brain concentration) without Hill equation.

    Returns:
        posterior_curve_samples — raw brain drug concentration in model units.
    """
    k0 = trace.posterior["k0"].values.reshape(-1, n_subjects)
    k1 = trace.posterior["k1"].values.reshape(-1, n_subjects)
    k2 = trace.posterior["k2"].values.reshape(-1, n_subjects)
    y_init = trace.posterior["y_init"].values.reshape(-1, n_subjects)

    s = k0 + k1 + k2
    p = k0 * k2
    disc = np.maximum(s ** 2 - 4.0 * p, 1e-12)
    sqrt_disc = np.sqrt(disc)
    alpha = (s - sqrt_disc) / 2.0
    beta = (s + sqrt_disc) / 2.0

    subj = np.atleast_1d(subject_idx_grid)
    k1_g = k1[:, subj]
    y_init_g = y_init[:, subj]
    alpha_g = alpha[:, subj]
    beta_g = beta[:, subj]

    if len(subj) != len(t_grid):
        alpha_g = alpha_g[:, :, np.newaxis]
        beta_g = beta_g[:, :, np.newaxis]
        k1_g = k1_g[:, :, np.newaxis]
        y_init_g = y_init_g[:, :, np.newaxis]
        t_g = t_grid[np.newaxis, np.newaxis, :]
    else:
        t_g = t_grid[np.newaxis, :]

    return (
        k1_g * y_init_g / (beta_g - alpha_g)
        * (np.exp(-alpha_g * t_g) - np.exp(-beta_g * t_g))
    )


def compute_posterior_predictive_y1(trace, t_grid, subject_idx_grid, n_subjects):
    """Compute y1 (plasma/DMT) predictions on a fine time grid from posterior samples.

    Returns:
        (posterior_curve_samples, posterior_predictive_samples)
    """
    k0 = trace.posterior["k0"].values.reshape(-1, n_subjects)
    k1 = trace.posterior["k1"].values.reshape(-1, n_subjects)
    k2 = trace.posterior["k2"].values.reshape(-1, n_subjects)
    y_init = trace.posterior["y_init"].values.reshape(-1, n_subjects)
    plasma_sigma = trace.posterior["plasma_sigma"].values.reshape(-1)

    s = k0 + k1 + k2
    p = k0 * k2
    disc = np.maximum(s ** 2 - 4.0 * p, 1e-12)
    sqrt_disc = np.sqrt(disc)
    alpha = (s - sqrt_disc) / 2.0
    beta = (s + sqrt_disc) / 2.0

    subj = np.atleast_1d(subject_idx_grid)
    k2_g = k2[:, subj]
    y_init_g = y_init[:, subj]
    alpha_g = alpha[:, subj]
    beta_g = beta[:, subj]

    if len(subj) != len(t_grid):
        alpha_g = alpha_g[:, :, np.newaxis]
        beta_g = beta_g[:, :, np.newaxis]
        k2_g = k2_g[:, :, np.newaxis]
        y_init_g = y_init_g[:, :, np.newaxis]
        t_g = t_grid[np.newaxis, np.newaxis, :]
        plasma_sigma_expanded = plasma_sigma[:, np.newaxis, np.newaxis]
    else:
        t_g = t_grid[np.newaxis, :]
        plasma_sigma_expanded = plasma_sigma[:, np.newaxis]

    # eq 16: y1
    posterior_curve_samples = (
        y_init_g / (beta_g - alpha_g)
        * ((k2_g - alpha_g) * np.exp(-alpha_g * t_g)
           - (k2_g - beta_g) * np.exp(-beta_g * t_g))
    )

    posterior_predictive_samples = np.random.normal(
        loc=posterior_curve_samples, scale=plasma_sigma_expanded
    )

    return posterior_curve_samples, posterior_predictive_samples
