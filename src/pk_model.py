"""
Two-compartment pharmacokinetic model.

Compartment equations (IV bolus):
    dP/dt = -(k0 + k1)*P + k2*B
    dB/dt =        k1*P  - k2*B

Analytical solution (Eqs. 16-17 from paper):
    alpha + beta = k0 + k1 + k2
    alpha * beta = k0 * k2

    P(t) = P0 / (beta - alpha) * [(k2 - alpha)*exp(-alpha*t)
                                   - (k2 - beta)*exp(-beta*t)]

Discriminant identity:
    (k0+k1+k2)^2 - 4*k0*k2 = (k0+k1-k2)^2 + 4*k1*k2 > 0  for k1,k2 > 0
so beta > alpha > 0 is guaranteed for any positive rate constants.

Parametrisation notes
---------------------
PK rate constants span many orders of magnitude across individuals and are
strictly positive, making log-normal distributions the natural choice
(standard in NONMEM, Stan PK models, etc.).  We use the non-centred
parametrisation to improve sampler geometry in the hierarchical model.

From the data: observed peak concentrations range 86–588 ng/mL and the
overall elimination half-life is roughly 5–10 min, implying combined
decay rates ~0.1–0.3 min⁻¹.  Priors are centred on these scales.
"""
import numpy as np
import pytensor.tensor as pt
import pymc as pm


# ---------------------------------------------------------------------------
# NumPy version — for post-hoc plotting and interpolation
# ---------------------------------------------------------------------------

def alpha_beta_np(k0, k1, k2):
    """Return (alpha, beta) eigenvalues — numpy, element-wise."""
    s = k0 + k1 + k2
    disc = np.sqrt(np.maximum(s**2 - 4.0 * k0 * k2, 1e-30))
    return (s - disc) / 2.0, (s + disc) / 2.0


def plasma_concentration_np(t, P0, k0, k1, k2):
    """
    P(t) — numpy version.

    Parameters
    ----------
    t   : array-like, times in minutes
    P0  : scalar or array, initial plasma concentration (ng/mL)
    k0, k1, k2 : scalar or array rate constants (min^-1)
    """
    alpha, beta = alpha_beta_np(k0, k1, k2)
    denom = beta - alpha
    return P0 / denom * (
        (k2 - alpha) * np.exp(-alpha * np.asarray(t))
        - (k2 - beta) * np.exp(-beta * np.asarray(t))
    )


# ---------------------------------------------------------------------------
# PyTensor version — for use inside PyMC model
# ---------------------------------------------------------------------------

def alpha_beta_pt(k0, k1, k2):
    """Return (alpha, beta) eigenvalues — pytensor tensors."""
    s = k0 + k1 + k2
    disc = pt.sqrt(s**2 - 4.0 * k0 * k2)
    return (s - disc) / 2.0, (s + disc) / 2.0


def plasma_concentration_pt(t, P0, k0, k1, k2):
    """P(t) — pytensor version for use inside a PyMC model."""
    alpha, beta = alpha_beta_pt(k0, k1, k2)
    denom = beta - alpha
    return P0 / denom * (
        (k2 - alpha) * pt.exp(-alpha * t)
        - (k2 - beta) * pt.exp(-beta * t)
    )


# ---------------------------------------------------------------------------
# PyMC partial-pooling hierarchical model  (log-normal, non-centred)
# ---------------------------------------------------------------------------

def build_model(data: dict) -> pm.Model:
    """
    Build the Bayesian two-compartment PK model with partial pooling.

    Parametrisation
    ---------------
    PK parameters are log-normally distributed across subjects.  We use the
    non-centred (Matt trick) reparametrisation to avoid funnel pathologies:

        log_k0_mu   ~ Normal(log(0.3), 1.0)   # population log-mean
        log_k0_sigma ~ HalfNormal(0.7)         # between-subject log-SD
        k0_offset[i] ~ Normal(0, 1)            # standardised subject offset
        k0[i] = exp(log_k0_mu + log_k0_sigma * k0_offset[i])

    Likewise for k1, k2, P0.

    Priors are centred on physiologically plausible values:
        k0, k1, k2 ~ 0.3 min⁻¹  (half-life ~ 2 min; heavy tails allow slower)
        P0         ~ 200 ng/mL   (covers the 86–588 ng/mL observed peaks)

    Observation noise
    -----------------
        plasma_sigma ~ HalfNormal(50)   # 50 ng/mL scale; avoids the
                                         # degenerate large-sigma solution

    Parameters
    ----------
    data : dict returned by data_loader.load_data()

    Returns
    -------
    pm.Model
    """
    subject_ids = data["subject_ids"]
    t_obs = data["t_obs_flat"]
    P_obs = data["P_obs_flat"]
    subj_idx = data["subj_idx_flat"]

    coords = {"subject": subject_ids}

    # Log-centre priors from observed data
    log_k_center = np.log(0.3)   # 0.3 min⁻¹  →  half-life ~ 2.3 min
    log_P0_center = np.log(200)  # 200 ng/mL

    with pm.Model(coords=coords) as model:

        # ------------------------------------------------------------------ #
        # Population-level log-means (hyperpriors)
        # ------------------------------------------------------------------ #
        log_k0_mu = pm.Normal("log_k0_mu", mu=log_k_center, sigma=1.5)
        log_k1_mu = pm.Normal("log_k1_mu", mu=log_k_center, sigma=1.5)
        log_k2_mu = pm.Normal("log_k2_mu", mu=log_k_center, sigma=1.5)
        log_P0_mu = pm.Normal("log_P0_mu", mu=log_P0_center, sigma=1.5)

        # Between-subject variability (log-scale SD)
        log_k0_sigma = pm.HalfNormal("log_k0_sigma", sigma=0.7)
        log_k1_sigma = pm.HalfNormal("log_k1_sigma", sigma=0.7)
        log_k2_sigma = pm.HalfNormal("log_k2_sigma", sigma=0.7)
        log_P0_sigma = pm.HalfNormal("log_P0_sigma", sigma=0.7)

        # ------------------------------------------------------------------ #
        # Non-centred subject offsets (standard Normal)
        # ------------------------------------------------------------------ #
        k0_offset = pm.Normal("k0_offset", mu=0.0, sigma=1.0, dims="subject")
        k1_offset = pm.Normal("k1_offset", mu=0.0, sigma=1.0, dims="subject")
        k2_offset = pm.Normal("k2_offset", mu=0.0, sigma=1.0, dims="subject")
        P0_offset = pm.Normal("P0_offset", mu=0.0, sigma=1.0, dims="subject")

        # ------------------------------------------------------------------ #
        # Subject-level parameters (log-normal, positive by construction)
        # ------------------------------------------------------------------ #
        k0 = pm.Deterministic(
            "k0", pt.exp(log_k0_mu + log_k0_sigma * k0_offset), dims="subject"
        )
        k1 = pm.Deterministic(
            "k1", pt.exp(log_k1_mu + log_k1_sigma * k1_offset), dims="subject"
        )
        k2 = pm.Deterministic(
            "k2", pt.exp(log_k2_mu + log_k2_sigma * k2_offset), dims="subject"
        )
        P0 = pm.Deterministic(
            "P0", pt.exp(log_P0_mu + log_P0_sigma * P0_offset), dims="subject"
        )

        # ------------------------------------------------------------------ #
        # Observation-level parameter lookup
        # ------------------------------------------------------------------ #
        k0_obs = k0[subj_idx]
        k1_obs = k1[subj_idx]
        k2_obs = k2[subj_idx]
        P0_obs = P0[subj_idx]

        t = pm.Data("t_obs", t_obs)

        # ------------------------------------------------------------------ #
        # Predicted concentration for each observation
        # ------------------------------------------------------------------ #
        y_pred = pm.Deterministic(
            "y_pred",
            plasma_concentration_pt(t, P0_obs, k0_obs, k1_obs, k2_obs),
        )

        # ------------------------------------------------------------------ #
        # Observation noise  (scale informed by data range ~50 ng/mL)
        # ------------------------------------------------------------------ #
        plasma_sigma = pm.HalfNormal("plasma_sigma", sigma=50.0)

        # ------------------------------------------------------------------ #
        # Likelihood
        # ------------------------------------------------------------------ #
        pm.Normal("P_obs", mu=y_pred, sigma=plasma_sigma, observed=P_obs)

    return model
