# Implementation Plan: Two-Compartment PK Model with Partial Pooling

## Objective

Fit a Bayesian two-compartment pharmacokinetic (PK) model with partial pooling to the
DMT plasma concentration data (`data/plasma_clean.csv`) and produce smooth interpolated
plasma concentration curves for every subject across the entire session.

---

## Data overview

| Column        | Description                                      |
|---------------|--------------------------------------------------|
| `subject`     | Subject ID (S01–S13)                             |
| `condition`   | `dmt` or `placebo`                               |
| `dose_mg`     | Administered dose (7, 14, 18, 20 mg) or NaN     |
| `time_point`  | Ordinal index 0–4                                |
| `time_min`    | Actual time in minutes (measurement times vary)  |
| `plasma_conc` | Observed DMT plasma concentration (ng/mL)        |
| `is_imputed`  | Whether the row was imputed                      |

We will use **only the `dmt` condition** (13 subjects, 5 sparse time points each).
Placebo rows have no drug dynamics and serve only as baseline reference.

---

## Step 1 — Environment setup

**File:** `requirements.txt` (or `pyproject.toml`)

Dependencies:
```
pymc>=5.0
numpy
pandas
scipy
matplotlib
arviz
```

Install:
```bash
pip install pymc arviz matplotlib pandas scipy
```

---

## Step 2 — Data loading and preprocessing

**File:** `src/data_loader.py`

Tasks:
1. Load `data/plasma_clean.csv`.
2. Filter to `condition == 'dmt'`.
3. Build per-subject arrays: `time_min` and `plasma_conc`.
4. Create integer subject index array (`subject_idx`) for indexing into
   per-subject parameter vectors.
5. Assemble flat arrays (all subjects concatenated) for PyMC likelihood:
   - `t_obs_flat`  — observed times, shape `(N_total,)`
   - `P_obs_flat`  — observed plasma concentrations, shape `(N_total,)`
   - `subj_idx_flat` — integer subject index, shape `(N_total,)`
6. Store subject metadata: list of subject IDs, n_subjects, per-subject
   observation counts.

---

## Step 3 — Analytical solution of the two-compartment ODE

**File:** `src/pk_model.py`

The two-compartment system (IV bolus administration):

```
dP/dt = -(k0 + k1)*P(t) + k2*B(t)
dB/dt =           k1*P(t) - k2*B(t)
```

Analytical solution (equations 16–17 from the paper):

```
α + β = k0 + k1 + k2
α * β = k2 * k0

P(t) = P(0) / (β - α) * [(k2 - α)*exp(-α*t) - (k2 - β)*exp(-β*t)]
B(t) = k1*P(0) / (β - α) * [exp(-α*t) - exp(-β*t)]
```

Implementation notes:
- `α` and `β` are the two roots of the characteristic equation, computed from
  `k0`, `k1`, `k2` via the quadratic formula.
- Ensure `β > α` (swap if necessary) so the denominator `β - α > 0`.
- The function must be compatible with PyMC/PyTensor tensor arithmetic
  (use `pytensor.tensor` operations, not numpy, inside the PyMC model).
- Expose a pure-numpy version for post-hoc plotting and interpolation.

Function signatures:
```python
def alpha_beta(k0, k1, k2):
    """Returns (alpha, beta) eigenvalues."""

def plasma_concentration(t, P0, k0, k1, k2):
    """P(t) — works with both numpy arrays and pytensor tensors."""
```

Note: `B(t)` (Eq. 17) is **not implemented** — we have no brain/EEG data to fit it against.

---

## Step 4 — Bayesian partial pooling model in PyMC

**File:** `src/pk_model.py` (continued) or `src/model.py`

### Graphical model structure (from paper, Figure 20)

```
Hyperpriors (population-level, scalar):
    k0_sigma   ~ Exponential(1)
    k1_sigma   ~ Exponential(1)
    k2_sigma   ~ Exponential(1)
    P0_sigma   ~ Exponential(1)      # y_init_sigma in paper

Subject-level parameters (one per subject, shape = [n_subjects]):
    k0[i]  ~ HalfCauchy(k0_sigma)
    k1[i]  ~ HalfCauchy(k1_sigma)
    k2[i]  ~ HalfCauchy(k2_sigma)
    P0[i]  ~ HalfNormal(P0_sigma)   # initial plasma concentration

Deterministic intermediate variables:
    alpha[i], beta[i] = alpha_beta(k0[i], k1[i], k2[i])
    y_pred[i, t]      = plasma_concentration(t, P0[i], k0[i], k1[i], k2[i])

Observation noise:
    plasma_sigma ~ HalfCauchy(1)

Likelihood (only plasma observed):
    P_obs[i, t] ~ Normal(mu = y1[subj_idx, t], sigma = plasma_sigma)
```

### PyMC implementation checklist

1. Use `coords` to name dimensions: `subject`, `obs`.
2. Declare hyperpriors as scalar random variables.
3. Declare per-subject params with `dims="subject"`.
4. Index predicted values with `subj_idx_flat` to align with flat obs arrays.
5. Use `pm.Deterministic` to expose `alpha`, `beta`, `y_pred` for diagnostics.
6. Add `plasma_sigma ~ pm.HalfCauchy(1)` as a shared noise parameter.
7. Likelihood: `pm.Normal("P_obs", mu=y_pred, sigma=plasma_sigma, observed=P_obs_flat)`.

---

## Step 5 — MCMC sampling

**File:** `src/run_inference.py`

```python
with model:
    trace = pm.sample(
        draws=1000,
        chains=2,
        target_accept=0.9,    # increase if divergences
        return_inferencedata=True,
    )
```

Save trace to disk:
```python
import arviz as az
az.to_netcdf(trace, "results/trace.nc")
```

Diagnostics to check:
- `az.summary(trace)` — R-hat < 1.01, ESS > 400 for all parameters.
- `az.plot_trace(trace)` — visual chain mixing.
- Count divergences: `trace.sample_stats.diverging.sum()`.

---

## Step 6 — Posterior predictive interpolation

**File:** `src/interpolation.py`

Goal: produce smooth plasma concentration curves from `t=0` to `t=t_max`
at fine resolution (e.g., 500 points) for each subject.

Algorithm:
1. Load trace (`az.from_netcdf`).
2. Extract posterior samples of `k0`, `k1`, `k2`, `P0` per subject
   (shape: `[chain, draw, subject]`).
3. Flatten chain × draw into one sample dimension.
4. For each subject `i`, for each posterior sample `s`:
   - Compute `P(t_grid, P0[s,i], k0[s,i], k1[s,i], k2[s,i])`.
5. Compute per-subject statistics over the sample dimension:
   - `P_mean[i, t]` — posterior mean
   - `P_hdi_low[i, t]`, `P_hdi_high[i, t]` — 94% HDI (via `az.hdi`)
6. Return a DataFrame with columns:
   `subject, time_min, plasma_mean, plasma_hdi_low, plasma_hdi_high`

Save to: `results/plasma_interpolated.csv`

---

## Step 7 — Visualisation

**File:** `src/plot_results.py`

For each subject (DMT condition):
- Plot observed `plasma_conc` as scatter points (different marker for imputed).
- Overlay posterior mean curve (solid line).
- Shade 94% HDI band.
- Mark the administered dose in the title.

Grid layout: one subplot per subject (4×4 or similar).

Save to: `results/figures/posterior_predictions.png`

Additional plots:
- Box/violin plots of posterior `k0`, `k1`, `k2` per subject.
- Hyperparameter posteriors (`k0_sigma`, etc.).

---

## Step 8 — Output files

```
results/
├── trace.nc                      # ArviZ InferenceData (posterior samples)
├── plasma_interpolated.csv       # Smooth curves per subject
└── figures/
    ├── posterior_predictions.png
    ├── parameter_posteriors.png
    └── diagnostics/
        ├── trace_plots.png
        └── summary.csv
```

---

## File structure

```
src/
├── data_loader.py      # Step 2
├── pk_model.py         # Steps 3 & 4
├── run_inference.py    # Step 5
├── interpolation.py    # Step 6
└── plot_results.py     # Step 7
results/                # Created at runtime
```

---

## Key design decisions

| Decision | Rationale |
|----------|-----------|
| Partial pooling only (no pooled/unpooled) | Paper shows it has best out-of-sample performance |
| Fit only to plasma data (`P(t)`) | We have no LZc/EEG data; plasma_conc is the only observed variable |
| Analytical ODE solution (not numerical) | Faster, differentiable, no ODE solver needed inside MCMC |
| HalfCauchy priors on rate params | Heavy-tailed, enforces positivity, follows paper's design |
| Exponential hyperpriors on sigmas | Standard choice for hierarchical scale parameters |
| Shared `plasma_sigma` across subjects | Simpler; can be extended to per-subject if needed |
| 2 chains × 1000 draws | Follows paper; increase to 4 × 2000 if R-hat issues |
