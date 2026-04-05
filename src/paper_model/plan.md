# Partial Pooling Two-Compartment PK Model — Implementation Plan

## Goal

Implement the Bayesian partial pooling two-compartment pharmacokinetic model from
Section 5.3.3 of `interpolation_framework.pdf` (Figure 20), fit it to LZc data, and
reproduce Figures 26 and 27.

---

## The Model

### Two-compartment ODE (Section 5.2)

Plasma and brain compartments with rate parameters k0 (elimination), k1 (plasma→brain),
k2 (brain→plasma). IV bolus analytical solutions:

```
P(t) = P(0)/(β-α) * [(k2-α)e^(-αt) - (k2-β)e^(-βt)]     (eq. 16)
B(t) = k1*P(0)/(β-α) * [e^(-αt) - e^(-βt)]                (eq. 17)

where:  α + β = k1 + k2 + k0
        α * β = k2 * k0
```

P(0) = initial plasma concentration (y_init in the model).

### Partial pooling hierarchy (Section 5.3.3, Figure 20)

```
Hyperpriors (1 dim each):
    k0_sigma  ~ Exponential
    k1_sigma  ~ Exponential
    k2_sigma  ~ Exponential
    y_init_sigma ~ Exponential

Per-subject (N_subjects dims each):
    k0[i]     ~ HalfCauchy(k0_sigma)
    k1[i]     ~ HalfCauchy(k1_sigma)
    k2[i]     ~ HalfCauchy(k2_sigma)
    y_init[i] ~ HalfNormal(y_init_sigma)

Deterministic:
    alpha[i] + beta[i] = k0[i] + k1[i] + k2[i]
    alpha[i] * beta[i] = k0[i] * k2[i]
    => solve via quadratic formula

    y1[i](t) = plasma curve F1 (eq. 16)   [not observed]
    y2[i](t) = brain curve F2 (eq. 17)    [fitted to LZc]

Observation noise:
    lz_sigma     ~ HalfCauchy
    plasma_sigma ~ HalfCauchy   [present in graph but plasma not observed]

Likelihood:
    lz ~ Normal(y2, lz_sigma)   [observed = raw LZc post-injection]
```

### Sampler (Section 5.4)

- 2 chains, 1000 draws each
- PyMC NUTS sampler (default)

---

## Data

### Input files (already exist)

- `results/lzc/lzc_results.csv` — columns: `time_min, lzc_raw, lzc_normalized, baseline_mean, subject`
- `results/lzc/injection_offsets.csv` — columns: `subject, injection_time_min`

### Data preparation

- Use `lzc_raw` (not normalized) — the model fits absolute LZc values
- Build integer subject index for PyMC broadcasting
- The paper uses 10 subjects (excludes S03, S09, S11 per Figure 26). We include
  all 13 but make filtering configurable via `EXCLUDED_SUBJECTS`.

## Files to Create

All in `src/paper_model/`:

### 1. `prepare_data.py` — Data loading & transformation

```python
def load_and_prepare():
    """
    Returns dict with:
        t_model     : np.array — times in minutes (N_obs,)
        lzc_normalized    : np.array —
        subject_idx: np.array — integer subject index per obs (N_obs,)
        subject_names: list   — ordered subject labels
        baseline_means: np.array — mean pre-injection LZc per subject (N_subjects,)
    """
```

Steps:

1. Load lzc_results.csv

2. Build integer subject index (0, 1, 2, ...)
3. Optionally exclude subjects (S03, S09, S11)

### 2. `partial_pooling_model.py` — PyMC model definition & fitting

```python
def build_model(t, lzc, subject_idx, n_subjects, baseline_means):
    """Build the PyMC partial pooling two-compartment model."""

def fit_model(model, draws=1000, chains=2):
    """Run MCMC and return InferenceData."""

def compute_posterior_predictive(trace, t_grid, subject_idx_grid, n_subjects, baseline_means):
    """Compute y2 predictions on a fine time grid for plotting."""
```

Key implementation details:

- alpha, beta from quadratic: `s = k0+k1+k2`, `p = k0*k2`,
  `disc = s^2 - 4p`, `alpha = (s - sqrt(disc))/2`, `beta = (s + sqrt(disc))/2`
- Add small epsilon to disc for numerical stability: `disc = max(disc, 1e-12)`
- y2 = k1[subj_idx] _ y_init[subj_idx] / (beta[subj_idx] - alpha[subj_idx]) _
  (exp(-alpha[subj_idx]*t) - exp(-beta[subj_idx]*t)) + baseline_means[subj_idx]
- Likelihood: lz ~ Normal(mu=y2, sigma=lz_sigma, observed=lzc)

### 3. `plot_predictions.py` — Figure 26 replica

Per-subject grid plot:

- Purple scatter: observed normalize LZc vs time (post-injection)
- Black line: posterior mean of predicted LZc (y2) on a fine time grid
- Grey shading: 94% HDI of posterior predictive
- Layout: 4 rows x 3 cols (for 10 subjects), titled per subject
- X-axis: 0–14 min, Y-axis: 0–2.5 LZc
- Title: "Two Compartment: Partial Pooling Model Predictions vs Observed Data"

### 4. `plot_parameters.py` — Figure 27 replica

Three vertically stacked subplots (k0, k1, k2):

- Boxplot per subject from posterior samples
- Green line = median, green triangle = mean
- Annotate each box with μ=... σ=...
- Y-axis: "Value", X-axis: "Subject ID"
- Title per panel: "Partial Pooling Model k0/k1/k2 Box Plots"

### 5. `run_pipeline.py` — Entry point

```python
"""
Usage: python -m src.paper_model.run_pipeline [--draws 1000] [--chains 2] [--load-trace path]
"""
```

Steps:

1. Call prepare_data.load_and_prepare()
2. Call partial_pooling_model.build_model() + fit_model()
   (or load existing trace from --load-trace)
3. Save trace to results/paper_model/partial_pooling_trace.nc
4. Call plot_predictions (Figure 26)
5. Call plot_parameters (Figure 27)

---

## Output files (in `results/paper_model/`)

- `partial_pooling_trace.nc` — ArviZ InferenceData
- `partial_pooling_predictions.png` — Figure 26
- `partial_pooling_parameters.png` — Figure 27

---

## Implementation Order

| Step | File                       | Depends on |
| ---- | -------------------------- | ---------- |
| 1    | `prepare_data.py`          | —          |
| 2    | `partial_pooling_model.py` | Step 1     |
| 3    | `plot_predictions.py`      | Step 2     |
| 4    | `plot_parameters.py`       | Step 2     |
| 5    | `run_pipeline.py`          | All above  |

---

## Dependencies (already in requirements.txt)

- pymc >= 5.0
- arviz
- numpy, pandas, matplotlib, scipy
