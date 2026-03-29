# Model Summary: Bayesian Two-Compartment PK Model with Partial Pooling

## Problem

We have DMT plasma concentration measurements for 13 subjects, each with only 4 post-dose
time points (at approximately t = 1–3, 5, 10–14, and 20 minutes after IV injection).
The goal is to fit a pharmacokinetically meaningful model and reconstruct smooth, continuous
plasma concentration curves for every subject, with honest uncertainty quantification.

---

## Drug dynamics: the two-compartment ODE

The body is divided into two compartments:
- **Central (plasma)**: where DMT is injected and measured.
- **Peripheral (brain/tissue)**: where DMT redistributes after injection.

Drug moves between compartments and is eliminated from plasma according to:

```
dP/dt = -(k0 + k1) · P(t) + k2 · B(t)
dB/dt =          k1 · P(t) - k2 · B(t)
```

| Parameter | Meaning | Typical posterior value |
|-----------|---------|------------------------|
| `k0`      | Elimination rate from plasma (min⁻¹) | ~0.13 min⁻¹ (half-life ~5 min) |
| `k1`      | Transfer rate plasma → brain (min⁻¹) | Poorly identified (plasma data only) |
| `k2`      | Transfer rate brain → plasma (min⁻¹) | Poorly identified (plasma data only) |
| `P0`      | Initial plasma concentration at t=0 (ng/mL) | Scales with dose; ~200–850 ng/mL |

Because this is a linear ODE system, it has a closed-form solution (no numerical solver needed):

```
α, β  =  eigenvalues of the rate matrix
       =  [(k0+k1+k2) ± sqrt((k0+k1+k2)² - 4·k0·k2)] / 2

P(t)  =  P0 / (β - α) · [(k2 - α)·exp(-αt) − (k2 - β)·exp(-βt)]
```

The two-exponential shape captures the fast initial distribution phase (rate β)
and the slower elimination phase (rate α).

**Note on identifiability:** with only plasma data, `k1` and `k2` cannot be
identified separately — only their combined effect on the plasma decay shape
contributes to the likelihood. Their posteriors remain wide (prior-dominated).
Only `k0` and `P0` are well-determined by the data.

---

## Bayesian hierarchical model

### Why Bayesian?

With 4 observations per subject and 4 parameters to estimate per subject, classical
maximum likelihood would overfit badly. Bayesian inference regularises the estimates
via prior distributions and returns full posterior uncertainty rather than point estimates.

### Why hierarchical (partial pooling)?

The 13 subjects share the same biology but differ individually (different doses,
body weights, metabolic rates). Partial pooling is the optimal compromise:
- **Complete pooling** (one shared parameter set): ignores individual differences.
- **No pooling** (independent per-subject fits): too little data per subject to be reliable.
- **Partial pooling**: subjects have their own parameters, but those parameters
  are drawn from a shared population distribution. Subjects "borrow strength"
  from each other — a sparse subject is regularised toward the group mean (shrinkage).

### Model structure

```
Population level (hyperpriors):
    log_k0_mu   ~ Normal(log(0.3), 1.5)     ← population log-mean of k0
    log_k0_sigma ~ HalfNormal(0.7)          ← between-subject variability in k0
    (same structure for k1, k2, P0)

Subject level (non-centred parametrisation):
    k0_offset[i] ~ Normal(0, 1)
    k0[i]        = exp(log_k0_mu + log_k0_sigma · k0_offset[i])
    (same for k1[i], k2[i], P0[i])

Observation noise:
    plasma_sigma ~ HalfNormal(50)           ← in ng/mL

Likelihood:
    P_obs[j] ~ Normal(P(t[j], P0[subj[j]], k0[subj[j]], ...), plasma_sigma)
```

### Key design choices

**Log-normal parametrisation** for all PK parameters (not HalfCauchy as in the
original paper). PK rate constants are strictly positive and span orders of magnitude,
making log-normal the natural distribution. HalfCauchy priors led to a degenerate
MCMC solution (rates → thousands of min⁻¹, P0 → 0, sigma absorbing everything).

**Non-centred (Matt trick) parametrisation** for subject offsets. Instead of sampling
`k0[i] ~ LogNormal(mu, sigma)` directly, we sample a standardised offset and compute
`k0[i] = exp(mu + sigma · offset)`. This avoids the "funnel" geometry that causes
divergences in hierarchical models.

**Pre-dose baseline excluded from likelihood.** The observation at `time_point = 0`
(t = 0 min, plasma concentration ~0.2–2 ng/mL) is the pre-injection measurement.
The model sets `P(0) = P0` (the post-bolus concentration), which is necessarily large,
so including t = 0 in the likelihood creates a contradiction. Only the 4 post-dose
time points (time_points 1–4) are used for fitting.

---

## MCMC sampling

| Setting | Value |
|---------|-------|
| Sampler | NUTS (No-U-Turn Sampler) via PyMC 5 |
| Chains | 2 |
| Tuning draws | 1 000 |
| Posterior draws | 1 000 |
| `target_accept` | 0.9 |
| Total posterior samples | 2 000 |
| Divergences | 2 (< 0.1 %) |
| Runtime | ~7 seconds |

Posterior diagnostics (R-hat, ESS) are saved to `results/figures/diagnostics/summary.csv`.

---

## Posterior predictive interpolation

For each subject, the 2 000 posterior samples of `(k0, k1, k2, P0)` are used to
evaluate the analytical PK curve on a 500-point time grid from t = 0 to t = t_max.
The resulting 2 000 curves are summarised as:

- **Posterior mean**: the expected plasma concentration at each time point.
- **94% HDI band**: the shortest interval containing 94% of the posterior probability mass.

Output is saved to `results/plasma_interpolated.csv`.

---

## Output files

```
results/
├── trace.nc                             ← full posterior (ArviZ InferenceData)
├── plasma_interpolated.csv             ← smooth curves per subject (500 time points)
└── figures/
    ├── posterior_predictions.png       ← main result: curves + observations per subject
    ├── parameter_posteriors.png        ← subject-level k0, k1, k2, P0 (log scale)
    ├── hyperparameter_posteriors.png   ← population-level parameters
    └── diagnostics/
        ├── trace_plots.png
        └── summary.csv                 ← R-hat, ESS for all parameters
```
