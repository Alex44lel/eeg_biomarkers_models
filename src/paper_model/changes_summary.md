# Partial Pooling Model with Hill Equation — Changes Summary

## Overview

We extended the two-compartment pharmacokinetic (PK) model to jointly fit **plasma DMT concentrations** (ng/mL) and **brain complexity (LZc)** using a **Hill dose-response equation**. The model estimates:

- **y1**: the DMT plasma concentration curve for each subject
- **y2**: the DMT brain concentration curve for each subject
- **Hill(y2)**: the predicted LZc response, linking brain drug levels to neural complexity

---

## 1. LZc Computation (`src/lzc/compute_lzc.py`)

### Change: Surrogate-normalized LZ76

**Before:** LZ76 complexity was normalized by the asymptotic bound `n/log2(n)`.

**After:** Normalized by the complexity of a **shuffled surrogate** of the same binary sequence (Casali et al. 2013).

```python
# Before
return len(sub_strings) / (n / np.log2(n))

# After
c = _lz76_count(s, n)
shuffled = np.random.permutation(binary_array).astype(np.uint8)
c_surr = _lz76_count(shuffled.tobytes(), n)
return min(c / c_surr, 1.0)
```

**Why:** The `n/log2(n)` normalizer is an asymptotic bound from the original Lempel-Ziv 1976 paper. It only becomes tight as sequence length n approaches infinity. For our short sequences (n=600 after decimating 3-second EEG trials from 1000Hz to 200Hz), the bound is too loose — every single EEG trial produced values above 1.0, and capping at 1.0 destroyed all variance (everything became exactly 1.0).

Surrogate normalization (Casali et al. 2013) takes a different approach: shuffle the binary sequence randomly to destroy all temporal structure, then compute LZ76 on that shuffled version. The shuffled sequence has the maximum possible complexity for a sequence with that same proportion of 0s and 1s. Dividing by this surrogate complexity gives a ratio in [0, 1] that measures "how complex is this signal relative to a random signal with the same statistics."

**Result:** LZc values now range from ~0.63 to ~0.87 across subjects, with visible increases after DMT injection in some subjects.

### Change: Removed baseline normalization

**Before:** LZc was computed as percentage change from baseline mean: `(lzc - baseline_mean) / baseline_mean * 100`.

**After:** Only raw LZc values (`lzc_raw`) are stored. The baseline computation was removed entirely.

**Why:** The Hill equation with E_max=1 requires the observed response (LZc) to be in [0, 1]. The surrogate-normalized LZ76 values already satisfy this naturally. Percentage-change normalization produced values in a completely different scale (~-5% to +25%) that was incompatible with E_max=1. Since we only use post-injection data in the model, the baseline period serves no purpose.

---

## 2. Data Preparation (`src/paper_model/prepare_data.py`)

### Change: Time axis shifted to injection-relative

**Before:** `t_model = time_min` (absolute time from recording start). A separate `--only-after-injection` CLI flag filtered post-injection data at runtime.

**After:** `t_model = time_min - injection_time_min` per subject, with automatic filtering to `t >= 0`.

```python
df["t_model"] = df["time_min"] - df["injection_time_min"]
df = df[df["t_model"] >= 0].copy()
```

**Why:** The two-compartment PK equations (eqs. 16-17) assume t=0 is when the drug enters the bloodstream. Using absolute recording time meant the exponential curves were evaluated at the wrong time reference — e.g., if injection happens at minute 5, computing y1(5) gives a decayed concentration when it should give the peak. Shifting so t=0=injection makes y_init genuinely represent the initial plasma concentration, and the exponential rise/decay dynamics start from the correct reference. This also eliminates the need for a separate filtering flag — post-injection data is guaranteed by the `t >= 0` filter.

### Change: LZc uses raw surrogate-normalized values

**Before:** `df["lzc"] = (df["lzc_normalized"] + 5) / 10` — an ad-hoc rescaling that put values in [0, ~3].

**After:** `df["lzc"] = df["lzc_raw"]` — direct use of surrogate-normalized LZ76, already in [0, 1].

**Why:** The Hill equation output is bounded by [0, E_max] = [0, 1]. If the observed data exceeds 1.0, the model can never fit those points. The old rescaling produced values up to ~3.0, causing the Hill equation to saturate (always predict 1.0) and the EC50 to collapse to near-zero. Using raw LZc in [0, 1] makes the data directly compatible with the Hill equation.

### Change: Plasma data in raw ng/mL

Plasma concentrations are returned without any normalization, in their original ng/mL units. The model fits these values directly (see `y_init` prior below).

### Excluded subjects

Currently: **S03, S04, S08, S09, S11** — configured in `EXCLUDED_SUBJECTS`. S04 was removed for being a low-dose outlier; S08 caused convergence issues (rate parameters exploded); S03, S09, S11 were excluded per the original paper.

---

## 3. Model (`src/paper_model/partial_pooling_model.py`)

### The Two-Compartment PK Model (unchanged structure)

The analytical solutions for an IV bolus two-compartment model:

**Plasma (eq. 16):**
$$y_1(t) = \frac{y_{\text{init}}}{\beta - \alpha} \left[ (k_2 - \alpha) e^{-\alpha t} - (k_2 - \beta) e^{-\beta t} \right]$$

**Brain (eq. 17):**
$$y_2(t) = \frac{k_1 \cdot y_{\text{init}}}{\beta - \alpha} \left[ e^{-\alpha t} - e^{-\beta t} \right]$$

Where alpha and beta are derived from k0, k1, k2 via the quadratic formula:
- alpha + beta = k0 + k1 + k2
- alpha * beta = k0 * k2

Both y1 and y2 are proportional to y_init. Since y_init is in ng/mL (see prior below), both compartments output values in ng/mL.

### Change: Hill tissue response equation

**Added between y2 and the LZc likelihood:**

```python
EC50 = pm.LogNormal("EC50", mu=np.log(100.0), sigma=1.0)
n_hill = pm.LogNormal("n_hill", mu=0, sigma=0.5)
y2_safe = pt.maximum(y2, 0.0)
lz_predicted = y2_safe**n_hill / (EC50**n_hill + y2_safe**n_hill)
```

**The Hill equation:**

$$\frac{E}{E_{\max}} = \frac{[A]^n}{\text{EC}_{50}^n + [A]^n}$$

| Variable | Model mapping | Meaning |
|----------|--------------|---------|
| E | LZc | Observed brain complexity response |
| E_max | 1 (fixed) | Maximum possible response |
| [A] | y2 | Brain drug concentration (ng/mL) |
| EC50 | fitted | Concentration producing 50% of max effect |
| n | fitted | Hill coefficient (steepness of dose-response) |

**Why:** The previous model assumed `LZc ≈ y2` — a direct linear relationship between brain drug concentration and neural complexity. This is pharmacologically naive. Biological responses to drug concentration typically follow a sigmoidal dose-response: negligible effect at low doses, steep increase around some threshold, then saturation. The Hill equation is the standard model for this relationship in pharmacology. It introduces two interpretable parameters: EC50 (the potency — how much drug is needed for half-max effect) and n (the cooperativity — how switch-like the response is).

### Prior: `EC50 ~ LogNormal(mu=log(100), sigma=1.0)`

**Intuition:** EC50 is the brain DMT concentration (in ng/mL) at which LZc reaches 50% of its maximum. Since y2 (brain concentration) is in ng/mL with typical values of ~100-500 ng/mL at peak, EC50 should be in a similar range.

- `mu=log(100)` centers the prior on **100 ng/mL**
- `sigma=1.0` gives a 95% prior interval of roughly **[5, 2000] ng/mL** — wide enough to be learned from data
- **LogNormal cannot reach 0**, which prevents the degenerate solution where EC50→0 causes the Hill equation to saturate at 1.0 for all concentrations

**What went wrong with the old prior:** `HalfCauchy(beta=1.0)` allowed EC50 to collapse to near-zero. When EC50≈0, the Hill equation becomes `y2^n / (0 + y2^n) = 1.0` for all y2>0 — a useless constant that ignores drug concentration entirely.

### Prior: `n_hill ~ LogNormal(mu=0, sigma=0.5)`

**Intuition:** The Hill coefficient controls the steepness of the dose-response curve.

- `mu=0` centers the prior on `exp(0) = 1.0` — the standard Michaelis-Menten kinetics
- `sigma=0.5` gives a 95% prior interval of roughly **[0.4, 2.7]**

| n_hill value | Curve behavior |
|-------------|----------------|
| ~0.5 | Very gradual response, drug effect ramps up slowly |
| ~1.0 | Hyperbolic (Michaelis-Menten) — the "default" assumption |
| ~2.0 | Sigmoidal — more switch-like, cooperative binding |

- **LogNormal cannot reach 0**, which prevents the degenerate solution where n→0 makes the Hill equation a flat constant (insensitive to concentration)

**What went wrong with the old prior:** `HalfNormal(sigma=2.0)` had substantial mass near 0. The sampler found n_hill≈0.09, at which point `y2^0.09` is nearly the same for all y2 values — the Hill equation lost all dose-response sensitivity and just predicted a constant ~0.77 for all time points.

### Prior: `y_init_sigma ~ Exponential(lam=0.002)`

**Intuition:** `y_init` represents the initial plasma DMT concentration at injection in ng/mL. Different subjects received different doses:
- 7 mg dose → first measured concentration ~94 ng/mL
- 20 mg dose → first measured concentration ~588 ng/mL

So y_init needs to cover the range ~50-700 ng/mL across subjects.

- `Exponential(lam=0.002)` has **mean = 500**, allowing y_init_sigma to be large enough for the per-subject y_init values to reach hundreds of ng/mL
- The old prior `Exponential(lam=1.0)` had mean=1, constraining y_init to ~1-2 — completely incompatible with ng/mL plasma data

The partial pooling structure `y_init[i] ~ HalfNormal(sigma=y_init_sigma)` means:
- All subjects share the same scale (y_init_sigma), learned from data
- Each subject gets their own y_init[i] that can differ based on their dose

### Prior: `plasma_sigma ~ HalfCauchy(beta=50.0)`

**Intuition:** This is the observation noise for plasma measurements in ng/mL. With plasma concentrations ranging from ~6 to ~588 ng/mL, a noise scale of ~50-100 ng/mL is reasonable (representing measurement error and inter-sample variability). The `beta=50.0` scale parameter gently suggests this range without being overly restrictive.

### Plasma likelihood: direct ng/mL fit

```python
pm.Normal("plasma_obs", mu=y1_plasma, sigma=plasma_sigma,
          observed=plasma_data["plasma_conc"])
```

**Why no `plasma_scale` parameter?** In an earlier iteration, we used a `plasma_scale` parameter to convert model units to ng/mL. This was needed when y_init was small (~1-2). Now that y_init is directly in ng/mL (thanks to the wider `y_init_sigma` prior), y1 naturally outputs ng/mL values and can be compared directly to the plasma data. This is simpler and avoids the identifiability issue between `plasma_scale` and `y_init` (the model couldn't distinguish "large y_init × small scale" from "small y_init × large scale").

### MCMC sampling configuration

```python
pm.sample(tune=1000, draws=draws, chains=chains, target_accept=0.95, cores=chains)
```

- **target_accept=0.95** (up from 0.9): The posterior has complex geometry from the hierarchical structure and nonlinear Hill equation. Higher target_accept reduces the NUTS step size, improving exploration of tight correlations at the cost of slower sampling.
- **cores=chains**: Enables parallel sampling — each chain runs on its own CPU core.

### Posterior predictive functions

Three compute functions for generating predictions on a fine time grid:

| Function | Returns | Used for |
|----------|---------|----------|
| `compute_posterior_predictive` | Hill(y2): LZc predictions | LZc vs observed plots |
| `compute_posterior_predictive_y2_raw` | Raw y2: brain concentration | Brain DMT plots |
| `compute_posterior_predictive_y1` | y1 + noise: plasma concentration | Plasma DMT plots |

`compute_posterior_predictive_y2_raw` was added because the main `compute_posterior_predictive` applies the Hill equation, returning LZc values in [0,1]. For brain concentration plots in ng/mL, we need the raw y2 before the Hill transformation.

---

## 4. Pipeline (`src/paper_model/run_pipeline.py`)

### Execution flow

1. **Load data** — LZc (post-injection, t=0 at injection) and optionally plasma (ng/mL)
2. **Build and fit** — or load existing trace from `--load-trace`
3. **Plot y1** — plasma DMT curves and predictive (ng/mL)
4. **Plot y2** — brain DMT curves (ng/mL, model units)
5. **Plot LZc** — predicted LZc (Hill output) vs observed LZc scatter
6. **Plot parameters** — per-subject boxplots of k0, k1, k2
7. **Parameter table** — summary statistics

### CLI flags

```
--observe-plasma    Use real plasma data as observed (recommended)
--plasma-only       Fit plasma data only, no LZc likelihood
--draws N           MCMC draws per chain (default 1000)
--chains N          Number of MCMC chains (default 2)
--load-trace PATH   Skip fitting, load pre-computed trace
```

### Removed

- `--only-after-injection` — post-injection filtering is now automatic in `prepare_data.py`

---

## 5. Plotting (`src/paper_model/plot_predictions.py`)

### y1 plots: Plasma DMT (ng/mL)

Single axis showing model y1 predictions and observed plasma data on the **same scale** (ng/mL). Since y_init is in ng/mL, the model's y1 output is directly in ng/mL — no scaling needed.

- **Curves plot**: posterior mean (black line) + 94% HDI (grey band) + plasma observations (red crosses)
- **Predictive plot**: same but with observation noise (`plasma_sigma`) added — shows where future measurements would fall

**Why single axis?** An earlier version used dual y-axes (model units on left, ng/mL on right). This was confusing because the two scales were arbitrary and hard to compare visually. With y_init in ng/mL, everything is on the same scale.

### y2 plots: Brain DMT concentration

Shows the estimated brain drug concentration from the PK model's eq. 17. These are the raw y2 values — the drug concentration in the brain compartment over time.

- **Shape**: starts at 0 (no drug in brain at injection), rises as drug transfers from plasma via k1, peaks, then decays as drug transfers back (k2) and is eliminated (k0)
- **No LZc overlay**: y2 is a drug concentration (ng/mL), LZc is a complexity measure [0,1] — comparing them on the same plot was misleading

### LZc plots: Predicted vs observed

Shows the Hill equation output (y2 → LZc) compared to observed LZc scatter.

- **Purple scatter**: observed LZc values per subject
- **Black line**: posterior mean of Hill(y2) — the model's predicted LZc
- **Grey band**: 94% HDI
- **y-axis**: [0, 1.05] since both predicted and observed LZc are in [0, 1]
- **Curves plot**: uncertainty in the mean prediction only
- **Predictive plot**: includes `lz_sigma` observation noise — shows where future LZc measurements would fall

### All plots: time axis labels

Added `ax.tick_params(labelbottom=True)` on all visible subplots. Without this, `sharex=True` causes matplotlib to hide tick labels on non-bottom rows.

---

## 6. Output Files

All saved in `results/paper_model/`:

| File | Content |
|------|---------|
| `partial_pooling_trace.nc` | ArviZ InferenceData (full MCMC trace) |
| `partial_pooling_y1_curves.png` | Plasma DMT — posterior mean curves |
| `partial_pooling_y1_predictive.png` | Plasma DMT — posterior predictive |
| `partial_pooling_y2_curves.png` | Brain DMT — posterior mean curves |
| `partial_pooling_y2_predictive.png` | Brain DMT — posterior predictive |
| `partial_pooling_lzc_curves.png` | LZc — predicted vs observed (curves) |
| `partial_pooling_lzc_predictive.png` | LZc — predicted vs observed (predictive) |
| `partial_pooling_parameters.png` | Per-subject parameter boxplots |
| `partial_pooling_param_table.png` | Parameter summary table |

---

## Running the Pipeline

```bash
# Recompute LZc from EEG (only needed once, or after changing compute_lzc.py)
env/bin/python -m src.lzc.compute_lzc

# Fit model and generate all plots
env/bin/python -m src.paper_model.run_pipeline --observe-plasma --draws 2000 --chains 4

# Load existing trace and regenerate plots only (no re-fitting)
env/bin/python -m src.paper_model.run_pipeline --observe-plasma --load-trace results/paper_model/partial_pooling_trace.nc
```
