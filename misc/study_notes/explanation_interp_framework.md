# Explanation: Bayesian Two-Compartment PK Model with Partial Pooling

This document explains every concept in the implementation from first principles —
why the model is structured this way, what each equation means biologically, and
why the Bayesian / partial-pooling approach is the right tool for this problem.

---

## 1. What problem are we solving?

We have sparse measurements of DMT plasma concentration for 13 subjects. Each subject
has only **5 time points**, sampled at irregular and subject-specific times. The goal
is to reconstruct a smooth, continuous plasma concentration curve for each subject —
essentially filling in all the time points we didn't measure.

Naive interpolation (e.g., cubic splines) would work mechanically but it has no
knowledge of the underlying biology. It cannot extrapolate beyond the measured range,
it has no uncertainty estimates, and it treats each subject in complete isolation.

The pharmacokinetic (PK) model encodes biological knowledge about how drugs move
through the body. By fitting this model to the data, we get:
1. Biologically meaningful curves (not just wiggly splines).
2. Principled uncertainty — we know *how confident* we are at each time point.
3. Information sharing across subjects via partial pooling.

---

## 2. Pharmacokinetics: what it models

Pharmacokinetics (PK) is the study of how a drug moves through the body over time.
It tracks four processes (ADME): **A**bsorption, **D**istribution, **M**etabolism,
**E**limination.

For IV bolus (intravenous injection), absorption is instant — the drug enters the
bloodstream all at once at `t=0`. So we only need to model distribution and elimination.

The core insight of PK modelling is that drug movement between body compartments
follows **first-order kinetics**: the rate of transfer is proportional to the current
concentration. This turns the biological process into a system of linear ODEs.

---

## 3. The one-compartment model (and why it fails here)

The simplest model treats the whole body as a single well-mixed compartment. Drug
enters at `t=0` and is eliminated at rate `k0`:

```
dP/dt = -k0 * P(t)
```

Solution: `P(t) = P(0) * exp(-k0 * t)` — pure exponential decay.

**Why this fails for DMT:** IV-injected DMT produces a plasma concentration profile
with a rapid initial rise (as the drug distributes to tissues) followed by a slower
elimination phase. A single exponential can only capture the decline, not the
initial distribution phase. Visually, the curve has a distinct "hump" shape
(rise then fall), which a single exponential cannot reproduce.

---

## 4. The two-compartment model

The two-compartment model acknowledges that the body is not a single homogeneous pool.
It divides the body into:

- **Central compartment** = plasma / blood (where we measure DMT directly)
- **Peripheral compartment** = brain / tissues (where DMT is redistributed)

### 4.1 Biological mechanism

After IV injection, DMT is initially in the plasma. Three things happen simultaneously:
- Some DMT is eliminated from plasma (metabolised/excreted) at rate `k0`.
- Some DMT moves from plasma to brain at rate `k1` (distribution).
- Some DMT moves back from brain to plasma at rate `k2` (redistribution).

This explains the hump shape: plasma concentration rises initially because... wait,
actually with IV bolus the plasma starts at its peak and then drops, but the *shape*
is biexponential (fast drop then slower tail) rather than a single exponential, because
the redistribution from brain back to plasma "refills" the plasma partially.

### 4.2 The ODEs

```
dP(t)/dt = -(k0 + k1) * P(t) + k2 * B(t)      [Eq. 13]
dB(t)/dt =          k1 * P(t) - k2 * B(t)      [Eq. 14]
```

Where:
- `P(t)` = drug concentration in plasma at time `t`
- `B(t)` = drug concentration in brain at time `t`
- `k0` = elimination rate from plasma (units: 1/min)
- `k1` = distribution rate from plasma to brain (units: 1/min)
- `k2` = redistribution rate from brain back to plasma (units: 1/min)

### 4.3 The analytical solution

Because this is a linear system of ODEs, it has a closed-form solution.
The solution involves two **exponential modes** with rates α and β:

```
α + β = k0 + k1 + k2
α * β = k2 * k0
```

These are the eigenvalues of the 2×2 rate matrix. They are found via the
quadratic formula: given `S = k0+k1+k2` and `Q = k2*k0`:

```
α = (S - sqrt(S² - 4Q)) / 2
β = (S + sqrt(S² - 4Q)) / 2
```

The plasma and brain concentration curves are then:

```
P(t) = P(0)/(β-α) * [(k2-α)*exp(-α*t) - (k2-β)*exp(-β*t)]    [Eq. 16]  ← we use this
B(t) = k1*P(0)/(β-α) * [exp(-α*t) - exp(-β*t)]                 [Eq. 17]  ← NOT used (no EEG data)
```

We only have plasma measurements, so **only Eq. 16 enters the likelihood**. Eq. 17 is shown for completeness but is not fitted.

**Physical interpretation:**
- The `exp(-α*t)` term corresponds to the *fast* phase (initial rapid distribution).
- The `exp(-β*t)` term corresponds to the *slow* phase (slower elimination).
- The plasma curve is a weighted difference of the two exponentials — this is what
  creates the characteristic biexponential shape.
- P(0) is the initial plasma concentration immediately after injection (t=0).
  It is proportional to the administered dose.

---

## 5. Bayesian inference

### 5.1 Why Bayesian?

With only 5 observations per subject, we have very little data to estimate 4 parameters
(k0, k1, k2, P0). Classical maximum likelihood estimation (MLE) would overfit —
finding parameter values that explain the 5 points perfectly but generalise poorly.

Bayesian inference solves this by incorporating **prior knowledge** about plausible
parameter values and returning a full **posterior distribution** (not just a point
estimate). The posterior encodes both the best-fit parameters *and* our uncertainty
about them.

### 5.2 Bayes' theorem

```
P(parameters | data) ∝ P(data | parameters) × P(parameters)
    posterior         ∝      likelihood       ×    prior
```

- **Prior `P(parameters)`**: what we believe before seeing the data.
  Rate constants are positive (use HalfCauchy / HalfNormal priors).
- **Likelihood `P(data | parameters)`**: how probable the observed plasma
  concentrations are given a particular set of PK parameters.
  We model observation noise as Gaussian: `P_obs ~ Normal(P(t), sigma)`.
- **Posterior `P(parameters | data)`**: updated belief after seeing data.

### 5.3 MCMC sampling

The posterior distribution has no closed form, so we use Markov Chain Monte Carlo
(MCMC) to draw samples from it. PyMC uses the **NUTS** (No-U-Turn Sampler) algorithm,
which efficiently explores high-dimensional parameter spaces.

Running 2 chains × 1000 draws gives us 2000 samples from the posterior. Each sample
is a complete set of model parameters. We can then:
- Average over samples to get the posterior mean curve.
- Use the 94% highest density interval (HDI) as an uncertainty band.

---

## 6. Multi-level modelling: partial pooling

### 6.1 The three alternatives

**Complete pooling:** one set of PK parameters for all 13 subjects.
- Problem: ignores individual differences. Underfits subjects who deviate from the mean.
- Equivalent to pretending all subjects are identical.

**No pooling (unpooled):** independent parameters for each subject, estimated
from each subject's 5 data points alone.
- Problem: each subject has only 5 observations — too few to reliably estimate
  4 parameters. Overfits to noise.
- Equivalent to pretending subjects share no common biology.

**Partial pooling:** each subject has their own parameters, but those parameters
are assumed to be drawn from a shared population distribution.
- This is the "Goldilocks" solution: subjects are allowed to differ, but they
  borrow statistical strength from each other.
- A subject with unusual data is "pulled" slightly toward the population mean —
  a phenomenon called **shrinkage** — which regularises the fit.

### 6.2 How partial pooling works mathematically

Instead of independent priors per subject, we add a **hyperprior** level:

```
# Population-level (hyperpriors)
k0_sigma ~ Exponential(1)        # controls spread of k0 across subjects
k1_sigma ~ Exponential(1)
k2_sigma ~ Exponential(1)
P0_sigma ~ Exponential(1)

# Subject-level (drawn from population distribution)
k0[i] ~ HalfCauchy(k0_sigma)    for each subject i
k1[i] ~ HalfCauchy(k1_sigma)
k2[i] ~ HalfCauchy(k2_sigma)
P0[i] ~ HalfNormal(P0_sigma)
```

The hyperparameters (k0_sigma, etc.) are themselves inferred from the data.
If subjects are very similar, the posterior of `k0_sigma` will be small,
pulling all subjects' `k0[i]` toward a common value. If subjects are very
different, `k0_sigma` will be large, allowing more inter-subject variation.

This is the hierarchical Bayesian model — also known as a **mixed-effects model**
in frequentist statistics.

### 6.3 Why does it outperform the alternatives?

- More data-efficient: 13 subjects × 5 points = 65 observations jointly inform
  all subject-level estimates, even though parameters are still per-subject.
- Better generalisation: shrinkage toward the population mean prevents overfitting
  to the idiosyncrasies of 5 sparse observations.
- The paper (Section 6.3) confirms this empirically using WAIC (model comparison),
  with partial pooling ranking best out-of-sample.

---

## 7. Prior distributions: choices and their meaning

| Parameter      | Prior           | Reason |
|----------------|-----------------|--------|
| `k0, k1, k2`  | `HalfCauchy(σ)` | Rate constants must be positive. HalfCauchy has a heavy tail, allowing occasional large values without strongly penalising them. |
| `P0`           | `HalfNormal(σ)` | Initial concentration must be positive. HalfNormal is lighter-tailed — appropriate since P(0) is roughly proportional to dose. |
| `k*_sigma`     | `Exponential(1)`| Scale parameters for the hyperprior. Exponential enforces positivity and penalises very large population variances. |
| `plasma_sigma` | `HalfCauchy(1)` | Observation noise must be positive. HalfCauchy is weakly informative. |

**Why HalfCauchy for rate constants?**
Rate constants in PK can vary by orders of magnitude across compounds and individuals.
The HalfCauchy distribution has very heavy tails (heavier than HalfNormal or Gamma),
which means it assigns non-negligible prior probability to both very small and very
large values. This makes it robust when we have little prior knowledge about the
scale of these parameters.

---

## 8. The likelihood: connecting model to data

We observe `P_obs[i]` at time `t[i]` for subject `s[i]`. The model predicts:

```
mu[i] = P(t[i], P0[s[i]], k0[s[i]], k1[s[i]], k2[s[i]])
```

The likelihood assumes Gaussian observation noise:

```
P_obs[i] ~ Normal(mu[i], plasma_sigma)
```

This encodes the assumption that measurements deviate from the true underlying
curve by normally distributed errors (instrument noise, biological variability
within the measurement window).

**Note on imputed values:** The dataset flags some rows as `is_imputed=True`.
These were filled in by linear interpolation in the original preprocessing step.
Treating them identically to real observations is reasonable given the model's
uncertainty is already captured via `plasma_sigma`.

---

## 9. Interpolation via posterior predictive distribution

Once we have posterior samples `{k0[i,s], k1[i,s], k2[i,s], P0[i,s]}` for
subject `i` and sample `s`, we can evaluate the analytical PK curve at any
time point `t` in a dense grid:

```python
for each sample s:
    P_curve[s, t] = plasma_concentration(t_grid, P0[i,s], k0[i,s], k1[i,s], k2[i,s])
```

The result is a distribution of curves. Summarising across samples:
- **Posterior mean curve**: `mean over s of P_curve[s, t]`
- **94% HDI band**: the central 94% of the distribution at each `t`

This gives us a smooth continuous curve with honest uncertainty quantification —
we know exactly how confident the model is at each time point.

The HDI is narrower near the observed data points (where the model is constrained
by data) and wider at extrapolated regions (where the prior dominates).

---

## 10. Why 94% HDI and not 95% confidence interval?

The 94% HDI (Highest Density Interval) is a Bayesian credible interval. It contains
the 94% most probable parameter values given the data. It is asymmetric when the
posterior is skewed — which is common for rate constants (bounded at zero).

A 95% frequentist confidence interval has a different (and arguably more confusing)
interpretation: "if we repeated this experiment many times, 95% of such intervals
would contain the true value." The HDI directly says: "given the data we observed,
there is a 94% probability that the true value lies in this interval."

The choice of 94% (rather than 95%) is a convention in the PyMC/ArviZ ecosystem,
following McElreath's *Statistical Rethinking*, to avoid false precision from the
round number 95.

---

## 11. Summary: why this approach works

| Challenge | Solution |
|-----------|----------|
| Only 5 observations per subject | Partial pooling borrows strength across subjects |
| Drug curves are non-linear (rise then fall) | Two-compartment ODE captures biexponential shape |
| Need uncertainty estimates | Full Bayesian posterior via MCMC |
| Parameters must be positive | HalfCauchy / HalfNormal priors enforce positivity |
| Need smooth curves at arbitrary time points | Analytical ODE solution evaluated on dense grid |
| Risk of overfitting with few data points | Hierarchical priors (hyperpriors) regularise estimates |
