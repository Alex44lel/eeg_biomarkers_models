# Polyphase / capacity sweep — findings (last updated 2026-04-29)

Self-contained context for designing the next round of runs. Cites specific MLflow
run names so you can pull them up directly. All multiseed runs use 5 seeds × 8 LOSO
folds at lr=5e-4, bs=64, dropout=0.3, wd=1e-4, smoothl1 huber_beta=10, EMA 0.999,
early-stop on val_r2 (mean-of-K phase R² when k>1), patience=40, max_epochs=300.

## TL;DR

1. **The multi-phase metric was inflating high-k results.** The apr28 single-phase
   sweep (keep only `k_idx==0`, one window per parent — same N as k=1) shows the
   k=5 default *collapses to ≈0 R²* (−0.007) and k=5 wide collapses identically
   (−0.006), vs the multi-phase numbers (+0.087 / +0.132). Phase averaging at
   evaluation time was acting as a free test-time ensemble that smoothed an
   essentially zero-signal model toward the mean. **The honest k=5 result is
   roughly zero**, not +0.13.
2. **At low k the inflation flips sign.** k=2 / k=3 / k=4 single-phase are
   *equal or slightly better* than multi-phase (k=2 wide single-phase
   **+0.377 ± 0.037** vs +0.351 ± 0.022 multi-phase). The K-row repetition is
   correlated noise that adds variance without adding signal at low k.
3. **New leaderboard:** `apr28_multiseed_k2_wide_singlephase` (+0.377 ± 0.037,
   932k params) is the new operational champion — by a hair (overlapping CIs)
   over `apr25_multiseed_k2_rfmatched` (+0.345 ± 0.021, 436k params). The
   apr25 default-channel run is still the best-stability / cheapest-good choice;
   the wide single-phase run is the highest-mean choice. Replicate on a fresh
   seed set before promoting.
4. **k=3 / k=4 fill in a smooth degradation curve, not a cliff.** apr28
   k=3 wide → +0.270; k=4 wide → +0.190; k=5 wide → +0.132 (multi-phase) /
   −0.006 (single-phase). The previously-mysterious "k=5 collapse" is just the
   bottom of a continuous slide as polyphase factor increases.
5. **The kernel-isolation 2×2 says architecture and rate are *both* axes,
   neither dominant.** Small kernels at k=1 (full rate, RF=125 ms) yield +0.191
   — far below the same-rate k=1 baseline (+0.299). Big kernels at k=5
   (RF=1555 ms, 2.5× the baseline) yield +0.110 — only marginally above the
   small-kernel k=5 (+0.087). The two contributions are roughly *additive*:
   neither knob alone closes the k=2 → k=5 gap.
6. **Anti-aliasing remains a non-issue (apr27 confirmed).** k=2 filt ties k=2
   unfilt; k=5/k=10 filt are if anything slightly worse. Source data is
   already software-low-passed at ~100 Hz (see `eeg_source_psd.png`).
7. **Polyphase as augmentation is mostly dead.** Low-k single-phase ≥ multi-
   phase suggests phase repetition at training time isn't useful augmentation;
   high-k multi-phase metric was an evaluation-side artifact. Going forward,
   compare on single-phase numbers — those are the honest LOSO scores.

## Full grid (capacity × k × phase-policy × filter-state)

Headline metric is `across_seed_mean_cv_r2 ± across_seed_std_cv_r2` over 5 seeds.
RF in raw-ms is held within ~1% (622–630 ms) across the entire grid by design,
except for the two kernel-isolation cells (RF=125 ms and 1555 ms — labeled).
See also `apr26_capacity_sweep_table.png` for the same data at a glance.

### Unfiltered multi-phase (raw stride-K split — the original build)

| name                              | params    | kernels    | channels      | RF (raw-ms) | cv_r2 ± std        |
|-----------------------------------|-----------|------------|---------------|------------:|--------------------|
| apr26_multiseed_k1_narrow         |   229,945 | 63, 15, 15 | 32, 64, 128   | 623 ms      | +0.2769 ± 0.038    |
| apr24_exp1                        |   788,593 | 63, 15, 15 | 64, 128, 256  | 623 ms      | +0.2987 ± 0.063    |
| apr26_multiseed_k1_wide           | 1,675,945 | 63, 15, 15 | 96, 192, 384  | 623 ms      | +0.3436 ± 0.042    |
| apr26_multiseed_k2_narrow         |   125,497 | 31, 8, 8   | 32, 64, 128   | 622 ms      | +0.2289 ± 0.044    |
| apr25_multiseed_k2_rfmatched      |   436,337 | 31, 8, 8   | 64, 128, 256  | 622 ms      | **+0.3448 ± 0.021** (champ — multi-phase) |
| apr26_multiseed_k2_wide           |   932,521 | 31, 8, 8   | 96, 192, 384  | 622 ms      | +0.3512 ± 0.022    |
| apr28_multiseed_k3_narrow         |    93,753 | 24, 4, 6   | 32, 64, 128   | 624 ms      | +0.1633 ± 0.067    |
| apr28_multiseed_k3_default        |   323,697 | 24, 4, 6   | 64, 128, 256  | 624 ms      | +0.2378 ± 0.045    |
| apr28_multiseed_k3_wide           |   689,833 | 24, 4, 6   | 96, 192, 384  | 624 ms      | +0.2704 ± 0.040    |
| apr28_multiseed_k4_default        |   260,209 | 13, 3, 5   | 64, 128, 256  | 628 ms      | +0.0914 ± 0.074    |
| apr28_multiseed_k4_wide           |   563,881 | 13, 3, 5   | 96, 192, 384  | 628 ms      | +0.1902 ± 0.029    |
| apr26_multiseed_k5_narrow         |    64,057 | 13, 3, 4   | 32, 64, 128   | 625 ms      | +0.0336 ± 0.030    |
| apr25_multiseed_k5_rfmatched      |   227,441 | 13, 3, 4   | 64, 128, 256  | 625 ms      | +0.0873 ± 0.047    |
| apr26_multiseed_k5_wide           |   490,153 | 13, 3, 4   | 96, 192, 384  | 625 ms      | +0.1323 ± 0.017    |
| apr25_multiseed_k10_rfmatched     |   166,001 | 15, 3, 2   | 64, 128, 256  | 630 ms      | −0.0376 ± 0.059    |
| apr26_multiseed_k10_wide          |   348,841 | 15, 3, 2   | 96, 192, 384  | 630 ms      | −0.0682 ± 0.047    |

### Single-phase (apr28, `--single_phase`: keep only `k_idx==0`)

Same datasets, kernels and capacity as the matching multi-phase rows above; the
loader discards all rows except phase 0, so each parent window contributes once
(same trial count as the k=1 baseline; signal is decimated by k inside that one
window). Comparison row-by-row with multi-phase isolates the phase-repetition
contribution from the pure downsampling effect.

| name                                     | params  | kernels  | channels      | RF (raw-ms) | cv_r2 ± std        | Δ vs multi-phase |
|------------------------------------------|--------:|----------|---------------|------------:|--------------------|-----------------:|
| apr28_multiseed_k2_default_singlephase   | 436,337 | 31, 8, 8 | 64, 128, 256  | 622 ms      | +0.3266 ± 0.031    | −0.018           |
| apr28_multiseed_k2_wide_singlephase      | 932,521 | 31, 8, 8 | 96, 192, 384  | 622 ms      | **+0.3769 ± 0.037** (new high-mean) | +0.026 |
| apr28_multiseed_k3_default_singlephase   | 323,697 | 24, 4, 6 | 64, 128, 256  | 624 ms      | +0.2595 ± 0.015    | +0.022           |
| apr28_multiseed_k3_wide_singlephase      | 689,833 | 24, 4, 6 | 96, 192, 384  | 624 ms      | +0.2772 ± 0.031    | +0.007           |
| apr28_multiseed_k4_default_singlephase   | 260,209 | 13, 3, 5 | 64, 128, 256  | 628 ms      | +0.1252 ± 0.036    | +0.034           |
| apr28_multiseed_k4_wide_singlephase      | 563,881 | 13, 3, 5 | 96, 192, 384  | 628 ms      | +0.1952 ± 0.029    | +0.005           |
| apr28_multiseed_k5_default_singlephase   | 227,441 | 13, 3, 4 | 64, 128, 256  | 625 ms      | **−0.0067 ± 0.031** | −0.094 (collapse) |
| apr28_multiseed_k5_wide_singlephase      | 490,153 | 13, 3, 4 | 96, 192, 384  | 625 ms      | **−0.0063 ± 0.072** | −0.139 (collapse) |

### Anti-aliased polyphase (apr27, FIR low-pass before each phase)

| name                              | params    | kernels    | channels      | RF (raw-ms) | cv_r2 ± std        | vs unfilt |
|-----------------------------------|-----------|------------|---------------|------------:|--------------------|-----------|
| apr27_multiseed_k2_filt           |   436,337 | 31, 8, 8   | 64, 128, 256  | 622 ms      | +0.3397 ± 0.018    | −0.005 (control passes) |
| apr27_multiseed_k5_filt_wide      |   490,153 | 13, 3, 4   | 96, 192, 384  | 625 ms      | +0.0694 ± 0.084    | **−0.063** (worse, std 5×) |
| apr27_multiseed_k10_filt_wide     |   348,841 | 15, 3, 2   | 96, 192, 384  | 630 ms      | −0.0859 ± 0.055    | −0.018 |

### Kernel-isolation 2×2 (apr28 — disentangle kernels from polyphase rate)

|              | small kernels (13, 3, 4)             | medium kernels (31, 8, 8)            |
|--------------|--------------------------------------|--------------------------------------|
| **k=1**      | apr28_multiseed_k1_smallkern  +0.1907 ± 0.024 (RF=125 ms) | apr24_exp1 (kernels 63/15/15) — n/a |
| **k=5**      | apr25_multiseed_k5_rfmatched  +0.0873 ± 0.047 (RF=625 ms) | apr28_multiseed_k5_bigkern  +0.1103 ± 0.095 (RF=1555 ms) |

Reading: small kernels alone cost ~0.11 R² at k=1 (+0.299 → +0.191); big kernels
alone gain ~0.02 R² at k=5 (+0.087 → +0.110, well within noise). Architecture and
polyphase rate each contribute, neither dominates, and the contributions look
roughly additive (close-to-baseline only when *both* knobs are healthy).

## Why the apr27 anti-aliasing experiment closed the dataset axis

**Hypothesis tested:** the apr25 k=5/k=10 collapse was caused by aliasing in the
unfiltered build (`build_downsampled_dataset.py:62` does `eeg[..., i::k]` with no
low-pass filter). If true, applying a 31-tap FIR low-pass at the new Nyquist
*before* each phase would rescue performance.

**Result:** rejected. Anti-aliasing made k=5 wide *slightly worse*
(+0.132 → +0.069, Δ ≈ 1.6σ — borderline; variance jumped 5×) and left k=10 wide
similar (−0.068 → −0.086). The k=2 control was unchanged (+0.345 → +0.340),
confirming the filter itself is benign at k=2.

**Spectral audit then showed the rejected hypothesis was even more wrong than
"didn't work":** a Welch PSD over 200 random trials × 32 channels showed the
source data has been software-low-passed at ~100 Hz before being saved. Power
drops by **~9 orders of magnitude** between the gamma_high band (50–100 Hz) and
100–150 Hz:

| band | range (Hz) | power (μV²) | % of <100 Hz total |
|------|-----------:|-------------|-------------------:|
| alpha (dominant) | 8–13 | 15.65 | 52.8% |
| beta             | 13–30 | 5.49  | 18.5% |
| gamma_low        | 30–50 | 1.17  | 3.9% |
| gamma_high       | 50–100 | 0.012 | 0.04% |
| **100–150**      |        | **7.8e-10** | ~0% |
| 150–500          |        | <1e-11 | ~0% |

There is **essentially nothing above 100 Hz in the source** to either alias from
or to recover. So the apr27 experiment was testing a hypothesis that couldn't be
true: there was no high-frequency content for the unfiltered build to be
exploiting in the first place. The plot is at `eeg_source_psd.png`.

Note also: surface EEG can barely register cortical activity above ~80 Hz to
begin with — the skull is a low-pass filter, and anything you do see up there
on scalp is dominated by EMG / oculomotor / cardiac artifacts. The earlier
"DMT signal lives in high-gamma" framing was wrong on multiple counts.

**Best explanation for the small apr27 Δ at k=5:** (a) the FIR Hamming filter's
soft transition band attenuates real EEG content in the 80–100 Hz range
(gamma_high, 0.04% of total power) that the unfiltered build preserves cleanly;
(b) the polyphase phases become more similar after low-pass filtering, weakening
augmentation diversity. Both are small effects and (b) likely dominates the
variance increase. None of this is about high-frequency signal.

## Why k=2 outperforms k=5 then — architecture, rate, and a metric artifact

If both k=2 and k=5 preserve the same useful band (0–100 Hz, since that's all
the source has), the gap between them must reflect (a) what the model can *do*
with the input, (b) the granularity of the input itself, and (c) — newly — the
metric inflation from phase averaging at high k. The apr28 single-phase runs
take (c) off the table: with k=5 single-phase ≈ 0 R², the *real* gap is
≈ +0.34 R², 30% wider than the multi-phase number suggested. The
kernel-isolation 2×2 then attributes that gap to (a) and (b) jointly. Three
architectural factors collapse k=5:

**1. Kernel-shrinking from RF-match constraint.** Holding RF in raw-ms at ~623
forces small kernels at high k:

| k  | kernels      | smallest deeper kernel |
|----|--------------|------------------------|
| 1  | 63, 15, 15   | 15                      |
| 2  | 31, 8, 8     | 8                       |
| 5  | **13, 3, 4** | **3** (almost 1×1)      |
| 10 | 15, 3, 2     | 2                       |

A size-3 conv after a stride-8 first layer can correlate just 3 adjacent
features. The model's expressivity in the deeper layers is severely cramped.

**2. Few output positions before pooling.** After cumulative stride 128:

| k  | input L | output positions | feature averaging strength |
|----|---------|------------------|-----------------------------|
| 1  | 3000    | ~23              | strong                      |
| 2  | 1500    | ~11              | adequate                    |
| 5  | 600     | **~4**           | **weak**                    |
| 10 | 300     | **~2**           | **very weak**               |

`AdaptiveAvgPool1d(1)` then collapses to a single feature. With ~4 positions at
k=5, the trial-level feature is a noisier estimate.

**3. Time-domain resolution.** Even when the *band* is preserved, the time-
domain CNN features are quantized at the sample rate. At k=5 (5 ms/sample), any
neural transient briefer than 5 ms is unrepresentable. Phase-locking, edge
timing, and short bursts get blurred. Fourier content can survive, but
time-domain features the CNN learns don't.

This is why anti-aliasing didn't help and capacity didn't help: the bottleneck
isn't input information or parameter count — it's the *architecture* the
RF-match constraint forced at high k, plus the time-domain rate itself
(see kernel-isolation 2×2). And about half of the previously-reported "k=5
works a little" was just phase-averaging at evaluation time, not real signal.

## Per-subject patterns (across all conditions)

The hard subjects (S01, S05, S13) and the easy ones (S07, S12) hold up to the
same intuition across the whole sweep:

- **S07 / S12** are robust across every condition (R² stays ≥ +0.4 in almost
  every cell). Their plasma signal must encode in slow-envelope, low-frequency
  features that survive any sample rate.
- **S01 / S05 / S13** crash hardest at k=5/k=10. They evidently rely on
  fine-temporal / high-frequency content that's lost or corrupted at lower rates.
- **S02 is a striking apr27 outlier.** At k=5, S02 went from −0.13 (unfilt) to
  −0.63 (filt) — Δ −0.50. The aliased high-freq fragments were apparently
  load-bearing for S02 specifically; clean filtering removed them. Different
  subjects encode plasma in different bands; that heterogeneity is a real
  feature of this dataset, not noise.
- **S13 at k=10 anti-aliased** went from −0.37 to +0.01 (Δ +0.38) — the only
  case where filtering helped a hard subject substantially. Suggests S13's
  signal is in a low-freq band that gets corrupted by aliased gamma noise at
  k=10 unfiltered.

## Operational champion (use as new baseline)

Two viable champions now; pick by the criterion that matters for the next
experiment.

**Highest mean: `apr28_multiseed_k2_wide_singlephase`** (new). cv_r2 =
+0.3769 ± 0.037, pooled_r2 = +0.787, kernels 31/8/8, channels 96/192/384,
932k params, RF=622 raw-ms, dataset `pk_k2` with `--single_phase`.
Caveat: only one seed-set tested; std is ~2× the apr25 default-channel champ.
Replicate before adopting as a hard baseline.

**Best stability / cheapest: `apr25_multiseed_k2_rfmatched`** (still). cv_r2 =
+0.345 ± 0.021, pooled_r2 = +0.773, kernels 31/8/8, channels 64/128/256,
436k params (~2× cheaper), RF=622 raw-ms. Multi-phase, but the apr28 single-
phase counterpart (`apr28_multiseed_k2_default_singlephase`, +0.327 ± 0.031)
is a statistical tie — multi-phase or single-phase doesn't matter here.

`apr27_multiseed_k2_filt` (+0.340 ± 0.018) is also a statistical tie at k=2
default; anti-aliasing is benign at k=2 but provides no measurable benefit.

## Code-state caveats for cross-run comparison

The polyphase R² semantics changed on 2026-04-27 (mean-of-K phase R²). Runs before
that date logged `cv_mean_val_r2` as the *pooled-over-phases* R² for k>1 datasets;
runs after log it as the mean across phases of within-phase R². For k=1 the two
are identical.

- Runs computed under **OLD** semantics (pooled over phases): `apr24_exp1`,
  `apr25_multiseed_k2_rfmatched`, `apr25_multiseed_k5_rfmatched`,
  `apr25_multiseed_k10_rfmatched`. The k>1 entries are slightly *low-bound*
  estimates of the mean-of-K equivalent.
- Runs computed under **NEW** semantics: all `apr26_multiseed_*` and all
  `apr27_multiseed_*`.
- In practice the difference is +0.01 to +0.03 R² (mean-of-K is usually slightly
  higher), so it doesn't change the conclusions above.

The new code also adds:
- per-phase metrics `best_val_r2_kidx{N}` per fold,
- per-phase scatter and DMT-evolution plots,
- `k_idx` column in `predictions_all_seeds.npz`,
- `pooled_val_r2_pooled_all` legacy sibling (the old all-points pooled R²).

## What's left to try

The polyphase / dataset axis is now fully mapped. The remaining axes worth
exploring are model-side and training-side, not data-side.

1. **Replicate `apr28_multiseed_k2_wide_singlephase` on a fresh seed set**
   (e.g. 10/20/30/40/50). It nominally beats the apr25 default-channel champ
   by +0.03 R² but the seeds are correlated with every other apr-* run; one
   fresh draw decides whether single-phase + wide is a real lift or just a
   favorable sample. Cheap (≈3–4 h on a30) and resolves the leaderboard.
2. **Replicate `apr26_multiseed_k1_wide` (+0.344 ± 0.042) on the same
   alt-seed set.** k=1 wide is statistically tied with k=2 default at the
   current seeds; an independent draw is the cheapest way to tell whether
   "polyphase helps a little" or "it's all capacity, k=1 with capacity is
   fine". This was already on the list; apr28 hasn't changed its priority.
3. **Subject-conditioned models.** The huge per-subject heterogeneity
   (S07/S12 +0.6; S01/S05 sometimes negative) suggests subjects encode plasma
   in different bands. Options: subject-ID embedding at the regressor head;
   per-subject calibration on a small number of labeled trials of the held-out
   subject (a "few-shot" relaxation of strict LOSO — has to be motivated/
   defended in writeup); meta-learning. The biggest open lever for the dataset.
4. **Spectral-input frontend.** Replace the first-conv stack with a fixed
   FFT/wavelet bank, then learn the rest of the CNN on (channels × bands ×
   time). Most of the plasma-correlated content lives in 4–50 Hz; an
   architecture that exposes those bands explicitly might be a better fit
   than a strided CNN that has to learn the band decomposition from scratch.
5. **Drop the `--single_phase` toggle into all future runs.** It's a free
   honesty check: if a multi-phase result is much better than its single-phase
   sibling, the gap is metric-side, not model-side. Cheap to track.

### Things that are *not* worth doing

These follow directly from the apr27/apr28 results.

- **More capacity at k=5 / k=10.** Capacity alone doesn't move the needle and
  the single-phase honest score is ≈ 0; there's nothing to amplify.
- **More polyphase variants** (different filter shapes, different K).
  Anti-aliasing is a no-op (source already low-passed). The k-axis has been
  swept at K=1/2/3/4/5/10 in capacity narrow/default/wide; the curve is smooth
  and monotonically declining — no hidden sweet spot.
- **Reading the multi-phase metric as the headline at high k.** The apr28
  single-phase results show it's an evaluation-side ensemble, not a model
  property. Past mlflow rows still display the multi-phase number; quote
  single-phase when comparing models from now on.

## Pointers

- Build pipeline (unfiltered, the original):
  `src/models/reg_simpleCNN/build_downsampled_dataset.py`
  (raw stride at line 62: `out[i::k] = eeg[..., i:i+k*l_ds:k]`).
- Build pipeline (anti-aliased, apr27):
  `src/models/reg_simpleCNN/build_downsampled_dataset_filt.py`
  (FIR Hamming low-pass via filtfilt, then phase-extract).
- Single-phase loader toggle (apr28): `--single_phase` in
  `src/models/reg_simpleCNN/multiseed_cv.py` / `train_cv.py`; the dataset
  layer drops all rows with `k_idx != 0`, so K-fold metric paths short-circuit.
- Model: `src/models/reg_simpleCNN/model.py`
  (`SimpleCNN` accepts `channels=...` since 2026-04-26; default 64/128/256).
- Training pipeline (single seed): `src/models/reg_simpleCNN/train_cv.py`
  (`run_fold`, `polyphase_metrics`, `plot_predicted_vs_actual`).
- Multiseed driver: `src/models/reg_simpleCNN/multiseed_cv.py`.
- Dataset registry: `src/models/reg_simpleCNN/dataset.py`
  (`DATASET_PATHS` includes `pk_k{2,3,4,5,10}` and `pk_k{2,5,10}_filt`).
- Cluster mlruns mirror: `~/Desktop/tfm_code/mlruns_cluster/` (synced via
  `scripts/pull_mlruns.sh`; rsync without `--delete`, paths re-written).
- SLURM / shell wrappers (one per experiment family):
  - `src/models/reg_simpleCNN/shell_and_logs/run_apr25_polyphase_rf_matched.sh` — original unfiltered sweep
  - `src/models/reg_simpleCNN/shell_and_logs/run_apr26_capacity_sweep.sh` — k=1/2/5/10 capacity sweep
  - `src/models/reg_simpleCNN/shell_and_logs/run_apr27_polyphase_filt.sh` — anti-aliased rebuild
  - `src/models/reg_simpleCNN/shell_and_logs/run_apr28_kernel_isolation.sh` — 2×2 architecture-vs-rate
  - `src/models/reg_simpleCNN/shell_and_logs/run_apr28_k34_capacity_sweep.sh` — fills k=3, k=4 columns
  - `src/models/reg_simpleCNN/shell_and_logs/run_apr28_singlephase_sweep.sh` — single-phase honest metric
- Source-data spectrum plot: `misc/study_notes/eeg_source_psd.png` (proves
  the source is brick-wall low-passed at ~100 Hz).
- Capacity sweep table image: `misc/study_notes/apr26_capacity_sweep_table.png`
  (regenerated 2026-04-29 with apr27/apr28 rows + champion highlights).
