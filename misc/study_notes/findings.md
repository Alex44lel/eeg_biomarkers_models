# Polyphase / capacity sweep — findings (last updated 2026-04-27)

Self-contained context for designing the next round of runs. Cites specific MLflow
run names so you can pull them up directly. All multiseed runs use 5 seeds × 8 LOSO
folds at lr=5e-4, bs=64, dropout=0.3, wd=1e-4, smoothl1 huber_beta=10, EMA 0.999,
early-stop on val_r2 (mean-of-K phase R² when k>1), patience=40, max_epochs=300.

## TL;DR

1. **Champion: `apr25_multiseed_k2_rfmatched`** — k=2 polyphase, 436k params,
   `cv_mean_val_r2 = +0.345 ± 0.021`. Best mean × best stability × cheapest model.
   `apr27_multiseed_k2_filt` (anti-aliased k=2) ties it at +0.340 ± 0.018.
2. **k=1 baseline (`apr24_exp1`) was undersized.** k=1 wide reaches +0.344, nearly
   matching k=2 default — half the previously-reported "polyphase win" was actually
   capacity the baseline didn't have.
3. **k=5 / k=10 don't recover with capacity.** k=5 wide → +0.132, still far below
   k=2. At equal capacity (~230k) k=1 beats k=5 by 0.19 R². Capacity is not the
   bottleneck.
4. **k=5 / k=10 don't recover with anti-aliasing either.** The apr27 rebuild
   made k=5 wide drop from +0.132 to +0.069 (Δ ≈ 1.6σ — small effect, but with
   variance jumping 5×) and k=10 wide stay at −0.086. The original interpretation
   ("aliased high-gamma being used as signal") was **wrong**: a power-spectrum
   audit (`misc/study_notes/eeg_source_psd.png`) shows the source data was
   software-low-passed at ~100 Hz during preprocessing, with power dropping
   **9 orders of magnitude** between 50–100 Hz and 100–150 Hz. There is
   essentially nothing above 100 Hz to alias.
5. **The k=5 collapse is *architectural*, not spectral.** With the same input
   band (0–100 Hz) preserved at all polyphase factors, the gap between k=2
   (+0.345) and k=5 (+0.087) reflects what the model can *do* with the input,
   not what's *in* the input. The RF-matching constraint forces tiny kernels
   `(13, 3, 4)` at k=5 vs `(31, 8, 8)` at k=2; only ~4 output positions before
   pooling vs ~11 at k=2; 5 ms time resolution vs 2 ms.
6. **The polyphase story closes here for the dataset axis.** k=2 works, k≥5
   doesn't, and dataset-level fixes (anti-aliasing) can't rescue them. The open
   question is whether *architectural* fixes (bigger kernels at k=5, accepting
   a larger raw-ms RF) can — see "What's left to try."

## Full grid (capacity × k × filter-state)

Headline metric is `across_seed_mean_cv_r2 ± across_seed_std_cv_r2` over 5 seeds.
RF in raw-ms is held within 1.3% (622–630 ms) across the entire grid by design.

### Unfiltered polyphase (raw stride-K split, the original build)

| name                              | params    | kernels    | channels      | RF (raw-ms) | cv_r2 ± std        |
|-----------------------------------|-----------|------------|---------------|------------:|--------------------|
| apr26_multiseed_k1_narrow         |   229,945 | 63, 15, 15 | 32, 64, 128   | 623 ms      | +0.2769 ± 0.038    |
| apr24_exp1                        |   788,593 | 63, 15, 15 | 64, 128, 256  | 623 ms      | +0.2987 ± 0.063    |
| apr26_multiseed_k1_wide           | 1,675,945 | 63, 15, 15 | 96, 192, 384  | 623 ms      | +0.3436 ± 0.042    |
| apr26_multiseed_k2_narrow         |   125,497 | 31, 8, 8   | 32, 64, 128   | 622 ms      | +0.2289 ± 0.044    |
| apr25_multiseed_k2_rfmatched      |   436,337 | 31, 8, 8   | 64, 128, 256  | 622 ms      | **+0.3448 ± 0.021** (champion) |
| apr26_multiseed_k2_wide           |   932,521 | 31, 8, 8   | 96, 192, 384  | 622 ms      | +0.3512 ± 0.022    |
| apr26_multiseed_k5_narrow         |    64,057 | 13, 3, 4   | 32, 64, 128   | 625 ms      | +0.0336 ± 0.030    |
| apr25_multiseed_k5_rfmatched      |   227,441 | 13, 3, 4   | 64, 128, 256  | 625 ms      | +0.0873 ± 0.047    |
| apr26_multiseed_k5_wide           |   490,153 | 13, 3, 4   | 96, 192, 384  | 625 ms      | +0.1323 ± 0.017    |
| apr26_multiseed_k10_narrow        |    49,721 | 15, 3, 2   | 32, 64, 128   | 630 ms      | (not synced/run)   |
| apr25_multiseed_k10_rfmatched     |   166,001 | 15, 3, 2   | 64, 128, 256  | 630 ms      | −0.0376 ± 0.059    |
| apr26_multiseed_k10_wide          |   348,841 | 15, 3, 2   | 96, 192, 384  | 630 ms      | −0.0682 ± 0.047    |

### Anti-aliased polyphase (apr27, FIR low-pass before each phase)

| name                              | params    | kernels    | channels      | RF (raw-ms) | cv_r2 ± std        | vs unfilt |
|-----------------------------------|-----------|------------|---------------|------------:|--------------------|-----------|
| apr27_multiseed_k2_filt           |   436,337 | 31, 8, 8   | 64, 128, 256  | 622 ms      | +0.3397 ± 0.018    | −0.005 (control passes) |
| apr27_multiseed_k5_filt_wide      |   490,153 | 13, 3, 4   | 96, 192, 384  | 625 ms      | +0.0694 ± 0.084    | **−0.063** (worse, std 5×) |
| apr27_multiseed_k10_filt_wide     |   348,841 | 15, 3, 2   | 96, 192, 384  | 630 ms      | −0.0859 ± 0.055    | −0.018 |

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

## Why k=2 outperforms k=5 then — the architectural argument

If both k=2 and k=5 preserve the same useful band (0–100 Hz, since that's all
the source has), the +0.26 R² gap between them must reflect what the model can
*do* with the input, not what's *in* the input. Three architectural factors
collapse k=5:

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
RF-match constraint forced at high k.

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

- Run name: `apr25_multiseed_k2_rfmatched` (run id `8856d0eb5b`)
- Dataset: `pk_k2`
- Kernels: 31, 8, 8 (RF=311 sub-samples = 622 raw-ms)
- Strides: 8, 4, 4
- Channels: 64, 128, 256 (default)
- Params: 436,337
- Result: cv_r2 = +0.345 ± 0.021, pooled_r2 = +0.773
- Why it wins: best stability (std 2× tighter than any k=1 cell), best per-subject
  consistency (S10/S13 specifically improved over k=1), cheap to train and serve.

`apr27_multiseed_k2_filt` is a statistical tie (+0.340 ± 0.018) and uses the
anti-aliased build. Either is fine; the filtered version is theoretically
cleaner DSP but provides no measurable performance benefit at k=2.

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

### apr28 kernel-isolation sweep (proposed, 2 runs)

The architectural argument above implies **two clean experiments** that
isolate kernel size from polyphase rate. Together with the existing k=1 default
and k=5 default runs, they form a 2×2 design:

|              | small kernels (13, 3, 4)             | medium kernels (31, 8, 8)            |
|--------------|--------------------------------------|--------------------------------------|
| **k=1**      | **NEW: apr28_multiseed_k1_smallkern**| apr24_exp1 (kernels 63/15/15) — n/a  |
| **k=5**      | apr25_multiseed_k5_rfmatched (+0.087)| **NEW: apr28_multiseed_k5_bigkern**  |

- **Exp A — `apr28_multiseed_k5_bigkern`**: k=5 dataset (`pk_k5`) with k=2's
  kernels `(31, 8, 8)`. Gives up the RF-match constraint (RF = 311 sub-samples
  × 5 ms = **1555 raw-ms**, 2.5× the baseline). Tests whether the k=5 collapse
  reverses when the model has expressive kernels and more output positions
  (~5 vs ~4 at apr25). If R² climbs toward k=2 territory (+0.30+), the
  collapse was *purely* architectural — RF-matching via kernel-shrinking is
  what hurt. If still ~+0.10, polyphase decimation harms the model in some
  way beyond architecture (likely time-domain feature granularity).

- **Exp B — `apr28_multiseed_k1_smallkern`**: k=1 dataset (`pk`) with k=5's
  small kernels `(13, 3, 4)`. RF = 125 raw-ms (much shorter than baseline 623,
  same as apr25 k=5). Tests whether full-resolution input rescues a small-
  kernel model. If R² climbs toward k=1 territory (+0.30+), polyphase rate
  *also* matters and the architectural story is incomplete. If ~+0.10
  (matching apr25 k=5), kernel architecture alone explains the gap.

These two runs disentangle the two axes:
- Both improve  → kernels and rate both contribute (additive).
- Only Exp A improves → kernels alone are the bottleneck.
- Only Exp B improves → polyphase rate alone is the bottleneck.
- Neither → something else is going on (back to drawing board).

Hyperparams: same as apr25 / apr26 default-channel runs (lr=5e-4, bs=64,
default channels 64/128/256, smoothl1 huber_beta=10, EMA 0.999, patience=40,
5 seeds × 8 LOSO folds). Runtime ~3–4h on a30.

### Other directions if the kernel-isolation sweep doesn't fully resolve the gap

1. **k=1 wide replication on a fresh seed set.** apr26_multiseed_k1_wide
   reaches +0.344 ± 0.042 on (42, 123, 7, 2024, 0). Worth verifying on
   (10, 20, 30, 40, 50) to confirm it's a stable second champion alongside
   k=2 default.
2. **Subject-conditioned models.** The huge per-subject heterogeneity
   (S07/S12 +0.6; S01/S05 sometimes negative) suggests subjects encode plasma
   in different bands. A model that conditions on subject ID at training time,
   or per-subject calibration on a few labeled trials of the held-out subject,
   could exploit this. Nontrivial under strict LOSO but worth considering.
3. **Spectral-input frontend.** Replace the first-conv stack with a fixed
   FFT/wavelet bank, then learn a CNN on the (channels × bands × time)
   representation. Might extract more from the 4–50 Hz range where the actual
   plasma-correlated content lives.

## Things NOT to do

- Don't sweep more capacity at k=5 / k=10. Capacity is not the bottleneck.
- Don't rebuild polyphase with different filter shapes (sharper FIR, IIR, etc.).
  The data is already low-passed at ~100 Hz upstream, so dataset-level filtering
  has nothing to operate on.
- Don't claim DMT signal lives above 100 Hz. The source data has been low-
  passed at ~100 Hz before reaching us, plasma-correlated content is in beta
  (UP with plasma) and delta/theta/low-gamma (DOWN), all in the 4–50 Hz range.
- Don't compare new k=2 / k=5 / k=10 numbers against the OLD-semantics apr25
  numbers without noting the +0.01–0.03 offset.
- Don't promote `apr26_multiseed_k2_wide` (+0.351 ± 0.022) over the default
  k=2 — they're a statistical tie and `wide` is 2× the params.

## Pointers

- Build pipeline (unfiltered, the original):
  `src/models/reg_simpleCNN/build_downsampled_dataset.py`
  (raw stride at line 62: `out[i::k] = eeg[..., i:i+k*l_ds:k]`).
- Build pipeline (anti-aliased, apr27):
  `src/models/reg_simpleCNN/build_downsampled_dataset_filt.py`
  (FIR Hamming low-pass via filtfilt, then phase-extract).
- Model: `src/models/reg_simpleCNN/model.py`
  (`SimpleCNN` accepts `channels=...` since 2026-04-26; default 64/128/256).
- Training pipeline (single seed): `src/models/reg_simpleCNN/train_cv.py`
  (`run_fold`, `polyphase_metrics`, `plot_predicted_vs_actual`).
- Multiseed driver: `src/models/reg_simpleCNN/multiseed_cv.py`.
- Dataset registry: `src/models/reg_simpleCNN/dataset.py`
  (`DATASET_PATHS` includes `pk_k{2,5,10}_filt` since apr27).
- Cluster mlruns mirror: `~/Desktop/tfm_code/mlruns_cluster/` (synced via
  `scripts/pull_mlruns.sh`; rsync without `--delete`, paths re-written).
- SLURM wrappers (one per experiment family):
  - `scripts/cluster_apr25_polyphase_rf.sh` — original unfiltered sweep
  - `scripts/cluster_apr26_capacity_sweep_{narrow,wide}.sh` — capacity sweep
  - `scripts/cluster_apr27_polyphase_filt.sh` — anti-aliased rebuild
  - `scripts/cluster_apr28_kernel_isolation.sh` — kernel-vs-rate isolation
- Source-data spectrum plot: `misc/study_notes/eeg_source_psd.png` (proves
  the source is brick-wall low-passed at ~100 Hz).
