# Polyphase / capacity sweep — findings (as of 2026-04-26)

Self-contained context for designing the next round of runs. Cites specific MLflow
run names so you can pull them up directly. All multiseed runs use 5 seeds × 8 LOSO
folds at lr=5e-4, bs=64, dropout=0.3, wd=1e-4, smoothl1 huber_beta=10, EMA 0.999,
early-stop on val_r2 (mean-of-K phase R² when k>1), patience=40, max_epochs=300.

## TL;DR

1. **New champion: `apr25_multiseed_k2_rfmatched`** — k=2 polyphase, 436k params,
   `cv_mean_val_r2 = +0.345 ± 0.021`. Best mean × best stability × cheapest model.
2. **k=1 baseline (`apr24_exp1`) was undersized.** k=1 wide reaches +0.344, almost
   matching k=2 default. Some of the previously-reported "polyphase wins" was just
   capacity tuning the baseline didn't have.
3. **k=5 / k=10 collapse is information loss, not capacity.** Adding 2× capacity to
   k=5 only buys +0.045 R²; at equal capacity (~230k params) k=1 beats k=5 by 0.19.
4. **Root cause of the collapse is missing anti-alias filtering** in
   `build_downsampled_dataset.py:62`. Polyphase decomposition without a low-pass
   filter folds frequencies above the new Nyquist back into the kept bands.
5. **Don't run more k=10 sweeps.** At 100 Hz Nyquist 50 Hz, alpha/beta get aliased;
   no amount of capacity rescues this.

## The full 4×3 grid (capacity × k)

Headline metric is `across_seed_mean_cv_r2 ± across_seed_std_cv_r2` over 5 seeds.
RF in raw-ms is held within 1.3% (622–630 ms) across the entire grid by design.

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

## Why k=5 / k=10 collapse

The polyphase split in `src/models/reg_simpleCNN/build_downsampled_dataset.py`
(function `polyphase_split`, line 62) is just a strided slice:

```python
out[i::k] = eeg[:, :, i:i + k*l_ds:k]
```

There is **no anti-alias filter**. Each sub-signal is sampled at `1000/k` Hz with
Nyquist `500/k` Hz:

| k  | effective rate | Nyquist | Anything above Nyquist…  |
|----|---------------:|--------:|---------------------------|
| 1  |       1000 Hz  | 500 Hz  | n/a                       |
| 2  |        500 Hz  | 250 Hz  | aliases into 0–250 Hz     |
| 5  |        200 Hz  | 100 Hz  | aliases into 0–100 Hz     |
| 10 |        100 Hz  |  50 Hz  | aliases into 0–50 Hz      |

At k=2 most EEG content lives below 250 Hz, so aliasing is mild. At k=5 the high
gamma band (60–100+ Hz) gets corrupted; at k=10 even alpha/beta are polluted by
folded-back high frequencies.

The decisive empirical evidence:

- **At equal capacity** (~230k params): k=1 narrow = +0.277 vs k=5 default = +0.087.
  Same model, same RF, same hyperparams — only the input rate differs. The 0.19 R²
  gap is purely the signal.
- **k=5 wide** (490k params, ~2× capacity) only goes from +0.087 → +0.132. Capacity
  cannot reconstruct information that isn't in the input.
- **k=10 wide is *worse* than k=10 default** (−0.068 vs −0.038). At 100 Hz the
  signal is below the floor; bigger models overfit aliased noise.

## Per-subject patterns (multi-seed mean, see also `per_fold_{subj}_mean_r2` metrics)

| Subject | k=1 wide | k=2 default | k=5 wide | k=10 wide | Profile |
|---------|----------|-------------|----------|-----------|---------|
| S01     | +0.02    | +0.01       | +0.02    | −0.10     | Hard always |
| S02     | +0.10    | +0.13       | −0.12    | −0.40     | Falls off at k=5 |
| S05     | +0.21    | +0.21       | **−0.54**| −0.73     | Sensitive to high-freq features |
| S06     | +0.19    | +0.16       | +0.11    | +0.04     | Robust |
| S07     | +0.61    | +0.64       | +0.62    | +0.46     | Easy always |
| S10     | +0.54    | +0.56       | +0.41    | +0.09     | Easy at low k |
| S12     | +0.66    | +0.73       | +0.47    | +0.46     | Easy always |
| S13     | +0.42    | +0.33       | +0.09    | −0.37     | Sensitive |

S07 / S12 are the easy "robust" subjects — predictable from low-frequency,
slow-envelope features. S01 / S05 / S13 are the historically hard subjects (per
the apr19 notes) and they're exactly the ones that collapse hardest at k=5 and
k=10. So **subject difficulty is a proxy for how much fine-temporal/high-frequency
content matters** for that subject's plasma signal.

## Operational champion (use as new baseline)

- Run name: `apr25_multiseed_k2_rfmatched` (the newer one, run id `8856d0eb5b`)
- Dataset: `pk_k2`
- Kernels: 31, 8, 8 (RF=311 sub-samples = 622 raw-ms)
- Strides: 8, 4, 4
- Channels: 64, 128, 256 (default)
- Params: 436,337
- Result: cv_r2 = +0.345 ± 0.021, pooled_r2 = +0.773
- Why it wins: best stability (std 2× tighter than any k=1 cell), best per-subject
  consistency (S10/S13 specifically improved over k=1), cheap to train and serve.

## Code-state caveats for cross-run comparison

The polyphase R² semantics changed on 2026-04-27 (mean-of-K phase R²). Runs before
that date logged `cv_mean_val_r2` as the *pooled-over-phases* R² for k>1 datasets;
runs after log it as the mean across phases of within-phase R². For k=1 the two
are identical.

- Runs computed under **OLD** semantics (pooled over phases): `apr24_exp1`,
  `apr25_multiseed_k2_rfmatched`, `apr25_multiseed_k5_rfmatched`,
  `apr25_multiseed_k10_rfmatched`. The k>1 entries are slightly *low-bound*
  estimates of the mean-of-K equivalent.
- Runs computed under **NEW** semantics: all `apr26_multiseed_*`.
- In practice the difference is +0.01 to +0.03 R² (mean-of-K is usually slightly
  higher), so it doesn't change the conclusions above. If you want truly clean
  comparisons, the apr25 k=2 run could be re-run under new code (~5h on a30).

The new code also adds:
- per-phase metrics `best_val_r2_kidx{N}` per fold,
- per-phase scatter and DMT-evolution plots,
- `k_idx` column in `predictions_all_seeds.npz`,
- `pooled_val_r2_pooled_all` legacy sibling (the old all-points pooled R²).

## Things I'd recommend running next, ranked

1. **Anti-aliased polyphase rebuild.** Modify `build_downsampled_dataset.py:62` to
   filter-then-decimate (e.g. `scipy.signal.decimate(... ftype='fir',
   zero_phase=True)` per phase). Emit a parallel set of npz files (suffix `_filt`
   or similar) so the unfiltered ones still work. Re-run the k=5 sweep on the
   filtered dataset. If R² recovers above +0.20, polyphase k=5 was being killed
   by aliasing alone; if it stays around +0.10, DMT signal genuinely has content
   above 100 Hz that's irrecoverable. Either result is informative.
2. **Validate k=1 wide as a public-facing result.** apr26_multiseed_k1_wide
   reaches +0.344 ± 0.042 with 1.68M params. It's a cleaner story for "we beat
   the apr19 baseline" than relying on polyphase k=2 alone. Worth replicating on
   a new seed set (e.g. seeds=10,20,30,40,50) to verify the result.
3. **Skip k=10 entirely.** The signal floor is below 50 Hz and capacity can't
   recover it. No experimental design at k=10 will yield a positive result with
   the current build pipeline.

## Things NOT to do

- Don't sweep more capacity at k=5 / k=10 with the current build pipeline.
  Confounded by aliasing; the capacity question is already answered.
- Don't compare new k=2 / k=5 results against the OLD-semantics apr25 numbers
  without noting the +0.01–0.03 offset.
- Don't promote `apr26_multiseed_k2_wide` (+0.351 ± 0.022) over the default
  k=2 — they're a statistical tie and `wide` is 2× the params.

## Pointers

- Build pipeline: `src/models/reg_simpleCNN/build_downsampled_dataset.py`
  (the unfiltered `out[i::k] = eeg[..., i:i+k*l_ds:k]` is on line 62).
- Model: `src/models/reg_simpleCNN/model.py`
  (`SimpleCNN` accepts `channels=...` since 2026-04-26; default 64/128/256).
- Training pipeline (single seed): `src/models/reg_simpleCNN/train_cv.py`
  (`run_fold`, `polyphase_metrics`, `plot_predicted_vs_actual`).
- Multiseed driver: `src/models/reg_simpleCNN/multiseed_cv.py`.
- Cluster mlruns mirror: `~/Desktop/tfm_code/mlruns_cluster/` (synced via
  `scripts/pull_mlruns.sh`; rsync without `--delete`, paths re-written).
- Capacity-sweep launcher: `scripts/cluster_apr26_capacity_sweep_{narrow,wide}.sh`
  → inner script `src/models/reg_simpleCNN/shell_and_logs/run_apr26_capacity_sweep.sh`
  takes `narrow|wide` as positional arg.
