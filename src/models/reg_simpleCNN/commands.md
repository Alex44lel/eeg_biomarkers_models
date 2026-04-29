# reg_simpleCNN — common commands

All commands run from the project root (`tfm_code/`).

Section index:
1. Build datasets (PK, biexp, polyphase, anti-aliased, baseline-aware)
2. Single-seed LOSO CV — `train_cv.py`
3. Multi-seed LOSO CV — `multiseed_cv.py`
4. Linear subject adaptation experiment (baseline subtraction)
5. Recipes from past experiments (apr19 best, ablations, sweeps)
6. Notes (RF math, leakage guarantees, MLflow)

---

## 1. Build datasets

The base npz `data/eeg_dmt_regression.npz` is the ground-truth supervised set
(post-injection 3-s trials with PK posterior-mean plasma labels). Everything
else is derived from it.

### 1.1 PK posterior-mean plasma labels (the canonical dataset)

```bash
# Default — t in [2, 15] minutes post-injection, PK trace from results/paper_model/...
python -m src.models.reg_simpleCNN.build_dmt_dataset

# Custom time window
python -m src.models.reg_simpleCNN.build_dmt_dataset --t-min 1 --t-max 20

# Custom PK trace
python -m src.models.reg_simpleCNN.build_dmt_dataset \
    --trace results/paper_model/only_plasma/partial_pooling_trace.nc

# Baseline-aware: also collect ALL pre-injection trials (t<0) with NaN labels
# and is_baseline=True — required for the linear-adaptation experiment.
python -m src.models.reg_simpleCNN.build_dmt_dataset \
    --include-baseline \
    --out-name eeg_dmt_regression_with_baseline.npz
```

### 1.2 Per-subject bi-exponential plasma fit (alt label source)

Same trial windows as the PK build but labels come from a 2-exponential fit to
each subject's measured plasma points → exposed as `--dataset biexp`.

```bash
python -m src.models.reg_simpleCNN.build_biexp_dataset
```

### 1.3 Polyphase-downsampled views (`pk_k{K}`)

Reads `data/eeg_dmt_regression.npz` and writes one
`data/eeg_dmt_regression_k{K}.npz` per factor `K`. Each row is a polyphase view
`eeg[:, i::K]` (length `3000 // K`); the K splits of each parent trial keep the
same subject ID / label / time and gain `k_idx ∈ [0, K)` plus `orig_trial_id`.
Sanity asserts run on every build.

```bash
# Default — builds k=2, k=3, k=4 in one go
python -m src.models.reg_simpleCNN.build_downsampled_dataset

# Specific factors
python -m src.models.reg_simpleCNN.build_downsampled_dataset --k 5 10

# All currently-registered factors
python -m src.models.reg_simpleCNN.build_downsampled_dataset --k 2 3 4 5 10

# From a baseline-aware source — `is_baseline` is propagated to every phase
python -m src.models.reg_simpleCNN.build_downsampled_dataset \
    --src   data/eeg_dmt_regression_with_baseline.npz \
    --prefix eeg_dmt_regression_with_baseline \
    --k 2 3 4

# Custom source / output dir / filename prefix
python -m src.models.reg_simpleCNN.build_downsampled_dataset \
    --src   data/eeg_dmt_regression.npz \
    --k     2 \
    --out-dir data/ \
    --prefix  eeg_dmt_regression
```

Registered keys after building: `pk_k2`, `pk_k3`, `pk_k4`, `pk_k5`, `pk_k10`.

### 1.4 Anti-aliased polyphase (`pk_k{K}_filt`)

Same row shape as `pk_k{K}` but applies an FIR low-pass before each phase is
taken — eliminates aliasing of >Nyquist content into the kept band. Use to
test whether the k=5/k=10 collapse on the unfiltered builds was caused by
aliasing or genuine info loss.

```bash
python -m src.models.reg_simpleCNN.build_downsampled_dataset_filt --k 2 5 10
```

Registered keys after building: `pk_k2_filt`, `pk_k5_filt`, `pk_k10_filt`.

---

## 2. Single-seed LOSO CV — `train_cv.py`

One pass through all 8 LOSO folds with a fixed seed. Per-fold and aggregate
metrics logged to MLflow / DagsHub.

### 2.1 Sensible defaults

```bash
# Defaults match the apr19 best config (run55 stack: 64/128/256, EMA 0.999,
# Huber β=10, kernels=15/7/7, strides=8/4/4)
python -m src.models.reg_simpleCNN.train_cv \
    --run_name cv_baseline
```

### 2.2 Switch label source / dataset variant

```bash
# bi-exponential labels
python -m src.models.reg_simpleCNN.train_cv \
    --dataset biexp \
    --run_name cv_biexp

# Polyphase-downsampled (k=2)
python -m src.models.reg_simpleCNN.train_cv \
    --dataset pk_k2 \
    --run_name cv_pk_k2

# Anti-aliased polyphase
python -m src.models.reg_simpleCNN.train_cv \
    --dataset pk_k5_filt \
    --run_name cv_pk_k5_filt

# Polyphase BUT keep only k_idx==0 (no phase repetition; pure low-rate baseline)
python -m src.models.reg_simpleCNN.train_cv \
    --dataset pk_k4 --single_phase \
    --run_name cv_pk_k4_singlephase

# Ad-hoc npz (bypasses --dataset registry)
python -m src.models.reg_simpleCNN.train_cv \
    --data_path data/eeg_dmt_regression_with_baseline.npz \
    --run_name cv_explicit_path
```

### 2.3 Hyperparameter overrides

```bash
# Custom optimizer + custom architecture
python -m src.models.reg_simpleCNN.train_cv \
    --lr 5e-4 --batch_size 64 --dropout 0.3 --weight_decay 1e-4 \
    --kernels 15 7 7 --strides 8 4 4 \
    --epochs 300 --patience 40 \
    --loss smoothl1 --huber_beta 10 \
    --description "apr19 best config replay" \
    --run_name cv_apr19_replay

# Wider channels (capacity ablation at fixed RF)
python -m src.models.reg_simpleCNN.train_cv \
    --kernels 15 7 7 --strides 8 4 4 \
    --channels 96 192 384 \
    --description "wider channels at fixed RF" \
    --run_name cv_wider_channels

# More layers / longer RF
python -m src.models.reg_simpleCNN.train_cv \
    --kernels 15 7 7 7 --strides 8 4 4 2 \
    --description "4-block stack, longer RF" \
    --run_name cv_4block

# Restrict to hardest folds for fast iteration
python -m src.models.reg_simpleCNN.train_cv \
    --subjects S05 S13 \
    --run_name cv_hard_subjects
```

### 2.4 Useful flag reference (`train_cv.py` only)

| Flag | Default | What it does |
|---|---|---|
| `--lr` / `--batch_size` / `--epochs` / `--patience` | 5e-4 / 64 / 300 / 40 | optimizer + early-stop budget |
| `--dropout` / `--weight_decay` | 0.3 / 1e-4 | regularisation |
| `--loss {mse,smoothl1}` | smoothl1 | loss function |
| `--huber_beta` | 10.0 | smoothl1 β (ng/mL) |
| `--mixup_alpha` | 0.0 | enable mixup with `Beta(α, α)` |
| `--kernels` / `--strides` / `--channels` | 15 7 7 / 8 4 4 / default 64 128 256 | per-block conv config |
| `--no_se` | off | drop Squeeze-and-Excitation blocks |
| `--early_stop {r2,loss}` | r2 | early-stop criterion (max R² / min loss) |
| `--single_phase` | off | on polyphase datasets, drop phases ≠ 0 |
| `--subjects` | all 8 | restrict CV pool |
| `--seed` | 42 | global RNG seed |
| `--log_model` | off | also log each fold's best PyTorch checkpoint to MLflow |
| `--experiment_name` / `--run_name` | `SimpleCNN_DMT_regression_CV` / auto | MLflow names |
| `--description` | "" | human-readable note logged with the run |
| `--baseline_subtraction` | off | linear subject adaptation (see §4) |

---

## 3. Multi-seed LOSO CV — `multiseed_cv.py`

Runs the full LOSO sweep N times with different seeds → honest error bars on
`cv_mean_val_r2` and per-fold stability.

```bash
# Defaults: 5 seeds (42, 123, 7, 2024, 0), apr19 best hyperparams
python -m src.models.reg_simpleCNN.multiseed_cv \
    --run_name multiseed_apr19_best

# Polyphase-downsampled variants
python -m src.models.reg_simpleCNN.multiseed_cv --dataset pk_k2  --run_name multiseed_k2
python -m src.models.reg_simpleCNN.multiseed_cv --dataset pk_k3  --run_name multiseed_k3
python -m src.models.reg_simpleCNN.multiseed_cv --dataset pk_k4  --run_name multiseed_k4
python -m src.models.reg_simpleCNN.multiseed_cv --dataset pk_k5  --run_name multiseed_k5
python -m src.models.reg_simpleCNN.multiseed_cv --dataset pk_k10 --run_name multiseed_k10

# Anti-aliased polyphase sweep
python -m src.models.reg_simpleCNN.multiseed_cv --dataset pk_k2_filt  --run_name multiseed_k2_filt
python -m src.models.reg_simpleCNN.multiseed_cv --dataset pk_k5_filt  --run_name multiseed_k5_filt
python -m src.models.reg_simpleCNN.multiseed_cv --dataset pk_k10_filt --run_name multiseed_k10_filt

# 8-seed stability check with explicit seeds
python -m src.models.reg_simpleCNN.multiseed_cv \
    --seeds 1 2 3 4 5 6 7 8 \
    --description "apr19 8-seed stability check" \
    --run_name multiseed_apr19_8seeds

# Custom MLflow experiment to keep sweeps separate
python -m src.models.reg_simpleCNN.multiseed_cv \
    --experiment_name SimpleCNN_polyphase_sweep \
    --dataset pk_k4 \
    --run_name multiseed_k4_5seeds
```

Same dataset / model / training flags as `train_cv.py`. Extras:
- `--seeds 42 123 7 ...` — explicit seed list (default 5 seeds).
- Parent-run metrics: `across_seed_mean_cv_r2`, `across_seed_std_cv_r2`,
  `per_fold_{subject}_mean_r2`. Artifacts: `predictions_all_seeds.npz`,
  `per_subject_r2_stability.png`, `across_seed_r2_summary.png`.

---

## 4. Linear subject adaptation experiment (baseline subtraction)

Subtracts each subject's mean pre-injection feature vector from post-injection
features before the regressor:

    feat' = extract_features(x) - λ · μ_s

where `μ_s` is the per-subject baseline computed by averaging
`extract_features` over that subject's pre-injection trials, and `λ` is a
learnable scalar (init 1.0). For LOSO val the held-out subject's own
pre-injection trials are used (no leakage — pre-inj plasma ≈ 0).

### 4.1 One-time dataset prep

```bash
# Build the baseline-aware base npz (post-inj + ALL pre-inj trials)
python -m src.models.reg_simpleCNN.build_dmt_dataset \
    --include-baseline \
    --out-name eeg_dmt_regression_with_baseline.npz

# (Optional) downsampled views — `is_baseline` propagates to every phase
python -m src.models.reg_simpleCNN.build_downsampled_dataset \
    --src    data/eeg_dmt_regression_with_baseline.npz \
    --prefix eeg_dmt_regression_with_baseline \
    --k 2 3 4
```

### 4.2 Train with linear adaptation

```bash
# Single-seed LOSO with baseline subtraction enabled
python -m src.models.reg_simpleCNN.train_cv \
    --data_path data/eeg_dmt_regression_with_baseline.npz \
    --baseline_subtraction \
    --description "linear adaptation: full pre-inj baseline, learnable λ" \
    --run_name cv_linadapt

# Same on a polyphase view
python -m src.models.reg_simpleCNN.train_cv \
    --data_path data/eeg_dmt_regression_with_baseline_k2.npz \
    --baseline_subtraction \
    --run_name cv_linadapt_k2

# Multi-seed for honest error bars
python -m src.models.reg_simpleCNN.multiseed_cv \
    --data_path data/eeg_dmt_regression_with_baseline.npz \
    --baseline_subtraction \
    --description "linear adaptation 5-seed" \
    --run_name multiseed_linadapt
```

What gets logged extra: `baseline_lambda` per epoch, `baseline_subtraction=True`
parameter on every fold's MLflow run.

---

## 5. Recipes from past experiments

These are named, working configs from the `train.md` autoresearch trail. Copy
and adapt — every flag is meaningful.

### 5.1 apr19 best — `cv_mean_val_r2 = +0.358`

The reigning config. SimpleCNN with wider channels (64/128/256), EMA decay
0.999, Huber β=10. Defaults of `train_cv.py` already match this; the explicit
form is below for the record.

```bash
python -m src.models.reg_simpleCNN.train_cv \
    --kernels 15 7 7 --strides 8 4 4 \
    --channels 64 128 256 \
    --dropout 0.3 --weight_decay 1e-4 \
    --lr 5e-4 --batch_size 64 \
    --epochs 300 --patience 40 \
    --loss smoothl1 --huber_beta 10 \
    --early_stop r2 \
    --description "apr19 best (run55 stack)" \
    --run_name cv_apr19_best
```

### 5.2 SE ablation

```bash
python -m src.models.reg_simpleCNN.train_cv \
    --no_se \
    --description "ablate SE blocks" \
    --run_name cv_no_se
```

### 5.3 Mixup augmentation

```bash
python -m src.models.reg_simpleCNN.train_cv \
    --mixup_alpha 0.2 \
    --description "mixup α=0.2 for label-scale interpolation" \
    --run_name cv_mixup_a02
```

### 5.4 MSE instead of Huber

```bash
python -m src.models.reg_simpleCNN.train_cv \
    --loss mse \
    --description "ablate loss: MSE instead of smoothl1" \
    --run_name cv_mse_loss
```

### 5.5 Hard-fold focus (S05 + S13 only)

Useful for fast iteration on the cross-subject scale-mismatch problem; full
LOSO is the source of truth, so don't promote configs that only help here.

```bash
python -m src.models.reg_simpleCNN.train_cv \
    --subjects S05 S13 \
    --description "hard folds only — fast iteration" \
    --run_name cv_hard_only
```

### 5.6 Receptive-field sweep

```bash
# 2 blocks (RF ≈ 71 ms)
python -m src.models.reg_simpleCNN.train_cv \
    --kernels 15 7 --strides 8 4 \
    --description "2-block, shorter RF" \
    --run_name cv_2block

# 3 blocks default (RF = 255 ms)  → apr19 best, see §5.1

# 4 blocks (RF ≈ 0.5 s)
python -m src.models.reg_simpleCNN.train_cv \
    --kernels 15 7 7 7 --strides 8 4 4 2 \
    --description "4-block, RF≈0.5 s" \
    --run_name cv_4block

# 5 blocks (RF ≈ 1 s)
python -m src.models.reg_simpleCNN.train_cv \
    --kernels 15 7 7 7 7 --strides 8 4 4 2 2 \
    --description "5-block, RF≈1 s" \
    --run_name cv_5block
```

### 5.7 Polyphase aliasing diagnostic

Compare same-K filtered vs unfiltered to attribute degradation to aliasing
vs information loss.

```bash
python -m src.models.reg_simpleCNN.multiseed_cv --dataset pk_k5      --run_name multiseed_k5
python -m src.models.reg_simpleCNN.multiseed_cv --dataset pk_k5_filt --run_name multiseed_k5_filt
# diff cv_mean_val_r2 → if filt >> unfiltered, the gap is aliasing.
```

---

## 6. Notes

- **Receptive field across `k`**: the same `(15, 7, 7) / (8, 4, 4)` config has
  RF=255 *samples* regardless of `k`, but in *raw-rate ms* that is 255·k ms
  (510 ms at k=2, 2 550 ms at k=10). Compare CV results across `k` with that
  in mind.
- **Subject grouping**: LOSO keeps all K polyphase splits of a subject
  together by construction. `train_cv.run_fold` asserts subject +
  `orig_trial_id` disjointness on every fold.
- **Sanity gate**: across all 8 LOSO folds, no more than 4 should have
  `val_r2 < 0`. The autoresearch loop in `train.md` enforces this.
- **MLflow / DagsHub**: both training scripts call `dagshub.init(...)` and
  write to the `Alex44lel/eeg_biomarkers_models` repo. Local UI:
  `mlflow ui --backend-store-uri mlruns/`.
- **DagsHub overhead**: each run pays ~10 s/epoch in remote HTTP calls. For
  local-only iterations, set `LOCAL=--local` (or whatever the relevant entry
  point exposes) before running.
- **Autoresearch protocol**: see `train.md`. One change per commit, keep only
  if `cv_mean_val_r2` strictly improves; otherwise `git reset --hard HEAD~1`.
- **Results log**: append per-run summaries to `results.tsv` (gitignored).
