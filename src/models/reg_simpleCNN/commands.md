# reg_simpleCNN — common commands

All commands run from the project root (`tfm_code/`).

---

## 1. Build polyphase-downsampled datasets

`build_downsampled_dataset.py` reads `data/eeg_dmt_regression.npz` and writes one
`data/eeg_dmt_regression_k{K}.npz` per factor `K`. Each row of the output is a
polyphase view `eeg[:, i::K]` (length `3000 // K`); the K splits of each parent
trial keep the same subject ID / label / time and gain a `k_idx ∈ [0, K)` plus
`orig_trial_id` field. Sanity asserts run on every build.

```bash
# Default — builds k=2, k=3, k=4 in one go
python -m src.models.reg_simpleCNN.build_downsampled_dataset

# Specific factors
python -m src.models.reg_simpleCNN.build_downsampled_dataset --k 5 10

# All currently-registered factors
python -m src.models.reg_simpleCNN.build_downsampled_dataset --k 2 3 4 5 10

# Custom source / output dir / filename prefix
python -m src.models.reg_simpleCNN.build_downsampled_dataset \
    --src data/eeg_dmt_regression.npz \
    --k 2 \
    --out-dir data/ \
    --prefix eeg_dmt_regression
```

Registered dataset keys after building: `pk_k2`, `pk_k3`, `pk_k4`, `pk_k5`, `pk_k10`.

---

## 2. Single-seed LOSO CV — `train_cv.py`

One pass through all 8 LOSO folds with a fixed seed. Per-fold and aggregate
metrics are logged to MLflow / DagsHub.

```bash
# Minimum — defaults match the apr19 best config (run55 stack, wider 64/128/256,
# EMA 0.999, Huber β=10, kernels=15/7/7, strides=8/4/4)
python -m src.models.reg_simpleCNN.train_cv \
    --run_name cv_baseline

# Polyphase-downsampled dataset (k=2)
python -m src.models.reg_simpleCNN.train_cv \
    --dataset pk_k2 \
    --run_name cv_pk_k2

# Custom hyperparams + custom kernels/strides
python -m src.models.reg_simpleCNN.train_cv \
    --lr 5e-4 --batch_size 64 --dropout 0.3 --weight_decay 1e-4 \
    --kernels 15 7 7 --strides 8 4 4 \
    --epochs 300 --patience 40 \
    --loss smoothl1 --huber_beta 10 \
    --description "apr19 best config replay" \
    --run_name cv_apr19_replay

# Restrict CV to a subset of subjects (smaller fold count)
python -m src.models.reg_simpleCNN.train_cv \
    --subjects S05 S13 \
    --run_name cv_hard_subjects

# Ad-hoc npz path (bypasses --dataset registry)
python -m src.models.reg_simpleCNN.train_cv \
    --data_path data/eeg_dmt_regression_k4.npz \
    --run_name cv_pk_k4_explicit

# Custom MLflow experiment + run name
python -m src.models.reg_simpleCNN.train_cv \
    --experiment_name SimpleCNN_polyphase_sweep \
    --run_name cv_pk_k2_seed42 \
    --dataset pk_k2
```

Useful flags (full list in `parse_args` of `train_cv.py`):
- `--experiment_name` — MLflow experiment (default `SimpleCNN_DMT_regression_CV`).
- `--run_name` — MLflow run name *inside* the experiment (auto-generated if omitted).
- `--early_stop {r2,loss}` — early-stopping criterion (default `r2`).
- `--mixup_alpha 0.2` — enable mixup with `Beta(α, α)`.
- `--no_se` — drop Squeeze-and-Excitation blocks.
- `--log_model` — also log each fold's best PyTorch checkpoint to MLflow.

---

## 3. Multi-seed LOSO CV — `multiseed_cv.py`

Runs the full LOSO sweep N times with different seeds → honest error bars on
`cv_mean_val_r2` and per-fold stability. Defaults reproduce the apr19 best
config across 5 seeds.

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

# Custom seed list + override hyperparams
python -m src.models.reg_simpleCNN.multiseed_cv \
    --seeds 1 2 3 4 5 6 7 8 \
    --lr 5e-4 --batch_size 64 --dropout 0.3 --weight_decay 1e-4 \
    --kernels 15 7 7 --strides 8 4 4 \
    --epochs 300 --patience 40 \
    --description "apr19 8-seed stability check" \
    --run_name multiseed_apr19_8seeds

# Custom MLflow experiment to keep polyphase sweeps separate
python -m src.models.reg_simpleCNN.multiseed_cv \
    --experiment_name SimpleCNN_polyphase_sweep \
    --dataset pk_k4 \
    --run_name multiseed_k4_5seeds
```

Same dataset / model / training flags as `train_cv.py`. Extra flag:
- `--seeds 42 123 7 ...` — explicit seed list (default 5 seeds).
- `--experiment_name` — MLflow experiment (default `SimpleCNN_DMT_regression_CV`).

Parent-run metrics include `across_seed_mean_cv_r2`, `across_seed_std_cv_r2`,
`per_fold_{subject}_mean_r2`, etc. Artifacts include `predictions_all_seeds.npz`,
`per_subject_r2_stability.png`, `across_seed_r2_summary.png`.

---

## Notes

- **Receptive field across `k`**: the same `(15, 7, 7)/(8, 4, 4)` config has RF=255
  *samples* regardless of `k`, but in *raw-rate ms* that is 255·k ms (510 ms at
  k=2, 2 550 ms at k=10). Compare CV results across k with that in mind.
- **Subject grouping**: LOSO keeps all K polyphase splits of a subject together
  by construction. `train_cv.run_fold` asserts subject + `orig_trial_id`
  disjointness on every fold.
- **MLflow / DagsHub**: both training scripts call `dagshub.init(...)` and write
  to the `Alex44lel/eeg_biomarkers_models` repo. Local UI:
  `mlflow ui --backend-store-uri mlruns/`.
