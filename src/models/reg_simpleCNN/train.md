# autoresearch — DMT Plasma EEG Regression (Start Simple)

Autonomous research loop for improving a regressor that predicts plasma DMT concentration (ng/mL) from 3-second EEG trials. The key principle: **start from the simplest possible model and incrementally add complexity only when it helps**.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr19`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**:
   - `src/models/reg_simpleCNN/model.py` — model architecture (SimpleCNN).
   - `src/models/reg_simpleCNN/train_cv.py` — LOSO cross-validation trainer (the script you actually run).
   - `src/models/reg_simpleCNN/dataset.py` — **READ ONLY**. Defines the task (subjects, labels, normalization, LOSO split).
4. **Verify data exists**: Check that `data/eeg_dmt_regression.npz` exists. If not, tell the human to run the preprocessing pipeline.
5. **Initialize results.tsv**: Create `src/models/reg_simpleCNN/results.tsv` with just the header row.
6. **Confirm and go**.

## Task context

- **Input**: 3-second EEG trials, shape `[B, 32, 3000]` (32 EEG channels × 3000 samples per trial).
- **Output**: Single scalar — predicted plasma DMT concentration in ng/mL (regression, labels stay raw ng/mL).
- **Primary metric**: `cv_mean_val_r2` (mean R² across LOSO folds — higher is better; perfect = 1.0, predicting the mean = 0, worse than mean = negative).
- **Secondary metrics (reported but not gates)**: MAE (ng/mL), RMSE (ng/mL), MSE, `cv_std_val_r2`, `pooled_val_r2`, `mean_best_epoch`.
- **Validation scheme**: Leave-One-Subject-Out (LOSO) cross-validation, fixed at 8 folds — one per subject in `["S01","S02","S05","S06","S07","S10","S12","S13"]`. Predictions on the held-out subject are used; folds are rotated; results are aggregated across folds.
- **Dataset**: 8 subjects, ~N trials each (varies by subject).
- **Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU 7 GB VRAM.

## The Starting Point

The baseline lives in `src/models/reg_simpleCNN/model.py` — `SimpleCNN`: three 1-D conv blocks (channels 32 → 64 → 128) with BatchNorm, ReLU, Squeeze-and-Excitation, and dropout, then global average pool and a linear head. Input `[B, 32, 3000]` → scalar prediction.

Default hyperparameters in `train_cv.py`: `lr=1e-3, batch_size=64, dropout=0.3, weight_decay=1e-4, epochs=300, patience=20` (early stop on val R², maximize). MSE loss, Adam optimizer.

Your job is to **iteratively improve it**, one change at a time.

## Experimentation

Each experiment runs a full LOSO CV sweep (8 folds). Launch with:

```
python -m src.models.reg_simpleCNN.train_cv --run_name <tag>_<runN> [--lr ... --batch_size ... --dropout ... --weight_decay ... --patience ... --epochs ... --seed 42]
```

Each fold trains a fresh model on 7 subjects and evaluates on the held-out subject. Early stopping per fold is on `val_r2`.

**What you CAN do:**

- Modify `src/models/reg_simpleCNN/model.py` — architecture (conv layers, kernel sizes, widths, dropout, SE, attention, residuals, heads, etc.).
- Modify `src/models/reg_simpleCNN/train_cv.py` — optimizer, LR schedule, loss function, gradient clipping, augmentation, mixup, early-stop patience logic, etc.
- Tune any CLI hyperparameter.

**What you CANNOT do:**

- Modify `src/models/reg_simpleCNN/dataset.py`. The subject list, labels, normalization, and LOSO split define the task.
- Change the metric functions or the CV iteration (`compute_regression_metrics`, the aggregation, pooled metrics). These define scoring.
- Change the early-stop metric to anything other than `val_r2`. You may tune patience.
- Install new packages. Only use what is already available.

Each phase builds on validated wins from the previous phase. **Only add one thing at a time** so you know what helped.

## Goals and Constraints

**The goal is simple: get the highest `cv_mean_val_r2`.** Monotonic improvement is the rule — a run is a KEEP only if `cv_mean_val_r2` strictly improves over the previous KEEP.

**Stability matters**: track `cv_std_val_r2` as a tiebreaker. A solution with 0.15 ± 0.05 is more robust than 0.20 ± 0.25, even though mean is higher; if std explodes, be skeptical.

**VRAM** is a hard constraint (7 GB). Do not exceed it.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome.

## Output format

At the end of every run, `train_cv.py` prints a structured summary. Critical sections (verbatim labels you can grep for):

```
======================================================================
  CV FOLD SUMMARY
======================================================================
 fold  subject  n_val  epochs    val_r2   val_mae  val_rmse    val_mse
    1      S01    ...     ...   +0.1234   12.3456   15.6789   250.1234
    ... (one row per fold)

======================================================================
  CV AGGREGATE (across folds)
======================================================================
metric                  mean        std        min        max
best_val_r2          +0.1234     0.0876    -0.0321    +0.2987
best_val_mae         12.3456     4.1234     8.7654    19.8765
best_val_rmse        15.6789     5.1234    10.2345    22.4567
best_val_mse        250.1234    80.5432   105.2345   520.4567
best_val_loss       250.1234    80.5432   105.2345   520.4567
best_epoch           45.3333    12.4321    22.0000    78.0000

======================================================================
  POOLED (concatenated held-out predictions, N=XXXX)
======================================================================
  pooled_val_r2:   +0.1234
  pooled_val_mae:  12.3456
  pooled_val_rmse: 15.6789
  pooled_val_mse: 250.1234

======================================================================
  RESULT (one-line, for autoresearch):
  RESULT cv_mean_val_r2=+0.1234 cv_std_val_r2=0.0876 cv_mean_val_mae=12.3456 cv_mean_val_rmse=15.6789 pooled_val_r2=+0.1234 pooled_val_mae=12.3456 pooled_val_rmse=15.6789 mean_best_epoch=45.3
======================================================================
  Total wall time: 1234.5s
```

All metrics are also logged to MLflow (parent run + one nested run per fold), at `mlruns/` relative to project root. For the loop the console `RESULT` line is sufficient.

Interpretation cheat sheet:

- `cv_mean_val_r2` — primary optimization target.
- `cv_std_val_r2` — cross-subject stability. Lower = more robust.
- `cv_min_val_r2` / `cv_max_val_r2` — worst / best fold. A very negative min on an otherwise-OK mean means one subject is catastrophic; investigate.
- `pooled_val_r2` — R² on concatenated held-out predictions (not the mean of per-fold R²). Sensitive to cross-subject label-range differences; useful for diagnosing.
- `mean_best_epoch` near `epochs` → consider more training time; far below → regularization kicks in fast.

## Logging results

Log each experiment to `src/models/reg_simpleCNN/results.tsv` (tab-separated). **Do NOT commit results.tsv** — leave it untracked (already covered by `.gitignore` patterns, or add it if not).

Header and columns:

```
commit	cv_mean_val_r2	cv_std_val_r2	cv_min_val_r2	cv_max_val_r2	cv_mean_val_mae	cv_std_val_mae	cv_mean_val_rmse	cv_mean_val_mse	pooled_val_r2	pooled_val_mae	pooled_val_rmse	mean_best_epoch	cv_scheme	n_folds	training_seconds	num_params	status	description
```

1. commit — git commit hash (short, 7 chars)  
   2–13. metrics (use 0 for crashes; keep sign on r² values)
2. `cv_scheme` — always `loso`
3. `n_folds` — always `8`
4. `training_seconds` — from `Total wall time:` line
5. `num_params` — from `Model parameters:` line (any fold — all are equal)
6. `status` — `keep`, `discard`, or `crash`
7. `description` — explanation of what was tried the reasoning to try it and a small brief analysis of what happened

Example:

```
commit	cv_mean_val_r2	cv_std_val_r2	cv_min_val_r2	cv_max_val_r2	cv_mean_val_mae	cv_std_val_mae	cv_mean_val_rmse	cv_mean_val_mse	pooled_val_r2	pooled_val_mae	pooled_val_rmse	mean_best_epoch	cv_scheme	n_folds	training_seconds	num_params	status	description
a1b2c3d	0.0421	0.1423	-0.1876	0.2654	15.32	4.81	20.15	405.87	0.0654	14.98	19.72	42	loso	8	812.3	124321	keep	baseline SimpleCNN
b2c3d4e	0.1102	0.1211	-0.0543	0.2987	14.21	4.56	18.95	372.51	0.1345	13.88	18.40	51	loso	8	954.1	124321	keep	lr=5e-4
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state: current branch/commit.
2. Modify code under `src/models/reg_simpleCNN/` (NOT `dataset.py`) with ONE experimental idea.
3. `git add -u && git commit -m "<short description>"`.
4. Run:
   ```
   python -m src.models.reg_simpleCNN.train_cv --run_name <tag>_<runN> > src/models/reg_simpleCNN/run.log 2>&1
   ```
5. Read and diagnose the run. Some greps you might find useful:

   **Final summary (the RESULT line is the canonical single source of truth)**:

   ```
   grep "^  RESULT " src/models/reg_simpleCNN/run.log
   ```

   **CV aggregate + pooled block** (mean/std/min/max across folds):

   ```
   awk '/^  CV AGGREGATE/,/^  Total wall time/' src/models/reg_simpleCNN/run.log
   ```

   **Per-fold table** (for the sanity-check on negative-r² folds):

   ```
   awk '/^  CV FOLD SUMMARY/,/^  CV AGGREGATE/' src/models/reg_simpleCNN/run.log
   ```

   **Per-epoch trace across all folds**:

   ```
   grep "^  Ep " src/models/reg_simpleCNN/run.log
   ```

   Each epoch line contains: `loss train/val | MAE train/val | RMSE train/val | R² train/val | seconds | [best val_r2 X@epY, pat Z/P]`.

   **Parameter count and wall time**:

   ```
   grep "^Model parameters:\|^  Total wall time:" src/models/reg_simpleCNN/run.log | head
   ```

   If the `RESULT` grep is empty, the run crashed. Run:

   ```
   tail -n 80 src/models/reg_simpleCNN/run.log
   ```

   to read the Python stack trace and attempt a fix. If you can't get things to work after a few attempts, give up — log `crash` and move on.

   you have access to all the logs, all folds, all epochs, all sumaries

6. Record in `src/models/reg_simpleCNN/results.tsv` (one row, tab-separated — **do not commit this file**, leave it untracked).
7. If `cv_mean_val_r2` strictly improved vs current best KEEP AND the sanity check passes (≤ 4 folds have `val_r2 < 0`) → **KEEP** (branch advances).
8. Otherwise → **DISCARD** (`git reset --hard HEAD~1` to drop the commit).

You are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. The branch is a monotonic trail of validated wins. Rewind further back sparingly (if ever).

**Timeout**: Each CV run is self-paced (early stop on `val_r2` per fold). Typical 10–30 min. Kill if > 60 min.

**Crashes**: If a run crashes (OOM, bug, etc.), use judgment — fix typos/imports and re-run, or skip fundamentally broken ideas with status `crash`.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep. You are autonomous. If you run out of ideas, think harder — re-read `model.py` and `train_cv.py` for angles, combine previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. The user then wakes up to experimental results, all completed by you while they slept!
