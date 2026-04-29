#!/bin/bash
# apr29 BANDPOWER MVP — 6 multiseed runs that all share:
#   - dataset eeg_dmt_regression_with_baseline_k2 (--dataset pk_k2_with_baseline)
#     so pre-injection trials are present for the linear-adaptation runs
#   - --single_phase (k_idx==0 only; with k=2 all phases produce ~the same
#     band powers, redundancy without phase-repetition augmentation)
#   - --model bandpower_linear (paper-style spectral baseline)
#   - 6 default bands (δ θ α β low-γ high-γ to ~100 Hz at fs=500)
#     → 32 channels × 6 bands = 192 features per trial
#   - Welch nperseg=512 (matches the apr29 spectral MVP n_fft choice)
#   - run55-derived hyperparams: lr=5e-4 bs=64 dropout=0.3
#     loss=smoothl1 huber_beta=10 EMA 0.999 patience=40 max_epochs=300
#     5 seeds × 8 LOSO folds.
#   - **weight_decay=1e-3** (raised 10× from CNN runs) — linear models with
#     192 features and ~1700 train samples need stronger L2 than a CNN whose
#     conv kernels carry their own inductive bias.
#
# DESIGN INTENT:
#   Tests Bansal et al.'s "linear adaptation" recipe in its native habitat —
#   per-trial spectral features fed to a Linear classifier, with each subject's
#   resting (pre-injection) signature subtracted. Compared to spectral_cnn,
#   this isolates the effect of the SUBTRACTION rather than the architecture.
#
# THE 6 RUNS — full 3×2 grid over hidden width × baseline subtraction:
#                       baseline_sub off                       on
#   linear              1. apr29_bandpower_linear_default      2. apr29_bandpower_linear_baseline
#   mlp64               3. apr29_bandpower_mlp64_default       4. apr29_bandpower_mlp64_baseline
#   mlp128              5. apr29_bandpower_mlp128_default      6. apr29_bandpower_mlp128_baseline
#
#   1. apr29_bandpower_linear_default
#        Paper-canonical: a single Linear(192, 1) head, no subtraction.
#        Pure spectral baseline — what does an unconditioned, expressivity-
#        starved model learn from band powers alone? Floor for the family.
#   2. apr29_bandpower_linear_baseline
#        Same head + --baseline_subtraction. The cleanest possible test of
#        whether subject-mean subtraction helps: no nonlinearity to confound.
#        If 2 ≫ 1, the subtraction is doing real work; if 2 ≈ 1, subject
#        adaptation is not separable from the rest in this architecture.
#   3. apr29_bandpower_mlp64_default
#        1-hidden-layer MLP (192 → 64 → 1, ~12k params, ReLU + Dropout 0.3).
#        Tests whether a tiny nonlinearity over band powers beats both the
#        linear head and the much larger spectral_cnn / SimpleCNN runs.
#   4. apr29_bandpower_mlp64_baseline
#        Pairs with run 3. Subtraction × nonlinearity interaction probe.
#   5. apr29_bandpower_mlp128_default
#        MLP 192 → 128 → 1 (~25k params). Capacity ablation; if mlp128 ≫
#        mlp64 then capacity matters and the linear head was under-fit.
#   6. apr29_bandpower_mlp128_baseline
#        Pairs with run 5; full 3×2 grid completion.
#
# DECISION RULE (after this sweep):
#   - Any baseline-subtraction run >= +0.04 above its non-subtraction pair →
#     subject adaptation is the win, promote --baseline_subtraction across
#     SimpleCNN / spectral_cnn / bandpower_linear families.
#   - Any bandpower run >= +0.36 mean → spectral features alone carry enough
#     plasma signal that handcrafted features beat the CNN — surprising and
#     strong publication angle. Queue follow-up bands ablation.
#   - All bandpower runs noticeably below +0.30 → handcrafted spectral
#     features are too lossy; the CNN is learning frequency content the
#     band integrals discard. Move on.
#   - Any run below +0.20 → likely not enough capacity / wrong feature
#     scaling; sanity-check before drawing conclusions.
#
# Reference comparisons (same MLflow experiment):
#   apr28_multiseed_k2_default_singlephase   +0.3266 ± 0.031   (SimpleCNN, ~436k)
#   apr28_multiseed_k2_wide_singlephase      +0.3769 ± 0.037   (SimpleCNN, ~932k)
#   apr29_spectral_n256_default              ?  (~506k, STFT + 2D CNN)
#   apr29_spectral_n256_baseline_sub         ?  (~506k, + subject adaptation)
#
# Run from project root:
#   bash src/models/reg_simpleCNN/shell_and_logs/run_apr29_bandpower_mvp.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python}"
MODULE="src.models.reg_simpleCNN.multiseed_cv"
COMMON="--lr 5e-4 --batch_size 64 --epochs 300 --patience 40 --dropout 0.3
        --weight_decay 1e-3 --loss smoothl1 --huber_beta 10.0
        --model bandpower_linear
        --bandpower_welch_nperseg 512
        --dataset pk_k2_with_baseline
        --single_phase
        --experiment_name SimpleCNN_DMT_regression_CV_seed"

echo "============================================================"
echo "  apr29 BANDPOWER MVP — 6 multiseed runs"
echo "  All on pk_k2_with_baseline + --single_phase + --model bandpower_linear"
echo "  Welch features: 6 bands × 32 ch = 192, nperseg=512, log + train z-score"
echo "============================================================"

# --- Run 1: Linear head (paper-canonical), no baseline subtraction ---
$PYTHON -m $MODULE $COMMON \
    --run_name "apr29_bandpower_linear_default" \
    --description "Bandpower MVP, paper-canonical: Linear(192, 1) head on 6-band Welch features (δ θ α β low-γ high-γ to 100 Hz at fs=500), per-feature train-fit z-score, log power. No subject adaptation. Floor for the bandpower family — establishes whether handcrafted spectral features carry plasma signal at all."
echo ">>> apr29_bandpower_linear_default done"

# --- Run 2: Linear head + --baseline_subtraction ---
$PYTHON -m $MODULE $COMMON \
    --baseline_subtraction \
    --run_name "apr29_bandpower_linear_baseline" \
    --description "Bandpower MVP + linear subject adaptation: Linear(192, 1) head + --baseline_subtraction (subtract λ × per-subject mean pre-injection feature before the head). Cleanest possible test of whether subject-mean subtraction helps — no nonlinearity to confound. Compare directly to apr29_bandpower_linear_default to isolate the subtraction's effect."
echo ">>> apr29_bandpower_linear_baseline done"

# --- Run 3: MLP-64 head, no baseline subtraction ---
$PYTHON -m $MODULE $COMMON \
    --bandpower_hidden 64 \
    --run_name "apr29_bandpower_mlp64_default" \
    --description "Bandpower MVP, small MLP head: 192 → 64 (ReLU, Dropout 0.3) → 1. ~12k params. Tests whether a tiny nonlinearity over band powers beats both the literal-paper Linear head (run 1) and the much larger spectral_cnn / SimpleCNN runs (~500k–1M params)."
echo ">>> apr29_bandpower_mlp64_default done"

# --- Run 4: MLP-64 head + --baseline_subtraction ---
$PYTHON -m $MODULE $COMMON \
    --bandpower_hidden 64 \
    --baseline_subtraction \
    --run_name "apr29_bandpower_mlp64_baseline" \
    --description "Bandpower MVP + nonlinearity + subject adaptation: 192 → 64 → 1 MLP head + --baseline_subtraction. Pairs with run 3 to test subtraction × nonlinearity interaction."
echo ">>> apr29_bandpower_mlp64_baseline done"

# --- Run 5: MLP-128 head, no baseline subtraction ---
$PYTHON -m $MODULE $COMMON \
    --bandpower_hidden 128 \
    --run_name "apr29_bandpower_mlp128_default" \
    --description "Bandpower MVP, MLP-128 head: 192 → 128 → 1 (~25k params). Capacity ablation against run 3 (mlp64) — tests whether the linear / mlp64 heads were under-fit."
echo ">>> apr29_bandpower_mlp128_default done"

# --- Run 6: MLP-128 head + --baseline_subtraction ---
$PYTHON -m $MODULE $COMMON \
    --bandpower_hidden 128 \
    --baseline_subtraction \
    --run_name "apr29_bandpower_mlp128_baseline" \
    --description "Bandpower MVP + nonlinearity + subject adaptation: 192 → 128 → 1 MLP head + --baseline_subtraction. Pairs with run 5 (and forms the upper-right corner of the 3×2 grid)."
echo ">>> apr29_bandpower_mlp128_baseline done"

echo ""
echo "============================================================"
echo "  apr29 bandpower MVP sweep complete (6 runs)."
echo "  Reference comparisons in SimpleCNN_DMT_regression_CV_seed:"
echo "    apr28_multiseed_k2_default_singlephase   +0.3266 ± 0.031   (SimpleCNN, ~436k)"
echo "    apr28_multiseed_k2_wide_singlephase      +0.3769 ± 0.037   (SimpleCNN, ~932k)"
echo "    apr29_spectral_n256_default              ?  (~506k, STFT + 2D CNN)"
echo "    apr29_spectral_n256_baseline_sub         ?  (~506k, + subject adaptation)"
echo "    apr29_bandpower_linear_default           ?  (~200 params, paper-canonical)"
echo "    apr29_bandpower_linear_baseline          ?  (~200 params, + subject adaptation)"
echo "    apr29_bandpower_mlp64_default            ?  (~12k params)"
echo "    apr29_bandpower_mlp64_baseline           ?  (~12k params, + subject adaptation)"
echo "    apr29_bandpower_mlp128_default           ?  (~25k params)"
echo "    apr29_bandpower_mlp128_baseline          ?  (~25k params, + subject adaptation)"
echo "  View: mlflow ui --backend-store-uri mlruns/"
echo "============================================================"
