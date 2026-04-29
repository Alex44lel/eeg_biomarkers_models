#!/bin/bash
# apr29 SIMPLECNN extra runs — 3 multiseed runs that all share:
#   - --model simplecnn (default; the 1D conv stack from model.py)
#   - dataset eeg_dmt_regression_k2 single-phase (k_idx==0 only)
#   - run55 hyperparams: lr=5e-4 bs=64 dropout=0.3 wd=1e-4 smoothl1 huber_beta=10
#     EMA 0.999, patience=40, max_epochs=300, 5 seeds × 8 LOSO folds
#   - strides 8/4/4 (default)
#
# DESIGN INTENT (companion to run_apr29_spectral_mvp.sh):
#   The spectral MVP swaps the temporal frontend; this script keeps the proven
#   1D conv backbone and probes two orthogonal levers we haven't tested yet at
#   the new single-phase honest-metric regime:
#     1. Bigger channels — keep the default k=2 RF-matched kernels (31, 8, 8)
#        but push the channel widths past the apr26 wide tier (96/192/384) up
#        to 128/256/512 (~1.6M params, +73% over wide). Tests whether channel
#        capacity has further headroom at the high-mean single-phase champ
#        (apr28_multiseed_k2_wide_singlephase, +0.377 ± 0.037).
#     2. Linear subject adaptation — toggle --baseline_subtraction on the two
#        existing k=2 single-phase champs (apr25 default, apr26 wide). Uses
#        the baseline-aware npz (pk_k2_with_baseline). Tests whether
#        subtracting each subject's mean pre-injection feature (× learnable λ)
#        closes the per-subject heterogeneity gap (S07/S12 vs S01/S05/S13)
#        seen across every prior run.
#
# THE 3 RUNS:
#   1. apr29_multiseed_k2_xwide
#        kernels (31, 8, 8), channels (128, 256, 512). ~1.61M params,
#        RF = 311 sub-samples × 2 ms = 622 raw-ms (matched). Pure channel-
#        capacity probe — same architecture/RF as apr26_k2_wide but at 1.33×
#        wider channels at every block. Compares directly to
#        apr28_multiseed_k2_wide_singlephase (+0.377 ± 0.037, 932k params).
#   2. apr29_multiseed_k2_wide_baseline_sub
#        kernels (31, 8, 8), channels (96, 192, 384), --baseline_subtraction.
#        ~932k params (+1 for λ). Same architecture as apr26_k2_wide /
#        apr28_multiseed_k2_wide_singlephase, but with subject adaptation on.
#        Direct subject-calibration ablation on the current high-mean champ.
#   3. apr29_multiseed_k2_default_baseline_sub
#        kernels (31, 8, 8), channels (64, 128, 256), --baseline_subtraction.
#        ~436k params. Same architecture as apr25_multiseed_k2_rfmatched
#        (+0.345 ± 0.021), but with subject adaptation on. Same probe as
#        run 2 at the cheaper / more stable param tier.
#
# DECISION RULE:
#   - Run 1 ≥ +0.39 mean → channel scaling has more headroom; queue an
#     even wider tier (160/320/640, ~2.5M params) and alt-seed replication.
#   - Run 2 or run 3 lifting their non-baseline-sub counterparts by ≥ +0.02
#     mean and improving the worst-subject (S01/S05/S13) median by ≥ +0.05
#     → subject calibration is a real lever; promote to a default flag for
#     all future SimpleCNN runs.
#   - All three near the existing champs (within ~1σ) → SimpleCNN is at its
#     ceiling; the spectral frontend (run_apr29_spectral_mvp.sh) and
#     subject-conditioned models become the next frontiers.
#
# Run from project root:
#   bash src/models/reg_simpleCNN/shell_and_logs/run_apr29_simplecnn_bigkern_baseline.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python}"
MODULE="src.models.reg_simpleCNN.multiseed_cv"
COMMON="--lr 5e-4 --batch_size 64 --epochs 300 --patience 40 --dropout 0.3
        --weight_decay 1e-4 --loss smoothl1 --huber_beta 10.0
        --strides 8 4 4
        --single_phase
        --model simplecnn
        --experiment_name SimpleCNN_DMT_regression_CV_seed"

echo "============================================================"
echo "  apr29 SIMPLECNN extras — 3 multiseed runs"
echo "  All on pk_k2 + --single_phase, --model simplecnn"
echo "============================================================"

# --- Run 1: bigger channels than wide, default k=2 kernels (no baseline_sub) ---
$PYTHON -m $MODULE $COMMON \
    --dataset pk_k2 \
    --kernels 31 8 8 \
    --channels 128 256 512 \
    --run_name "apr29_multiseed_k2_xwide" \
    --description "SimpleCNN extra-wide channel probe at k=2: default RF-matched kernels (31, 8, 8) at channels (128, 256, 512) — 1.33× the apr26 wide tier (96, 192, 384) at every block. ~1.61M params (+73% over wide). RF = 311 sub-samples × 2 ms = 622 raw-ms (unchanged, matches apr25 champ). Single-phase (k_idx==0 only). Tests whether channel capacity has further headroom past the apr28_multiseed_k2_wide_singlephase high-mean champ (+0.377 ± 0.037, 932k params). If R² climbs ≥ +0.02 with std ≤ 0.04, queue an even wider tier (160/320/640)."
echo ">>> apr29_multiseed_k2_xwide done"

# --- Run 2: wide channels + baseline_subtraction ---
# Uses the baseline-aware k=2 build (has is_baseline column) needed by the
# compute_mu_s_table refresh step in train_cv.run_fold.
$PYTHON -m $MODULE $COMMON \
    --dataset pk_k2_with_baseline \
    --kernels 31 8 8 \
    --channels 96 192 384 \
    --baseline_subtraction \
    --run_name "apr29_multiseed_k2_wide_baseline_sub" \
    --description "SimpleCNN wide + linear subject adaptation: kernels (31, 8, 8), channels (96, 192, 384), RF=622 raw-ms — identical architecture to apr26_multiseed_k2_wide / apr28_multiseed_k2_wide_singlephase but with --baseline_subtraction (subtract λ × per-subject mean pre-injection feature before regressor). ~932k params + 1 learnable scalar (λ). Dataset pk_k2_with_baseline + --single_phase. Probes whether subject calibration lifts the +0.377 high-mean champ and especially closes the S01/S05/S13 gap."
echo ">>> apr29_multiseed_k2_wide_baseline_sub done"

# --- Run 3: default channels + baseline_subtraction ---
$PYTHON -m $MODULE $COMMON \
    --dataset pk_k2_with_baseline \
    --kernels 31 8 8 \
    --channels 64 128 256 \
    --baseline_subtraction \
    --run_name "apr29_multiseed_k2_default_baseline_sub" \
    --description "SimpleCNN default + linear subject adaptation: kernels (31, 8, 8), channels (64, 128, 256), RF=622 raw-ms — identical architecture to apr25_multiseed_k2_rfmatched (+0.345 ± 0.021) but with --baseline_subtraction. ~436k params + 1 learnable scalar (λ). Dataset pk_k2_with_baseline + --single_phase. Same subject-adaptation probe as run 2 at the cheaper / more-stable param tier; cleanest comparison cell since the apr25 baseline used essentially the same config sans subtraction."
echo ">>> apr29_multiseed_k2_default_baseline_sub done"

echo ""
echo "============================================================"
echo "  apr29 simplecnn extras complete (3 runs)."
echo "  Reference comparisons in SimpleCNN_DMT_regression_CV_seed:"
echo "    apr25_multiseed_k2_rfmatched              +0.3448 ± 0.021  (default,  RF 622)"
echo "    apr26_multiseed_k2_wide                   +0.3512 ± 0.022  (wide,     RF 622, multi-phase)"
echo "    apr28_multiseed_k2_default_singlephase    +0.3266 ± 0.031  (default,  RF 622, single-phase)"
echo "    apr28_multiseed_k2_wide_singlephase       +0.3769 ± 0.037  (wide,     RF 622, single-phase, current high-mean)"
echo "    apr29_multiseed_k2_xwide                  ?  (channels 128/256/512, ~1.61M params, RF 622)"
echo "    apr29_multiseed_k2_wide_baseline_sub      ?  (wide + λ subject adaptation)"
echo "    apr29_multiseed_k2_default_baseline_sub   ?  (default + λ subject adaptation)"
echo "  View: mlflow ui --backend-store-uri mlruns/"
echo "============================================================"
