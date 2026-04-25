#!/bin/bash
# apr25 polyphase-downsampling sweep — multi-seed LOSO CV at k=2, k=5, k=10.
#
# Goal: test whether the polyphase data augmentation (each parent trial
# split into K interleaved sub-signals at a K× lower effective sample rate)
# helps generalisation, while keeping the temporal *receptive field in
# raw-rate ms* matched to the k=1 baseline (kernels=63/15/15, strides=8/4/4,
# RF=623 ms @ 1 kHz). Without this match, larger k would silently see a
# wider raw-time context and the comparison would not be apples-to-apples.
#
# RF math: at polyphase factor K, each sub-signal sample spans K ms of raw
# time, so RF_raw_ms = RF_samples × K. Targets:
#     k=2   RF_samples ≈ 312    → kernels=(31, 8, 8)   RF=311  raw=622 ms (Δ −0.2%)
#     k=5   RF_samples ≈ 125    → kernels=(13, 3, 4)   RF=125  raw=625 ms (Δ +0.3%)
#     k=10  RF_samples ≈  62    → kernels=(15, 3, 2)   RF=63   raw=630 ms (Δ +1.1%)
# Strides held at (8, 4, 4); all forward passes verified to compile.
#
# Caveat: matching RF via kernels alone shrinks the first-conv kernel and
# therefore the parameter count drops with k (≈788k @ k=1 → ≈436k @ k=2 →
# ≈227k @ k=5 → ≈166k @ k=10). Capacity is a confound to keep in mind when
# reading the results.
#
# Other hyperparams mirror the apr19 best config (run55, cv_mean_val_r2=+0.358):
# lr=5e-4, batch_size=64, dropout=0.3, weight_decay=1e-4, smoothl1 huber_beta=10,
# wider channels 64/128/256 (model default), EMA 0.999 (per-batch),
# early-stop on val_r2, patience=40, max_epochs=300. Seeds default = (42, 123, 7, 2024, 0).
#
# Run from project root:
#   bash src/models/reg_simpleCNN/shell_and_logs/run_apr25_polyphase_rf_matched.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python}"
MODULE="src.models.reg_simpleCNN.multiseed_cv"
COMMON="--lr 5e-4 --batch_size 64 --epochs 300 --patience 40 --dropout 0.3
        --weight_decay 1e-4 --loss smoothl1 --huber_beta 10.0
        --strides 8 4 4
        --experiment_name SimpleCNN_DMT_regression_CV_seed"

echo "========================================================"
echo "  apr25 polyphase RF-matched sweep (3 multi-seed runs)"
echo "  k=2, k=5, k=10 — RF held ≈ 623 raw-ms (k=1 baseline)"
echo "========================================================"

$PYTHON -m $MODULE $COMMON \
    --dataset pk_k2 \
    --kernels 31 8 8 \
    --run_name "apr25_multiseed_k2_rfmatched" \
    --description "Polyphase k=2 (L=1500). RF-matched to k=1 baseline (kernels=63/15/15, RF=623ms): kernels=(31,8,8) → RF=311 sub-samples = 622 raw-ms (Δ-0.2%). Tests whether the 2× polyphase augmentation (each trial → 2 interleaved sub-signals at 500 Hz effective rate) improves generalisation under a fair RF comparison. Param count ≈ 436k vs 788k baseline — capacity is a confound."

echo ">>> apr25_multiseed_k2_rfmatched done"

$PYTHON -m $MODULE $COMMON \
    --dataset pk_k5 \
    --kernels 13 3 4 \
    --run_name "apr25_multiseed_k5_rfmatched" \
    --description "Polyphase k=5 (L=600). RF-matched to k=1 baseline (kernels=63/15/15, RF=623ms): kernels=(13,3,4) → RF=125 sub-samples = 625 raw-ms (Δ+0.3%). 5× polyphase augmentation (each trial → 5 interleaved sub-signals at 200 Hz effective rate). Param count ≈ 227k — much smaller than baseline; if R² holds up, the model is wildly over-parameterised at k=1."

echo ">>> apr25_multiseed_k5_rfmatched done"

$PYTHON -m $MODULE $COMMON \
    --dataset pk_k10 \
    --kernels 15 3 2 \
    --run_name "apr25_multiseed_k10_rfmatched" \
    --description "Polyphase k=10 (L=300). RF-matched to k=1 baseline (kernels=63/15/15, RF=623ms): kernels=(15,3,2) → RF=63 sub-samples = 630 raw-ms (Δ+1.1%). 10× polyphase augmentation (each trial → 10 interleaved sub-signals at 100 Hz effective rate). Param count ≈ 166k. Most aggressive downsampling: tests whether DMT signal survives loss of fine temporal detail when model sees 10× more samples."

echo ">>> apr25_multiseed_k10_rfmatched done"

echo ""
echo "========================================================"
echo "  All 3 apr25 polyphase experiments completed."
echo "  Compare against the k=1 baseline (apr24_exp1) in"
echo "  experiment SimpleCNN_DMT_regression_CV_seed."
echo "  View: mlflow ui --backend-store-uri mlruns/"
echo "========================================================"
