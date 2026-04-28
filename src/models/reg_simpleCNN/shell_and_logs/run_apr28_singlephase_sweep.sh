#!/bin/bash
# apr28 SINGLE-PHASE polyphase sweep — k=2/3/4/5 on the *unfiltered* pk_kN
# datasets, keeping ONLY phase 0 of each parent window (--single_phase).
#
# HYPOTHESIS UNDER TEST: in the apr25/apr26/apr28 polyphase runs each parent
# 3-second window contributes K rows (one per phase). After (no) anti-aliasing
# the K phases share most of their information, so the K× row inflation can
# (a) act as cheap augmentation during training, and (b) inflate the effective
# n on the held-out subject — the per-window R² is averaged over K
# near-correlated phase predictions. Setting --single_phase keeps only k_idx==0
# so each parent trial appears exactly once: same number of independent
# windows as the k=1 baseline, but each window decimated by K (length
# 3000//K, sampling rate fs/K). This isolates the pure downsampling /
# bandwidth effect from the phase-repetition artifact.
#
# Usage (from project root):
#   bash src/models/reg_simpleCNN/shell_and_logs/run_apr28_singlephase_sweep.sh default
#   bash src/models/reg_simpleCNN/shell_and_logs/run_apr28_singlephase_sweep.sh wide
#
# Datasets (unfiltered — same npz files as apr26/apr28; --single_phase
# discards all rows with k_idx != 0 at load time, so k_factor effectively
# becomes 1 and the polyphase metric paths short-circuit):
#   pk_k2  → eeg_dmt_regression_k2.npz   (L=1500, fs/2 = 500 Hz, Nyquist 250 Hz)
#   pk_k3  → eeg_dmt_regression_k3.npz   (L=1000, fs/3 ≈ 333 Hz, Nyquist ≈167 Hz)
#   pk_k4  → eeg_dmt_regression_k4.npz   (L=750,  fs/4 = 250 Hz, Nyquist 125 Hz)
#   pk_k5  → eeg_dmt_regression_k5.npz   (L=600,  fs/5 = 200 Hz, Nyquist 100 Hz)
#
# RF in raw-ms is matched to the k=1 baseline (~624 ms) using the same
# per-k kernels as the multi-phase apr26/apr28 runs:
#   k=2  kernels=31/8/8   RF=311 sub-samples × 2 ms = 622 raw-ms
#   k=3  kernels=24/4/6   RF=208 sub-samples × 3 ms = 624 raw-ms
#   k=4  kernels=13/3/5   RF=157 sub-samples × 4 ms = 628 raw-ms
#   k=5  kernels=13/3/4   RF=125 sub-samples × 5 ms = 625 raw-ms
#
# Hyperparams identical to apr26/apr28 (apr19 best, run55):
# lr=5e-4, batch_size=64, epochs=300, patience=40, dropout=0.3,
# weight_decay=1e-4, smoothl1 huber_beta=10, EMA 0.999, strides 8 4 4.
# Seeds default = (42, 123, 7, 2024, 0).

set -e

CAPACITY="${1:-}"
case "$CAPACITY" in
  default) CHANNELS="--channels 64 128 256" ;;
  wide)    CHANNELS="--channels 96 192 384" ;;
  *)
    echo "Usage: $0 {default|wide}" >&2
    echo "  default → channels 64/128/256"  >&2
    echo "  wide    → channels 96/192/384"  >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python}"
MODULE="src.models.reg_simpleCNN.multiseed_cv"
COMMON="--lr 5e-4 --batch_size 64 --epochs 300 --patience 40 --dropout 0.3
        --weight_decay 1e-4 --loss smoothl1 --huber_beta 10.0
        --strides 8 4 4
        --single_phase
        --experiment_name SimpleCNN_DMT_regression_CV_seed"

echo "============================================================"
echo "  apr28 SINGLE-PHASE polyphase sweep — capacity=${CAPACITY}"
echo "  channels: ${CHANNELS#--channels }"
echo "  k = 2, 3, 4, 5  (unfiltered pk_kN, k_idx==0 only)"
echo "============================================================"

# --- k=2 single-phase (L=1500, Nyquist=250 Hz) ---
$PYTHON -m $MODULE $COMMON $CHANNELS \
    --dataset pk_k2 \
    --kernels 31 8 8 \
    --run_name "apr28_multiseed_k2_${CAPACITY}_singlephase" \
    --description "SINGLE-PHASE k=2 polyphase (pk_k2, k_idx==0 only, L=1500, Nyquist=250Hz) at ${CAPACITY^^} channels (${CHANNELS#--channels }). Same parent-window count as the k=1 baseline; each window decimated by 2. RF=311 sub-samples × 2ms = 622 raw-ms (kernels=31/8/8, strides=8/4/4). Compares directly to apr26_multiseed_k2_${CAPACITY} (multi-phase, same kernels/channels) — divergence isolates the phase-repetition contribution from the pure downsampling effect."
echo ">>> apr28_multiseed_k2_${CAPACITY}_singlephase done"

# --- k=3 single-phase (L=1000, Nyquist≈167 Hz) ---
$PYTHON -m $MODULE $COMMON $CHANNELS \
    --dataset pk_k3 \
    --kernels 24 4 6 \
    --run_name "apr28_multiseed_k3_${CAPACITY}_singlephase" \
    --description "SINGLE-PHASE k=3 polyphase (pk_k3, k_idx==0 only, L=1000, Nyquist≈167Hz) at ${CAPACITY^^} channels (${CHANNELS#--channels }). Same parent-window count as the k=1 baseline; each window decimated by 3. RF=208 sub-samples × 3ms = 624 raw-ms (kernels=24/4/6, strides=8/4/4). Compares directly to apr28_multiseed_k3_${CAPACITY} (multi-phase, same kernels/channels)."
echo ">>> apr28_multiseed_k3_${CAPACITY}_singlephase done"

# --- k=4 single-phase (L=750, Nyquist=125 Hz) ---
$PYTHON -m $MODULE $COMMON $CHANNELS \
    --dataset pk_k4 \
    --kernels 13 3 5 \
    --run_name "apr28_multiseed_k4_${CAPACITY}_singlephase" \
    --description "SINGLE-PHASE k=4 polyphase (pk_k4, k_idx==0 only, L=750, Nyquist=125Hz) at ${CAPACITY^^} channels (${CHANNELS#--channels }). Same parent-window count as the k=1 baseline; each window decimated by 4. RF=157 sub-samples × 4ms = 628 raw-ms (kernels=13/3/5, strides=8/4/4). Compares directly to apr28_multiseed_k4_${CAPACITY} (multi-phase, same kernels/channels)."
echo ">>> apr28_multiseed_k4_${CAPACITY}_singlephase done"

# --- k=5 single-phase (L=600, Nyquist=100 Hz) ---
$PYTHON -m $MODULE $COMMON $CHANNELS \
    --dataset pk_k5 \
    --kernels 13 3 4 \
    --run_name "apr28_multiseed_k5_${CAPACITY}_singlephase" \
    --description "SINGLE-PHASE k=5 polyphase (pk_k5, k_idx==0 only, L=600, Nyquist=100Hz) at ${CAPACITY^^} channels (${CHANNELS#--channels }). Same parent-window count as the k=1 baseline; each window decimated by 5. RF=125 sub-samples × 5ms = 625 raw-ms (kernels=13/3/4, strides=8/4/4). Compares directly to apr26_multiseed_k5_${CAPACITY} (multi-phase, same kernels/channels). At k=5 the phases are most aliased on the unfiltered build, so any gap vs the multi-phase result here is the strongest evidence that the phase-repetition was carrying the K-fold metric inflation."
echo ">>> apr28_multiseed_k5_${CAPACITY}_singlephase done"

echo ""
echo "============================================================"
echo "  apr28 single-phase ${CAPACITY} sweep complete (4 runs)."
echo "  Reference comparisons in SimpleCNN_DMT_regression_CV_seed:"
echo "    k=2  ${CAPACITY}:  apr26_multiseed_k2_${CAPACITY}    vs  apr28_multiseed_k2_${CAPACITY}_singlephase"
echo "    k=3  ${CAPACITY}:  apr28_multiseed_k3_${CAPACITY}    vs  apr28_multiseed_k3_${CAPACITY}_singlephase"
echo "    k=4  ${CAPACITY}:  apr28_multiseed_k4_${CAPACITY}    vs  apr28_multiseed_k4_${CAPACITY}_singlephase"
echo "    k=5  ${CAPACITY}:  apr26_multiseed_k5_${CAPACITY}    vs  apr28_multiseed_k5_${CAPACITY}_singlephase"
echo "  Run both capacities to fill the grid:"
echo "    bash src/models/reg_simpleCNN/shell_and_logs/run_apr28_singlephase_sweep.sh default"
echo "    bash src/models/reg_simpleCNN/shell_and_logs/run_apr28_singlephase_sweep.sh wide"
echo "  View: mlflow ui --backend-store-uri mlruns/"
echo "============================================================"
