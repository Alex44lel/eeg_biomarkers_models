#!/bin/bash
# apr28 capacity sweep for k=3 and k=4 polyphase (unfiltered).
# Fills the missing k=3 / k=4 columns of the apr26 capacity grid.
#
# Usage (from project root):
#   bash src/models/reg_simpleCNN/shell_and_logs/run_apr28_k34_capacity_sweep.sh narrow
#   bash src/models/reg_simpleCNN/shell_and_logs/run_apr28_k34_capacity_sweep.sh default
#   bash src/models/reg_simpleCNN/shell_and_logs/run_apr28_k34_capacity_sweep.sh wide
#
# RF matching (~624 ms) via:
#   k=3  kernels 24 4 6 → RF = 24 + (4-1)*8 + (6-1)*32 = 208 sub-samples × 3 ms = 624 raw-ms
#   k=4  kernels 13 3 5 → RF = 13 + (3-1)*8 + (5-1)*32 = 157 sub-samples × 4 ms = 628 raw-ms
#
# Datasets (unfiltered, no _filt builds available for k=3/k=4):
#   pk_k3  → eeg_dmt_regression_k3.npz   (5820 trials × 32 ch × 1000 samples, Nyquist ≈ 167 Hz)
#   pk_k4  → eeg_dmt_regression_k4.npz   (7760 trials × 32 ch × 750  samples, Nyquist = 125 Hz)
#
# Param counts (approximate):
#                        NARROW (32/64/128)   DEFAULT (64/128/256)   WIDE (96/192/384)
#   k=3  kernels=24/4/6      ~79k                  ~288k                ~639k
#   k=4  kernels=12/3/5      ~41k                  ~150k                ~334k
#
# Hyperparams identical to apr26 / apr19 best (run55):
# lr=5e-4, batch_size=64, epochs=300, patience=40, dropout=0.3, weight_decay=1e-4,
# smoothl1 huber_beta=10, EMA 0.999, strides 8 4 4. Seeds default = (42, 123, 7, 2024, 0).

set -e

CAPACITY="${1:-}"
case "$CAPACITY" in
  narrow)  CHANNELS="--channels 32 64 128"  ;;
  default) CHANNELS="--channels 64 128 256" ;;
  wide)    CHANNELS="--channels 96 192 384" ;;
  *)
    echo "Usage: $0 {narrow|default|wide}" >&2
    echo "  narrow  → channels 32/64/128"   >&2
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
        --experiment_name SimpleCNN_DMT_regression_CV_seed"

echo "============================================================"
echo "  apr28 k=3/k=4 capacity sweep — capacity=${CAPACITY}"
echo "  channels: ${CHANNELS#--channels }"
echo "============================================================"

# --- k=3 (Nyquist ≈ 167 Hz — retains delta/theta/alpha/beta/gamma_low) ---
$PYTHON -m $MODULE $COMMON $CHANNELS \
    --dataset pk_k3 \
    --kernels 24 4 6 \
    --run_name "apr28_multiseed_k3_${CAPACITY}" \
    --description "Capacity sweep: k=3 polyphase (L=1000, pk_k3, Nyquist≈167Hz) at ${CAPACITY^^} channels (${CHANNELS#--channels }). RF=208 sub-samples × 3ms = 624 raw-ms (kernels=24/4/6, strides=8/4/4). Fills missing k=3 column of apr26 grid. k=3 sits between the strong k=2 (+0.345) and the recovering k=5_filt_wide result — expected moderate degradation vs k=2 since Nyquist=167Hz removes most of the high-gamma content but retains low-gamma and below."
echo ">>> apr28_multiseed_k3_${CAPACITY} done"

# --- k=4 (Nyquist = 125 Hz — retains delta/theta/alpha/beta; cuts most gamma) ---
$PYTHON -m $MODULE $COMMON $CHANNELS \
    --dataset pk_k4 \
    --kernels 13 3 5 \
    --run_name "apr28_multiseed_k4_${CAPACITY}" \
    --description "Capacity sweep: k=4 polyphase (L=750, pk_k4, Nyquist=125Hz) at ${CAPACITY^^} channels (${CHANNELS#--channels }). RF=157 sub-samples × 4ms = 628 raw-ms (kernels=13/3/5, strides=8/4/4). Fills missing k=4 column of apr26 grid. k=4 Nyquist=125Hz cuts high-gamma; expected between k=2 (+0.345) and k=5 (+0.132), likely closer to k=5 if gamma content is important for predicting plasma DMT."
echo ">>> apr28_multiseed_k4_${CAPACITY} done"

echo ""
echo "============================================================"
echo "  apr28 k=3/k=4 ${CAPACITY} sweep complete (2 runs)."
echo "  Reference comparisons in SimpleCNN_DMT_regression_CV_seed:"
echo "    k=2  ${CAPACITY}:  apr26_multiseed_k2_${CAPACITY}"
echo "    k=5  ${CAPACITY}:  apr26_multiseed_k5_${CAPACITY}"
echo "  Run all 3 capacities to complete the k=3/k=4 columns:"
echo "    bash src/models/reg_simpleCNN/shell_and_logs/run_apr28_k34_capacity_sweep.sh narrow"
echo "    bash src/models/reg_simpleCNN/shell_and_logs/run_apr28_k34_capacity_sweep.sh default"
echo "    bash src/models/reg_simpleCNN/shell_and_logs/run_apr28_k34_capacity_sweep.sh wide"
echo "============================================================"
