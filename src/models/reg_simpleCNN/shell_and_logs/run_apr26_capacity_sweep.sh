#!/bin/bash
# apr26 capacity-vs-polyphase sweep — runs ONE capacity row of the 4×2 grid.
#
# Usage:
#   bash run_apr26_capacity_sweep.sh narrow   # 4 runs at channels=(32,64,128)
#   bash run_apr26_capacity_sweep.sh wide     # 4 runs at channels=(96,192,384)
#
# HYPOTHESIS: the apr25 polyphase sweep collapsed at k=5 / k=10 partly because
# matching RF via kernels-only also shrank the model (~227k @ k=5, ~166k @ k=10
# vs 788k @ k=1 baseline). This sweep adds two channel-width points at every k
# so we can disentangle two confounded effects:
#
#   A)  Information loss: losing fine temporal detail when sampling at
#                         200 Hz (k=5) or 100 Hz (k=10) effective.
#   B)  Capacity loss:    the kernel shrink that came along for the ride.
#
# RF in raw-ms is held within 1.3% across all 8 runs (622–630 ms); RF/L ≈ 21%.
# The only things varying are k (information rate) and channel width (capacity).
#
# Param counts per cell:
#                          NARROW (32/64/128)    WIDE (96/192/384)
#   k=1   pk        kernels=63/15/15    229,945            1,675,945
#   k=2   pk_k2     kernels=31/8/8      125,497              932,521
#   k=5   pk_k5     kernels=13/3/4       64,057              490,153
#   k=10  pk_k10    kernels=15/3/2       49,721              348,841
#
# Other hyperparams identical to apr25 / apr19 best (run55):
# lr=5e-4, batch_size=64, dropout=0.3, weight_decay=1e-4, smoothl1 huber_beta=10,
# EMA 0.999 (per-batch), early-stop on val_r2 (mean-of-K phase R² when k>1),
# patience=40, max_epochs=300. Seeds default = (42, 123, 7, 2024, 0).

set -e

CAPACITY="${1:-}"
case "$CAPACITY" in
  narrow) CHANNELS="--channels 32 64 128"  ;;
  wide)   CHANNELS="--channels 96 192 384" ;;
  *)
    echo "Usage: $0 {narrow|wide}" >&2
    echo "  narrow → channels=(32,64,128)" >&2
    echo "  wide   → channels=(96,192,384)" >&2
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
echo "  apr26 capacity-vs-polyphase sweep — capacity=${CAPACITY}"
echo "  4 runs at channels=${CHANNELS#--channels }"
echo "============================================================"

# --- k=1 ---
$PYTHON -m $MODULE $COMMON $CHANNELS \
    --dataset pk \
    --kernels 63 15 15 \
    --run_name "apr26_multiseed_k1_${CAPACITY}" \
    --description "Capacity sweep: k=1 (no polyphase, L=3000) at ${CAPACITY^^} channels=${CHANNELS#--channels }. RF=623ms (kernels=63/15/15). Tests whether the apr19 / apr24_exp1 baseline is over- or under-parameterised compared to the 788k default."
echo ">>> apr26_multiseed_k1_${CAPACITY} done"

# --- k=2 ---
$PYTHON -m $MODULE $COMMON $CHANNELS \
    --dataset pk_k2 \
    --kernels 31 8 8 \
    --run_name "apr26_multiseed_k2_${CAPACITY}" \
    --description "Capacity sweep: k=2 polyphase (L=1500) at ${CAPACITY^^} channels=${CHANNELS#--channels }. RF=311 sub-samples = 622 raw-ms (Δ-0.2% vs k=1). Tests whether the apr25 k=2 win (+0.345) survives at this capacity, and whether the polyphase aug benefit scales with model size."
echo ">>> apr26_multiseed_k2_${CAPACITY} done"

# --- k=5 — heart of the hypothesis test ---
$PYTHON -m $MODULE $COMMON $CHANNELS \
    --dataset pk_k5 \
    --kernels 13 3 4 \
    --run_name "apr26_multiseed_k5_${CAPACITY}" \
    --description "Capacity sweep: k=5 polyphase (L=600) at ${CAPACITY^^} channels=${CHANNELS#--channels }. RF=125 sub-samples = 625 raw-ms (Δ+0.3% vs k=1). KEY EXPERIMENT — the apr25 k=5 collapse (+0.087) coincided with a 3.5× capacity drop vs baseline. If R² at WIDE recovers toward k=2 territory, the collapse was capacity-driven; if it stays near +0.10 the 200 Hz effective rate is the bottleneck and bigger models can't reconstruct missing temporal detail."
echo ">>> apr26_multiseed_k5_${CAPACITY} done"

# --- k=10 ---
$PYTHON -m $MODULE $COMMON $CHANNELS \
    --dataset pk_k10 \
    --kernels 15 3 2 \
    --run_name "apr26_multiseed_k10_${CAPACITY}" \
    --description "Capacity sweep: k=10 polyphase (L=300) at ${CAPACITY^^} channels=${CHANNELS#--channels }. RF=63 sub-samples = 630 raw-ms (Δ+1.1% vs k=1). Tests whether k=10's apr25 collapse (-0.04) was even partially recoverable with capacity; if it stays near zero, 100 Hz effective rate is below the signal floor regardless of model size."
echo ">>> apr26_multiseed_k10_${CAPACITY} done"

echo ""
echo "============================================================"
echo "  apr26 ${CAPACITY} sweep done (4/8 runs of the full grid)."
echo "  Pair with the matching ${CAPACITY/narrow/wide}${CAPACITY/wide/narrow} sweep + the"
echo "  apr25 default-channel runs to populate the full 4×3 grid."
echo "============================================================"
