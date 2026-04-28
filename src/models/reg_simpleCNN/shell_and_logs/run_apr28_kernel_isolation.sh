#!/bin/bash
# apr28 KERNEL ISOLATION sweep — 2 multiseed runs that disentangle kernel size
# from polyphase sample-rate as causes of the apr25 k=5 collapse.
#
# BACKGROUND (see misc/study_notes/findings.md for the full story):
#   - The apr25 polyphase sweep showed k=5 (+0.087) and k=10 (-0.04) collapsing
#     vs k=2 (+0.345). The apr27 anti-aliasing experiment ruled out aliasing
#     as the cause: the source data turns out to be already low-passed at
#     ~100 Hz, so there's nothing above the new Nyquist to alias from.
#   - The remaining hypothesis: the collapse is *architectural*, driven by the
#     RF-match constraint forcing tiny kernels at high k (size 3, 4 in
#     deeper layers) and few output positions before pooling.
#
# THE 2×2 DESIGN — these two runs complete the matrix:
#
#                 small kernels (13, 3, 4)            medium kernels (31, 8, 8)
#   k=1 (pk)      apr28_multiseed_k1_smallkern  ←B    apr24_exp1 (kernels 63/15/15) — n/a
#   k=5 (pk_k5)   apr25_multiseed_k5_rfmatched +0.087 apr28_multiseed_k5_bigkern   ←A
#
# Exp A — apr28_multiseed_k5_bigkern:
#   k=5 dataset with k=2's kernels. Gives up RF-match (RF = 311 sub-samples
#   × 5 ms = 1555 raw-ms, 2.5× the apr25 baseline). Tests whether bigger
#   kernels + more output positions rescue k=5.
#   Predictions:
#     R² near +0.30+ → architectural cramping was the dominant cause; with
#                      proper kernels k=5 is fine.
#     R² near +0.10  → polyphase decimation at 5× still harms beyond
#                      architecture (likely time-domain feature granularity).
#
# Exp B — apr28_multiseed_k1_smallkern:
#   k=1 dataset with k=5's small kernels. RF = 125 raw-ms (much shorter than
#   the 623 ms baseline; same as apr25 k=5). Tests whether full-resolution
#   input rescues a small-kernel model.
#   Predictions:
#     R² near +0.30+ → polyphase rate also matters; small kernels are fine
#                      if you have 1 kHz time resolution.
#     R² near +0.10  → kernels are the dominant bottleneck regardless of rate.
#
# Hyperparams identical to apr25/apr26 default-channel runs:
# lr=5e-4, batch_size=64, dropout=0.3, weight_decay=1e-4, smoothl1 huber_beta=10,
# EMA 0.999, default channels (64/128/256), early-stop on val_r2 (mean-of-K
# phase R² when k>1), patience=40, max_epochs=300.
# Seeds default = (42, 123, 7, 2024, 0).
#
# Run from project root:
#   bash src/models/reg_simpleCNN/shell_and_logs/run_apr28_kernel_isolation.sh

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

echo "============================================================"
echo "  apr28 KERNEL ISOLATION sweep (2 runs)"
echo "  Disentangle kernel architecture from polyphase rate"
echo "============================================================"

# ============================================================
# Exp A: k=5 with k=2's kernels (architecture-rescue test)
# ============================================================
$PYTHON -m $MODULE $COMMON \
    --dataset pk_k5 \
    --kernels 31 8 8 \
    --run_name "apr28_multiseed_k5_bigkern" \
    --description "Kernel-isolation A: k=5 polyphase (L=600, pk_k5) with k=2's bigger kernels (31, 8, 8) instead of the apr25 RF-matched (13, 3, 4). RF = 311 sub-samples × 5 ms = 1555 raw-ms (vs baseline 622 raw-ms — 2.5× more temporal context). Default channels (64/128/256). Tests whether the apr25 k=5 collapse (+0.087) was caused by kernel-shrinking from the RF-match constraint. R² near +0.30+ → architectural cramping was the dominant cause; with proper kernels k=5 works. R² near +0.10 → polyphase decimation at 5× hurts beyond architecture (time-domain feature granularity)."
echo ">>> apr28_multiseed_k5_bigkern done"

# ============================================================
# Exp B: k=1 with k=5's kernels (rate-rescue test)
# ============================================================
$PYTHON -m $MODULE $COMMON \
    --dataset pk \
    --kernels 13 3 4 \
    --run_name "apr28_multiseed_k1_smallkern" \
    --description "Kernel-isolation B: k=1 (no polyphase, L=3000, pk) with k=5's small kernels (13, 3, 4) instead of the apr24/apr26 baseline (63, 15, 15). RF = 125 raw-ms (vs baseline 623 raw-ms — much shorter context, matching apr25 k=5 in raw-ms). Default channels (64/128/256). Tests whether full-resolution input rescues a small-kernel model. R² near +0.30+ → polyphase rate also matters and the architectural argument is incomplete. R² near +0.10 (matching apr25 k=5 +0.087) → kernel architecture alone explains the gap regardless of input rate."
echo ">>> apr28_multiseed_k1_smallkern done"

echo ""
echo "============================================================"
echo "  apr28 kernel-isolation sweep complete (2 runs)."
echo "  Reference comparison cells in SimpleCNN_DMT_regression_CV_seed:"
echo ""
echo "                      small kernels (13,3,4)   medium kernels (31,8,8)"
echo "    k=1 (pk):         apr28_k1_smallkern: ?    apr24_exp1: +0.299"
echo "    k=5 (pk_k5):      apr25_k5_rfmatched: +0.087  apr28_k5_bigkern: ?"
echo ""
echo "  Decision matrix:"
echo "    A=high, B=low    → kernels are the bottleneck (architectural)"
echo "    A=low,  B=high   → polyphase rate is the bottleneck"
echo "    A=high, B=high   → both contribute; gap is additive"
echo "    A=low,  B=low    → something else; revisit hypotheses"
echo ""
echo "  View: mlflow ui --backend-store-uri mlruns/"
echo "============================================================"
