#!/bin/bash
# apr27 ANTI-ALIASED polyphase sweep — 3 multiseed LOSO CV runs at k=2, 5, 10
# on the *filtered* polyphase datasets (pk_k2_filt / pk_k5_filt / pk_k10_filt).
#
# HYPOTHESIS UNDER TEST: the apr25 collapse at k=5 (+0.087) and k=10 (-0.038)
# was driven by aliasing in the raw stride-K split (build_downsampled_dataset.py
# had no anti-alias filter — see misc/study_notes/findings.md). The new
# pk_kN_filt builds apply a 31-tap zero-phase FIR low-pass at the new Nyquist
# before taking each phase, which removes the aliasing.
#
# Predictions for k=2: tiny change (most EEG content already lives below
# k=2's 250 Hz Nyquist; aliasing was always small there). This is the control.
# Predictions for k=5: substantial recovery if aliasing was the culprit;
# unchanged if DMT signal genuinely needs >100 Hz content.
# Predictions for k=10: probably still bad — the alpha/beta bands at 8-30 Hz
# are clean (Nyquist 50 Hz) but high beta and all gamma are gone irreversibly.
#
# Hyperparams: lr=5e-4, batch_size=64, dropout=0.3, weight_decay=1e-4,
# smoothl1 huber_beta=10, EMA 0.999, early-stop on val_r2 (mean-of-K phase R²),
# patience=40, max_epochs=300. Seeds default = (42, 123, 7, 2024, 0).
#
# Capacity choice per run:
#   k=2  → default channels 64/128/256 (~436k params). Direct apples-to-apples
#          with apr25 k=2 unfiltered baseline.
#   k=5  → WIDE channels 96/192/384 (~490k params). Most charitable test:
#          combines anti-aliasing fix with the apr26 capacity bump.
#          Compares cleanly to apr26_multiseed_k5_wide (+0.132 unfiltered).
#   k=10 → WIDE channels 96/192/384 (~349k params). Same logic as k=5.
#          Compares cleanly to apr26_multiseed_k10_wide (-0.068 unfiltered).
#
# RF in raw-ms is matched to k=1 baseline (~623 ms) by holding kernels at
# the same per-k values as apr25.
#
# Run from project root:
#   bash src/models/reg_simpleCNN/shell_and_logs/run_apr27_polyphase_filt.sh

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
echo "  apr27 ANTI-ALIASED polyphase sweep — k=2, 5, 10 (3 runs)"
echo "  Filtered builds (pk_kN_filt) vs apr25 unfiltered baseline"
echo "============================================================"

# --- k=2 (control: aliasing was already small at Nyquist=250 Hz) ---
$PYTHON -m $MODULE $COMMON \
    --dataset pk_k2_filt \
    --kernels 31 8 8 \
    --run_name "apr27_multiseed_k2_filt" \
    --description "Anti-aliased polyphase k=2 (L=1500, pk_k2_filt). 31-tap FIR Hamming low-pass at Nyquist=250 Hz, applied with filtfilt for zero phase, BEFORE taking each phase. RF=311 sub-samples = 622 raw-ms. Default channels (64/128/256), ~436k params. CONTROL: at k=2 there is little EEG content above 250 Hz, so the unfiltered apr25 result (+0.345) and this filtered run should be ~equal. Significant divergence would be a red flag."
echo ">>> apr27_multiseed_k2_filt done"

# --- k=5 wide (key experiment: anti-aliasing + extra capacity) ---
$PYTHON -m $MODULE $COMMON \
    --dataset pk_k5_filt \
    --kernels 13 3 4 \
    --channels 96 192 384 \
    --run_name "apr27_multiseed_k5_filt_wide" \
    --description "Anti-aliased polyphase k=5 (L=600, pk_k5_filt) at WIDE channels (96/192/384, ~490k params). 31-tap FIR Hamming low-pass at Nyquist=100 Hz, applied with filtfilt for zero phase, BEFORE taking each phase. RF=125 sub-samples = 625 raw-ms. KEY EXPERIMENT: combines two fixes against the apr25 k=5 collapse (+0.087) — (a) anti-aliasing, (b) restored capacity. Cleanest comparison is to apr26_multiseed_k5_wide which had wide channels but unfiltered data (+0.132). Recovery toward k=2 territory (+0.34) → aliasing was the dominant cause; staying near +0.13 → DMT signal genuinely needs content above 100 Hz."
echo ">>> apr27_multiseed_k5_filt_wide done"

# --- k=10 wide (most aggressive — most charitable test) ---
$PYTHON -m $MODULE $COMMON \
    --dataset pk_k10_filt \
    --kernels 15 3 2 \
    --channels 96 192 384 \
    --run_name "apr27_multiseed_k10_filt_wide" \
    --description "Anti-aliased polyphase k=10 (L=300, pk_k10_filt) at WIDE channels (96/192/384, ~349k params). 31-tap FIR Hamming low-pass at Nyquist=50 Hz, applied with filtfilt for zero phase, BEFORE taking each phase. RF=63 sub-samples = 630 raw-ms. Compares to apr26_multiseed_k10_wide unfiltered (-0.068). EXPECTED: alpha (8-13 Hz) and beta (13-30 Hz) are now clean of aliased gamma contamination, but high beta (30-50 Hz) and all gamma are gone irrecoverably. Modest recovery at best — even with both fixes, 100 Hz effective rate is below the floor for this signal."
echo ">>> apr27_multiseed_k10_filt_wide done"

echo ""
echo "============================================================"
echo "  apr27 anti-aliased sweep complete (3 runs)."
echo "  Reference comparisons in SimpleCNN_DMT_regression_CV_seed:"
echo "    k=2  default:  apr25_multiseed_k2_rfmatched  +0.345"
echo "                   apr27_multiseed_k2_filt       ?"
echo "    k=5  wide:     apr26_multiseed_k5_wide       +0.132"
echo "                   apr27_multiseed_k5_filt_wide  ?  (key)"
echo "    k=10 wide:     apr26_multiseed_k10_wide      -0.068"
echo "                   apr27_multiseed_k10_filt_wide ?"
echo "  View: mlflow ui --backend-store-uri mlruns/"
echo "============================================================"
