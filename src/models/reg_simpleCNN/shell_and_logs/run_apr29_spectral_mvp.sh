#!/bin/bash
# apr29 SPECTRAL FRONTEND MVP — 6 multiseed runs that all share:
#   - dataset eeg_dmt_regression_k2 (--dataset pk_k2)
#   - --single_phase (k_idx==0 only; same parent windows as k=1, decimated by 2)
#   - --model spectral_cnn (STFT magnitude/power → 2D conv backbone)
#   - run55 hyperparams: lr=5e-4 bs=64 dropout=0.3 wd=1e-4 smoothl1 huber_beta=10
#     EMA 0.999, patience=40, max_epochs=300, 5 seeds × 8 LOSO folds.
#
# DESIGN INTENT (see misc/study_notes/findings.md "What's left to try"):
#   The 1D-conv stack has to learn the band decomposition from scratch. A fixed
#   STFT frontend exposes (channels × freq × time) directly, then a small 2D
#   CNN learns over that. Most plasma-correlated EEG content lives in 4–50 Hz;
#   the STFT decomposition matches that prior. The MVP keeps the frontend
#   parameter-free and tests whether it beats — or even ties — the apr28
#   k=2_wide_singlephase champion (+0.377 ± 0.037).
#
# THE 6 RUNS (each ~1–2 h on a30) — full 2×3 grid over n_fft × variant:
#                       n_fft=256                         n_fft=512
#   default capacity    1. apr29_spectral_n256_default    2. apr29_spectral_n512_default
#   wide capacity       3. apr29_spectral_n256_wide       5. apr29_spectral_n512_wide
#   + baseline_sub      4. apr29_spectral_n256_baseline   6. apr29_spectral_n512_baseline
#
#   1. apr29_spectral_n256_default
#        n_fft=256 (bin width 1.95 Hz at fs=500), hop=32, channels 64/128/256.
#        ~506k params, F=52 bins (0–100 Hz), T=47 frames. Primary "is this
#        useful" run; compare to k=2_default_singlephase (+0.327 ± 0.031) at
#        matched param count and to k=2_wide_singlephase (+0.377 ± 0.037) at
#        matched headline metric.
#   2. apr29_spectral_n512_default
#        n_fft=512 (bin width 0.98 Hz), hop=64, channels 64/128/256.
#        ~506k params, F=103 bins, T=24 frames. Tests whether finer freq
#        resolution helps (alpha/beta edges aren't on bin boundaries with
#        n_fft=256). Same param count as run 1 — clean ablation on frontend.
#   3. apr29_spectral_n256_wide
#        n_fft=256, hop=32, channels 96/192/384.
#        ~1.1M params. Capacity ablation at the favored n_fft. If wide
#        underperforms run 1 by a clear margin, frontend is the bottleneck;
#        if it beats run 1, scaling matters and a deeper sweep is warranted.
#   4. apr29_spectral_n256_baseline_sub
#        Same as run 1 + --baseline_subtraction. Uses the baseline-aware
#        npz (pk_k2_with_baseline) which has the `is_baseline` column
#        of pre-injection trials needed by compute_mu_s_table. This is the
#        "subject-conditioned" probe — does subtracting each subject's
#        average pre-injection feature (scaled by a learnable λ) close
#        the per-subject heterogeneity gap (S07/S12 vs S01/S05/S13)?
#   5. apr29_spectral_n512_wide
#        n_fft=512, hop=64, channels 96/192/384. ~1.1M params. Capacity
#        ablation at the finer-freq-resolution frontend; pairs with run 3
#        to disentangle frontend from capacity.
#   6. apr29_spectral_n512_baseline_sub
#        Same as run 2 + --baseline_subtraction. Pairs with run 4 to test
#        whether the subject-calibration win (if any) depends on the
#        spectral resolution.
#
# DECISION RULE (after this sweep):
#   - Any spectral run >= +0.36 mean and std <= 0.04 → spectral frontend is
#     promising; queue alt-seed replication + n_fft refinement.
#   - All runs noticeably below +0.34 → spectral frontend is not better than
#     the time-domain CNN at this design point; revisit the frontend (sinc
#     conv / wavelets) or move on to subject conditioning (run 4 standalone).
#   - Run 4 (baseline_subtraction) clearly above runs 1–3 → the win comes
#     from subject calibration, not the spectral frontend. Promote that
#     direction independently of the frontend choice.
#
# Run from project root:
#   bash src/models/reg_simpleCNN/shell_and_logs/run_apr29_spectral_mvp.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python}"
MODULE="src.models.reg_simpleCNN.multiseed_cv"
COMMON="--lr 5e-4 --batch_size 64 --epochs 300 --patience 40 --dropout 0.3
        --weight_decay 1e-4 --loss smoothl1 --huber_beta 10.0
        --model spectral_cnn --f_max 100.0 --spectral_power 2.0
        --single_phase
        --experiment_name SimpleCNN_DMT_regression_CV_seed"

echo "============================================================"
echo "  apr29 SPECTRAL FRONTEND MVP — 6 multiseed runs"
echo "  All on pk_k2 + --single_phase, --model spectral_cnn"
echo "============================================================"

# --- Run 1: STFT n_fft=256, default capacity ---
$PYTHON -m $MODULE $COMMON \
    --dataset pk_k2 \
    --n_fft 256 --hop_length 32 \
    --channels 64 128 256 \
    --run_name "apr29_spectral_n256_default" \
    --description "Spectral MVP: STFT power (n_fft=256, hop=32, fs=500 → F=52 bins to 100 Hz, T=47 frames) → 2D conv backbone (channels 64/128/256, ~506k params). Dataset pk_k2 + --single_phase (k_idx==0 only, parity with apr28 single-phase numbers). Primary 'is this useful?' run; compare to apr28_multiseed_k2_default_singlephase (+0.327 ± 0.031) at matched param count."
echo ">>> apr29_spectral_n256_default done"

# --- Run 2: STFT n_fft=512, default capacity (alt freq resolution) ---
$PYTHON -m $MODULE $COMMON \
    --dataset pk_k2 \
    --n_fft 512 --hop_length 64 \
    --channels 64 128 256 \
    --run_name "apr29_spectral_n512_default" \
    --description "Spectral MVP: STFT power (n_fft=512, hop=64 → F=103 bins to 100 Hz, T=24 frames) → 2D conv backbone (64/128/256, ~506k params). Doubles freq resolution (0.98 Hz/bin vs 1.95 Hz at n_fft=256), halves time frames. Same dataset and capacity as run 1 — clean frontend ablation."
echo ">>> apr29_spectral_n512_default done"

# --- Run 3: STFT n_fft=256, wide capacity (capacity ablation) ---
$PYTHON -m $MODULE $COMMON \
    --dataset pk_k2 \
    --n_fft 256 --hop_length 32 \
    --channels 96 192 384 \
    --run_name "apr29_spectral_n256_wide" \
    --description "Spectral MVP capacity ablation: STFT power (n_fft=256, hop=32) → 2D conv backbone (channels 96/192/384, ~1.1M params). Tests whether scaling the spectral backbone matters at the favored n_fft. Compares to apr28_multiseed_k2_wide_singlephase (+0.377 ± 0.037) at matched param count."
echo ">>> apr29_spectral_n256_wide done"

# --- Run 4: STFT n_fft=256, default capacity, --baseline_subtraction ---
# Uses the baseline-aware k=2 build (has is_baseline column) needed by the
# compute_mu_s_table refresh step in train_cv.run_fold.
$PYTHON -m $MODULE $COMMON \
    --dataset pk_k2_with_baseline \
    --n_fft 256 --hop_length 32 \
    --channels 64 128 256 \
    --baseline_subtraction \
    --run_name "apr29_spectral_n256_baseline_sub" \
    --description "Spectral MVP + linear subject adaptation: STFT power (n_fft=256, hop=32) → 2D conv backbone (64/128/256, ~506k params) + --baseline_subtraction (subtract λ × per-subject mean pre-injection feature before regressor). Dataset pk_k2_with_baseline (has is_baseline column) + --single_phase. Tests whether subject calibration closes the per-subject heterogeneity gap (S07/S12 vs S01/S05/S13). λ is a single learnable scalar; init=1.0."
echo ">>> apr29_spectral_n256_baseline_sub done"

# --- Run 5: STFT n_fft=512, wide capacity ---
$PYTHON -m $MODULE $COMMON \
    --dataset pk_k2 \
    --n_fft 512 --hop_length 64 \
    --channels 96 192 384 \
    --run_name "apr29_spectral_n512_wide" \
    --description "Spectral MVP capacity ablation at finer freq resolution: STFT power (n_fft=512, hop=64 → F=103 bins, T=24 frames) → 2D conv backbone (channels 96/192/384, ~1.1M params). Pairs with apr29_spectral_n256_wide to disentangle frontend resolution from capacity. Same total param count as n256_wide → any divergence isolates the n_fft effect at the wide capacity tier."
echo ">>> apr29_spectral_n512_wide done"

# --- Run 6: STFT n_fft=512, default capacity, --baseline_subtraction ---
$PYTHON -m $MODULE $COMMON \
    --dataset pk_k2_with_baseline \
    --n_fft 512 --hop_length 64 \
    --channels 64 128 256 \
    --baseline_subtraction \
    --run_name "apr29_spectral_n512_baseline_sub" \
    --description "Spectral MVP + linear subject adaptation at finer freq resolution: STFT power (n_fft=512, hop=64) → 2D conv backbone (64/128/256, ~506k params) + --baseline_subtraction. Pairs with apr29_spectral_n256_baseline_sub to test whether the subject-calibration effect (if any) depends on n_fft. Dataset pk_k2_with_baseline + --single_phase. λ is a single learnable scalar; init=1.0."
echo ">>> apr29_spectral_n512_baseline_sub done"

echo ""
echo "============================================================"
echo "  apr29 spectral MVP sweep complete (6 runs)."
echo "  Reference comparisons in SimpleCNN_DMT_regression_CV_seed:"
echo "    apr28_multiseed_k2_default_singlephase  +0.3266 ± 0.031  (sim CNN, ~436k)"
echo "    apr28_multiseed_k2_wide_singlephase     +0.3769 ± 0.037  (sim CNN, ~932k)"
echo "    apr29_spectral_n256_default             ?  (~506k, frontend MVP)"
echo "    apr29_spectral_n512_default             ?  (~506k, finer freq)"
echo "    apr29_spectral_n256_wide                ?  (~1.1M, capacity)"
echo "    apr29_spectral_n256_baseline_sub        ?  (~506k, +baseline_sub)"
echo "    apr29_spectral_n512_wide                ?  (~1.1M, finer freq + capacity)"
echo "    apr29_spectral_n512_baseline_sub        ?  (~506k, finer freq + baseline_sub)"
echo "  View: mlflow ui --backend-store-uri mlruns/"
echo "============================================================"
