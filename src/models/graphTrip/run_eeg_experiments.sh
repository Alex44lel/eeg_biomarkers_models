#!/bin/bash
# Run 12 experiments with different hyperparameter configurations for graphTrip EEG.
# Usage: bash run_eeg_experiments.sh

set -e
cd "$(dirname "$0")"

PYTHON=python3

echo "================================================"
echo "  graphTrip EEG Experiment Suite (12 runs)"
echo "================================================"

# --- Baseline experiments (no coords vs coords) ---

# Experiment 1: Baseline, no coords
$PYTHON train_eeg.py --lr 1e-3 --batch_size 16 --dropout 0.1 --weight_decay 1e-4 \
    --cls_weight 1.0 --patience 15 --val_subjects S12 S13 \
    --run_name "exp01_baseline_nocoords"

# Experiment 2: Baseline, with coords
$PYTHON train_eeg.py --lr 1e-3 --batch_size 16 --dropout 0.1 --weight_decay 1e-4 \
    --cls_weight 1.0 --patience 15 --val_subjects S12 S13 --use_coords \
    --run_name "exp02_baseline_coords"

# --- Learning rate sweep ---

# Experiment 3: Lower LR, no coords
$PYTHON train_eeg.py --lr 1e-4 --batch_size 16 --dropout 0.1 --weight_decay 1e-4 \
    --cls_weight 1.0 --patience 20 --val_subjects S12 S13 \
    --run_name "exp03_lr1e-4_nocoords"

# Experiment 4: Lower LR, with coords
$PYTHON train_eeg.py --lr 1e-4 --batch_size 16 --dropout 0.1 --weight_decay 1e-4 \
    --cls_weight 1.0 --patience 20 --val_subjects S12 S13 --use_coords \
    --run_name "exp04_lr1e-4_coords"

# Experiment 5: Higher LR
$PYTHON train_eeg.py --lr 5e-3 --batch_size 16 --dropout 0.1 --weight_decay 1e-4 \
    --cls_weight 1.0 --patience 15 --val_subjects S12 S13 \
    --run_name "exp05_lr5e-3"

# --- Classification weight sweep ---

# Experiment 6: Higher cls weight (prioritize classification)
$PYTHON train_eeg.py --lr 1e-3 --batch_size 16 --dropout 0.1 --weight_decay 1e-4 \
    --cls_weight 5.0 --patience 15 --val_subjects S12 S13 \
    --run_name "exp06_cw5.0"

# Experiment 7: Lower cls weight (prioritize reconstruction)
$PYTHON train_eeg.py --lr 1e-3 --batch_size 16 --dropout 0.1 --weight_decay 1e-4 \
    --cls_weight 0.1 --patience 15 --val_subjects S12 S13 \
    --run_name "exp07_cw0.1"

# --- Regularization sweep ---

# Experiment 8: Higher dropout
$PYTHON train_eeg.py --lr 1e-3 --batch_size 16 --dropout 0.3 --weight_decay 1e-4 \
    --cls_weight 1.0 --patience 15 --val_subjects S12 S13 \
    --run_name "exp08_dropout0.3"

# Experiment 9: No dropout
$PYTHON train_eeg.py --lr 1e-3 --batch_size 16 --dropout 0.0 --weight_decay 1e-4 \
    --cls_weight 1.0 --patience 15 --val_subjects S12 S13 \
    --run_name "exp09_no_dropout"

# --- Architecture sweep ---

# Experiment 10: Larger model
$PYTHON train_eeg.py --lr 5e-4 --batch_size 16 --dropout 0.1 --weight_decay 1e-4 \
    --cls_weight 1.0 --hidden_dim 64 --latent_dim 64 --patience 20 \
    --val_subjects S12 S13 --use_coords \
    --run_name "exp10_large_coords"

# --- Different validation splits ---

# Experiment 11: Different val subjects
$PYTHON train_eeg.py --lr 1e-3 --batch_size 16 --dropout 0.1 --weight_decay 1e-4 \
    --cls_weight 1.0 --patience 15 --val_subjects S01 S06 --use_coords \
    --run_name "exp11_val_S01_S06_coords"

# Experiment 12: Tuned combo
$PYTHON train_eeg.py --lr 5e-4 --batch_size 8 --dropout 0.2 --weight_decay 5e-4 \
    --cls_weight 2.0 --hidden_dim 64 --latent_dim 32 --patience 25 \
    --val_subjects S12 S13 --use_coords \
    --run_name "exp12_tuned_combo"

echo ""
echo "================================================"
echo "  All 12 experiments completed."
echo "  View results: mlflow ui --backend-store-uri mlruns/"
echo "================================================"
