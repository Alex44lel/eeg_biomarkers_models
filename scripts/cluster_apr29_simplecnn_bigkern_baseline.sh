#!/bin/bash
#SBATCH --job-name=apr29_smplx
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/vol/bitbucket/ac5725/eeg_biomarkers_models/logs/slurm-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alejandroch2011@gmail.com

# apr29 SIMPLECNN extras — 3 multiseed runs on pk_k2 + --single_phase using
# --model simplecnn (the 1D conv stack from model.py).
#
# Runs:
#   1. apr29_multiseed_k2_xwide
#        kernels (31, 8, 8)    channels (128, 256, 512)  RF=622 raw-ms
#        ~1.61M params  no baseline_sub
#   2. apr29_multiseed_k2_wide_baseline_sub
#        kernels (31, 8, 8)    channels (96, 192, 384)   RF=622 raw-ms
#        ~932k  params  --baseline_subtraction (uses pk_k2_with_baseline)
#   3. apr29_multiseed_k2_default_baseline_sub
#        kernels (31, 8, 8)    channels (64, 128, 256)   RF=622 raw-ms
#        ~436k  params  --baseline_subtraction (uses pk_k2_with_baseline)
#
# See run_apr29_simplecnn_bigkern_baseline.sh for the full hypothesis,
# decision rule, and reference comparisons.
#
# Wall-time budget: each multiseed run is ~3–5 h on a100 (5 seeds × 8 LOSO
# folds; baseline_subtraction adds a per-epoch mu_s recomputation that runs
# in eval mode over each subject's pre-injection trials — small overhead).
# 3 runs × ~5h = ~15h. Matches apr28_kernel_isolation's 16h budget. The
# xwide run (~1.61M params) is the slowest of the three.

export PATH=/vol/bitbucket/${USER}/eeg_biomarkers_models/env/bin/:$PATH
source activate
. /vol/cuda/12.0.0/setup.sh
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}
export MLFLOW_TRACKING_URI=file:///vol/bitbucket/ac5725/eeg_biomarkers_models/mlruns

nvidia-smi

cd /vol/bitbucket/ac5725/eeg_biomarkers_models

bash src/models/reg_simpleCNN/shell_and_logs/run_apr29_simplecnn_bigkern_baseline.sh
