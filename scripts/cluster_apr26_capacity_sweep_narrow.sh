#!/bin/bash
#SBATCH --job-name=apr26_narrow
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=/vol/bitbucket/ac5725/eeg_biomarkers_models/logs/slurm-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alejandroch2011@gmail.com

# apr26 capacity-vs-polyphase sweep — NARROW row (4 of 8 runs).
# channels=(32, 64, 128) at every k.
# Param counts: ~230k @ k=1, ~125k @ k=2, ~64k @ k=5, ~50k @ k=10.
# Estimated wall time: ~5–6h on a30 (4 runs × ~75min, smaller models faster).
# 8h gives a safety margin.

export PATH=/vol/bitbucket/${USER}/eeg_biomarkers_models/env/bin/:$PATH
source activate
. /vol/cuda/12.0.0/setup.sh
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}
export MLFLOW_TRACKING_URI=file:///vol/bitbucket/ac5725/eeg_biomarkers_models/mlruns

nvidia-smi

cd /vol/bitbucket/ac5725/eeg_biomarkers_models

bash src/models/reg_simpleCNN/shell_and_logs/run_apr26_capacity_sweep.sh narrow
