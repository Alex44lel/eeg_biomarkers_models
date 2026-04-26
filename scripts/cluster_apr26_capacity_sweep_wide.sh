#!/bin/bash
#SBATCH --job-name=apr26_wide
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/vol/bitbucket/ac5725/eeg_biomarkers_models/logs/slurm-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alejandroch2011@gmail.com

# apr26 capacity-vs-polyphase sweep — WIDE row (4 of 8 runs).
# channels=(96, 192, 384) at every k.
# Param counts: ~1.68M @ k=1, ~933k @ k=2, ~490k @ k=5, ~349k @ k=10.
# Estimated wall time: ~8–10h on a30 (4 runs × ~120min, ~2.16× params vs default).
# 12h gives a safety margin.
#
# Runs independently of the narrow job — submit both, they share no state.

export PATH=/vol/bitbucket/${USER}/eeg_biomarkers_models/env/bin/:$PATH
source activate
. /vol/cuda/12.0.0/setup.sh
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}
export MLFLOW_TRACKING_URI=file:///vol/bitbucket/ac5725/eeg_biomarkers_models/mlruns

nvidia-smi

cd /vol/bitbucket/ac5725/eeg_biomarkers_models

bash src/models/reg_simpleCNN/shell_and_logs/run_apr26_capacity_sweep.sh wide
