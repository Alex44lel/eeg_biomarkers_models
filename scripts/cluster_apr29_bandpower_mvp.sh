#!/bin/bash
#SBATCH --job-name=apr29_bandpower
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/vol/bitbucket/ac5725/eeg_biomarkers_models/logs/slurm-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alejandroch2011@gmail.com

# apr29 BANDPOWER MVP — 6 multiseed runs on pk_k2_with_baseline + --single_phase,
# all using --model bandpower_linear (paper-style: Welch band-power features,
# Linear / small MLP head, optional baseline subtraction).
#
# Full 3x2 grid over hidden width × baseline subtraction:
#                       --baseline_subtraction off              on
#   linear              1. apr29_bandpower_linear_default       2. apr29_bandpower_linear_baseline
#   mlp64               3. apr29_bandpower_mlp64_default        4. apr29_bandpower_mlp64_baseline
#   mlp128              5. apr29_bandpower_mlp128_default       6. apr29_bandpower_mlp128_baseline
#
# Pairs with cluster_apr29_spectral_mvp.sh (STFT + 2D conv backbone). Together
# the two sweeps test the two ways spectral information can be exposed:
#   - Hand-crafted band powers fed to a tiny head (this script)        — paper-style
#   - Full STFT spectrogram fed to a 2D CNN  (spectral_mvp script)     — learned
#
# Wall-time budget: bandpower trials are TINY (192 features per trial → linear
# is < 1 ms / batch). Each multiseed run is ~30–60 min on a30 (5 seeds × 8
# LOSO folds × ~50 epochs of fast training). 6 runs × ~1 h = ~6–10 h total.
#
# See run_apr29_bandpower_mvp.sh for full hypothesis, predictions, and
# decision rule.

export PATH=/vol/bitbucket/${USER}/eeg_biomarkers_models/env/bin/:$PATH
source activate
. /vol/cuda/12.0.0/setup.sh
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}
export MLFLOW_TRACKING_URI=file:///vol/bitbucket/ac5725/eeg_biomarkers_models/mlruns

nvidia-smi

cd /vol/bitbucket/ac5725/eeg_biomarkers_models

bash src/models/reg_simpleCNN/shell_and_logs/run_apr29_bandpower_mvp.sh
