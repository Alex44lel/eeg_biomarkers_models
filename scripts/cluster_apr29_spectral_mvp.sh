#!/bin/bash
#SBATCH --job-name=apr29_spectral
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/vol/bitbucket/ac5725/eeg_biomarkers_models/logs/slurm-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alejandroch2011@gmail.com

# apr29 SPECTRAL FRONTEND MVP — 6 multiseed runs on pk_k2 + --single_phase,
# all using --model spectral_cnn (STFT magnitude/power → 2D conv backbone).
#
# Full 2x3 grid over n_fft × variant:
#                       n_fft=256                          n_fft=512
#   default capacity    1. apr29_spectral_n256_default     2. apr29_spectral_n512_default
#   wide capacity       3. apr29_spectral_n256_wide        5. apr29_spectral_n512_wide
#   + baseline_sub      4. apr29_spectral_n256_baseline    6. apr29_spectral_n512_baseline
#
# See run_apr29_spectral_mvp.sh for full hypothesis, predictions, and decision
# rule. Compares to the apr28 single-phase champions:
#   - k2_default_singlephase  +0.3266 ± 0.031   (~436k params, SimpleCNN)
#   - k2_wide_singlephase     +0.3769 ± 0.037   (~932k params, SimpleCNN)
#
# Wall-time budget: each multiseed run is ~3–4 h on a30 (5 seeds × 8 LOSO
# folds; spectral input is small → fast per-epoch, but 5×8=40 LOSO trainings
# per multiseed). 6 runs × ~4h = ~24h. If wall-time is a constraint, split
# into two SLURM jobs by editing run_apr29_spectral_mvp.sh to comment out
# half the runs.

export PATH=/vol/bitbucket/${USER}/eeg_biomarkers_models/env/bin/:$PATH
source activate
. /vol/cuda/12.0.0/setup.sh
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}
export MLFLOW_TRACKING_URI=file:///vol/bitbucket/ac5725/eeg_biomarkers_models/mlruns

nvidia-smi

cd /vol/bitbucket/ac5725/eeg_biomarkers_models

bash src/models/reg_simpleCNN/shell_and_logs/run_apr29_spectral_mvp.sh
