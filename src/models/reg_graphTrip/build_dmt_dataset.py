"""
Build NPZ dataset for DMT plasma concentration regression.

For each subject, loads 3-second EEG trials from data_trialsmxm_3s.mat,
filters to [2, 15] minutes post-injection, and assigns labels from the
PK model's posterior mean y1 curve (plasma DMT in ng/mL).

Usage (run from project root):
    python -m src.models.reg_graphTrip.build_dmt_dataset \
        [--trace PATH] [--t-min 2] [--t-max 15]
"""

import argparse
import numpy as np
import scipy.io as sio
import arviz as az
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RECORDINGS_DIR = PROJECT_ROOT / "data" / "recordings"
DATA_DIR = PROJECT_ROOT / "data"

SUBJECT_FOLDERS = {
    "S01AS": "S01",
    "S02WT": "S02",
    "S03BS": "S03",
    "S04SG": "S04",
    "S05LM": "S05",
    "S06ET": "S06",
    "S07CS": "S07",
    "S08EK": "S08",
    "S09BB": "S09",
    "S10DL": "S10",
    "S11NW": "S11",
    "S12AI": "S12",
    "S13MBJ": "S13",
}

EXCLUDED_SUBJECTS = ["S03", "S04", "S08", "S09", "S11"]

# EEG channels to keep (exclude ECG, VEOG, EMGfront, EMGtemp at indices 31-34)
EEG_CHANNELS = list(range(31)) + [35]  # 32 EEG channels

# Channel labels for the 32 EEG channels (from the original 36-channel montage)
EEG_CHANNEL_LABELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz", "Oz",
    "FC1", "FC2", "CP1", "CP2", "FC5", "FC6", "CP5", "CP6",
    "TP9", "TP10", "POz", "FCz",
]


def compute_y1_curve(trace, subject_idx, t_array):
    """Compute the posterior-mean y1 curve (plasma ng/mL) for one subject.

    Evaluates eq. 16 per posterior sample and then averages — matches
    partial_pooling_model.compute_posterior_predictive_y1. Plugging in the
    posterior means of the parameters first would bias the curve (Jensen's
    inequality: y1 is non-linear in k0, k1, k2, y_init).
    """
    n_subjects = trace.posterior["y_init"].shape[-1]
    k0 = trace.posterior["k0"].values.reshape(-1, n_subjects)[:, subject_idx]       # (S,)
    k1 = trace.posterior["k1"].values.reshape(-1, n_subjects)[:, subject_idx]
    k2 = trace.posterior["k2"].values.reshape(-1, n_subjects)[:, subject_idx]
    y_init = trace.posterior["y_init"].values.reshape(-1, n_subjects)[:, subject_idx]

    s = k0 + k1 + k2
    p = k0 * k2
    disc = np.maximum(s**2 - 4 * p, 1e-12)
    alpha = (s - np.sqrt(disc)) / 2.0       # (S,)
    beta = (s + np.sqrt(disc)) / 2.0        # (S,)

    # Broadcast (S, 1) against (T,) to get per-sample curves of shape (S, T)
    a = alpha[:, None]
    b = beta[:, None]
    k2_ = k2[:, None]
    y0 = y_init[:, None]
    t = t_array[None, :]

    y1_samples = (y0 / (b - a)) * ((k2_ - a) * np.exp(-a * t) - (k2_ - b) * np.exp(-b * t))
    return y1_samples.mean(axis=0)


def process_subject(folder_name, subj_id, trace, pk_subject_idx, t_min, t_max):
    """Load EEG trials for one subject, filter by time, assign PK labels."""
    mat_path = RECORDINGS_DIR / folder_name / "DMT" / "data_trialsmxm_3s.mat"
    if not mat_path.exists():
        print(f"  Skipping {subj_id}: {mat_path} not found")
        return None

    d = sio.loadmat(str(mat_path), squeeze_me=True)
    cells = d["data_trialsmxm_3s"]

    # Count baseline trials to determine injection time
    n_baseline = len(cells[0][()]["trial"])
    injection_time = n_baseline * 3.0 / 60.0

    # Collect all trials with their times (relative to injection)
    eeg_trials = []
    trial_times = []
    trial_idx = 0

    for cell_idx in range(len(cells)):
        cell = cells[cell_idx][()]
        trials = cell["trial"]
        for j in range(len(trials)):
            t_since_inj = (trial_idx + 0.5) * 3.0 / 60.0 - injection_time
            if t_min <= t_since_inj <= t_max:
                trial_data = trials[j][EEG_CHANNELS, :]  # (32, 3000)
                eeg_trials.append(trial_data)
                trial_times.append(t_since_inj)
            trial_idx += 1

    if not eeg_trials:
        print(f"  {subj_id}: no trials in [{t_min}, {t_max}] min")
        return None

    eeg_array = np.stack(eeg_trials)  # (N_trials, 32, 3000)
    times_array = np.array(trial_times)

    # Get plasma labels from PK model
    labels = compute_y1_curve(trace, pk_subject_idx, times_array)

    print(f"  {subj_id}: {len(eeg_trials)} trials, t=[{times_array.min():.2f}, {times_array.max():.2f}] min, "
          f"plasma=[{labels.min():.1f}, {labels.max():.1f}] ng/mL")

    return {
        "eeg": eeg_array,
        "labels": labels,
        "times": times_array,
        "subject": np.array([subj_id] * len(eeg_trials)),
    }


def main():
    parser = argparse.ArgumentParser(description="Build DMT regression dataset")
    parser.add_argument("--trace", type=str,
                        default="results/paper_model/only_plasma/partial_pooling_trace.nc",
                        help="Path to PK model trace")
    parser.add_argument("--t-min", type=float, default=2.0,
                        help="Start time in minutes post-injection (default: 2)")
    parser.add_argument("--t-max", type=float, default=15.0,
                        help="End time in minutes post-injection (default: 15)")
    args = parser.parse_args()

    # Load PK model trace and get subject list
    print(f"Loading PK trace from {args.trace}...")
    trace = az.from_netcdf(args.trace)

    # Get the subject names from prepare_data (same order as the trace)
    from src.paper_model.prepare_data import load_and_prepare
    pk_data = load_and_prepare()
    pk_subjects = pk_data["subject_names"]
    pk_subj_to_idx = {s: i for i, s in enumerate(pk_subjects)}
    print(f"PK model subjects: {pk_subjects}")

    # Process each subject
    all_eeg = []
    all_labels = []
    all_times = []
    all_subjects = []

    for folder_name, subj_id in sorted(SUBJECT_FOLDERS.items()):
        if subj_id in EXCLUDED_SUBJECTS:
            continue
        if subj_id not in pk_subj_to_idx:
            print(f"  Skipping {subj_id}: not in PK model")
            continue

        print(f"Processing {subj_id}...")
        result = process_subject(
            folder_name, subj_id, trace,
            pk_subj_to_idx[subj_id], args.t_min, args.t_max,
        )
        if result is not None:
            all_eeg.append(result["eeg"])
            all_labels.append(result["labels"])
            all_times.append(result["times"])
            all_subjects.append(result["subject"])

    # Concatenate and save
    eeg_data = np.concatenate(all_eeg, axis=0)     # (N, 32, 3000)
    labels = np.concatenate(all_labels, axis=0)     # (N,) ng/mL
    times = np.concatenate(all_times, axis=0)       # (N,) minutes
    subjects = np.concatenate(all_subjects, axis=0)  # (N,) subject IDs

    out_path = DATA_DIR / "eeg_dmt_regression.npz"
    np.savez(
        out_path,
        eeg_data=eeg_data,
        labels=labels,
        times=times,
        subjects=subjects,
        channel_labels=np.array(EEG_CHANNEL_LABELS),
    )

    print(f"\nSaved to {out_path}")
    print(f"  Total trials: {len(labels)}")
    print(f"  Subjects: {sorted(set(subjects))}")
    print(f"  EEG shape: {eeg_data.shape}")
    print(f"  Label range: [{labels.min():.1f}, {labels.max():.1f}] ng/mL")
    print(f"  Time range: [{times.min():.2f}, {times.max():.2f}] min")


if __name__ == "__main__":
    main()
