"""
Extract EEG trials from data_trialsmxm_3s.mat files into a single .npy dataset
for PyTorch training.

Includes only DMT sessions, groups 0-5, excluding subjects S03, S09, S11.
Group 0 (pre-injection) gets label=0, groups 1-5 get label=1.
"""

import numpy as np
import scipy.io as sio
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECORDINGS_DIR = PROJECT_ROOT / "data" / "recordings"
OUTPUT_DIR = PROJECT_ROOT / "data"

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

EXCLUDED_SUBJECTS = ["S03", "S09", "S11"]
GROUPS_TO_INCLUDE = list(range(6))  # 0, 1, 2, 3, 4, 5
FS = 1000  # sampling rate in Hz
TRIAL_DURATION = 3.0  # seconds


def extract_dataset():
    all_trials = []
    all_meta = []

    for folder_name, subj_id in sorted(SUBJECT_FOLDERS.items()):
        if subj_id in EXCLUDED_SUBJECTS:
            print(f"Skipping excluded subject {subj_id}")
            continue

        mat_path = RECORDINGS_DIR / folder_name / "DMT" / "data_trialsmxm_3s.mat"
        if not mat_path.exists():
            print(f"Skipping {subj_id}: file not found")
            continue

        print(f"Loading {subj_id} ({folder_name})...")
        d = sio.loadmat(str(mat_path), squeeze_me=True)
        cells = d["data_trialsmxm_3s"]

        n_cells = len(cells)
        print(f"  Total cells/groups in file: {n_cells}")

        for group_idx in GROUPS_TO_INCLUDE:
            if group_idx >= n_cells:
                print(f"  Warning: group {group_idx} not available (only {n_cells} groups)")
                continue

            cell = cells[group_idx][()]
            trials = cell["trial"]
            sampleinfo = cell["sampleinfo"]
            channel_labels = cell["label"]
            label = 0 if group_idx == 0 else 1

            n_trials = len(trials)
            print(f"  Group {group_idx}: {n_trials} trials, label={label}")

            for trial_i in range(n_trials):
                eeg = trials[trial_i]  # shape: (36, 3000)

                # Get timing from sampleinfo
                if sampleinfo.ndim == 2:
                    start_sample = int(sampleinfo[trial_i, 0])
                    end_sample = int(sampleinfo[trial_i, 1])
                elif sampleinfo.ndim == 1 and n_trials == 1:
                    start_sample = int(sampleinfo[0])
                    end_sample = int(sampleinfo[1])
                else:
                    start_sample = 0
                    end_sample = eeg.shape[1]

                start_time = start_sample / FS
                end_time = end_sample / FS

                all_trials.append(eeg)
                all_meta.append({
                    "subject": subj_id,
                    "group": group_idx,
                    "label": label,
                    "trial_index": trial_i,
                    "start_time_s": start_time,
                    "end_time_s": end_time,
                    "n_channels": eeg.shape[0],
                    "n_samples": eeg.shape[1],
                    "fs": FS,
                })

    # Balance: per subject, trim group 0 from the end to match groups 1-5 count
    balanced_trials = []
    balanced_meta = []
    for subj in sorted(set(m["subject"] for m in all_meta)):
        subj_indices = [i for i, m in enumerate(all_meta) if m["subject"] == subj]
        n_post = sum(1 for i in subj_indices if all_meta[i]["group"] > 0)
        # Collect group 0 indices (already in order), keep only the last n_post
        g0_indices = [i for i in subj_indices if all_meta[i]["group"] == 0]
        gpost_indices = [i for i in subj_indices if all_meta[i]["group"] > 0]
        n_drop = len(g0_indices) - n_post
        if n_drop > 0:
            # Drop from the end of group 0
            g0_indices = g0_indices[:n_post]
            print(f"  {subj}: balanced group 0 from {n_post + n_drop} to {n_post} trials (dropped {n_drop} from end)")
        keep_indices = g0_indices + gpost_indices
        for i in keep_indices:
            balanced_trials.append(all_trials[i])
            balanced_meta.append(all_meta[i])

    all_trials = balanced_trials
    all_meta = balanced_meta

    # Stack all trials into a single array: (N_trials, 36, 3000)
    eeg_data = np.stack(all_trials, axis=0)

    # Build structured metadata array
    n_total = len(all_meta)
    subjects = np.array([m["subject"] for m in all_meta], dtype="U10")
    groups = np.array([m["group"] for m in all_meta], dtype=np.int32)
    labels = np.array([m["label"] for m in all_meta], dtype=np.int32)
    trial_indices = np.array([m["trial_index"] for m in all_meta], dtype=np.int32)
    start_times = np.array([m["start_time_s"] for m in all_meta], dtype=np.float64)
    end_times = np.array([m["end_time_s"] for m in all_meta], dtype=np.float64)

    # Get channel labels from the last loaded file
    ch_labels = np.array([str(ch) for ch in channel_labels], dtype="U20")

    print(f"\nDataset summary:")
    print(f"  Total trials: {n_total}")
    print(f"  EEG shape: {eeg_data.shape}")
    print(f"  Subjects: {sorted(set(subjects))}")
    print(f"  Label 0 (pre-injection): {np.sum(labels == 0)} trials")
    print(f"  Label 1 (post-injection): {np.sum(labels == 1)} trials")

    for g in GROUPS_TO_INCLUDE:
        mask = groups == g
        print(f"  Group {g}: {np.sum(mask)} trials")

    # Save as a single .npy file using np.savez
    out_path = OUTPUT_DIR / "eeg_dmt_dataset.npz"
    np.savez(
        out_path,
        eeg=eeg_data,
        subjects=subjects,
        groups=groups,
        labels=labels,
        trial_indices=trial_indices,
        start_times=start_times,
        end_times=end_times,
        channel_labels=ch_labels,
        fs=np.array([FS]),
    )
    print(f"\nSaved dataset to {out_path}")
    print(f"  File size: {out_path.stat().st_size / 1e6:.1f} MB")

    return out_path


if __name__ == "__main__":
    extract_dataset()
