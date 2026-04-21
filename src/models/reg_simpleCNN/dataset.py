"""
PyTorch Dataset for EEG DMT plasma-concentration regression.

Loads the entire .npz dataset into RAM for fast training.
Supports subject-level train/validation splits.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "eeg_dmt_regression.npz"

# Subjects with PK posterior-mean y1 labels (matches reg_graphTrip / build_dmt_dataset.py)
ALL_SUBJECTS = ["S01", "S02", "S05", "S06", "S07", "S10", "S12", "S13"]


class EEGDataset(Dataset):
    """
    EEG dataset for plasma DMT regression.

    The source .npz already has the 32 pure EEG channels — no exclusion needed.

    Args:
        data_path: Path to .npz file (default: data/eeg_dmt_regression.npz).
        subjects:  List of subject IDs to include (e.g. ['S01', 'S02']).
        normalize: If True, z-score normalise per channel per trial.
    """

    def __init__(self, data_path=None, subjects=None, normalize=True):
        data_path = data_path or DEFAULT_DATA_PATH
        ds = np.load(data_path, allow_pickle=True)

        all_eeg = ds["eeg_data"]        # (N, 32, 3000)
        all_subjects = ds["subjects"]   # (N,) subject IDs
        all_labels = ds["labels"]       # (N,) plasma ng/mL
        all_times = ds["times"]         # (N,) minutes post-dose

        # Filter by subjects
        if subjects is not None:
            mask = np.isin(all_subjects, subjects)
            all_eeg = all_eeg[mask]
            all_labels = all_labels[mask]
            all_subjects = all_subjects[mask]
            all_times = all_times[mask]

        # Convert to float32
        all_eeg = all_eeg.astype(np.float32)

        # Z-score normalize per channel per trial (signal scaling only; labels stay raw ng/mL)
        if normalize:
            mean = all_eeg.mean(axis=2, keepdims=True)
            std = all_eeg.std(axis=2, keepdims=True) + 1e-8
            all_eeg = (all_eeg - mean) / std

        # Store in RAM as tensors
        self.eeg = torch.from_numpy(all_eeg)
        self.labels = torch.from_numpy(all_labels.astype(np.float32))
        self.subjects = all_subjects
        self.times = all_times.astype(np.float32)
        self.n_channels = self.eeg.shape[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.eeg[idx], self.labels[idx]


def get_subject_split(val_subjects, all_subjects=None):
    """Return (train_subjects, val_subjects) ensuring no overlap."""
    if all_subjects is None:
        all_subjects = ALL_SUBJECTS
    train_subjects = [s for s in all_subjects if s not in val_subjects]
    return train_subjects, val_subjects
