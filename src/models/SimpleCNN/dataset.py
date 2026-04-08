"""
PyTorch Dataset for EEG DMT classification.

Loads the entire .npz dataset into RAM for fast training.
Supports subject-level train/validation splits.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "eeg_dmt_dataset.npz"

# Non-EEG channel indices to exclude by default (ECG=31, VEOG=32, EMGfront=33, EMGtemp=34)
NON_EEG_INDICES = [31, 32, 33, 34]


class EEGDataset(Dataset):
    """
    EEG dataset for pre vs post injection classification.

    Args:
        data_path: Path to .npz file.
        subjects: List of subject IDs to include (e.g. ['S01', 'S02']).
        exclude_channels: Channel indices to exclude. Defaults to non-EEG channels.
        normalize: If True, z-score normalize per channel per trial.
    """

    def __init__(self, data_path=None, subjects=None, exclude_channels=None, normalize=True):
        data_path = data_path or DEFAULT_DATA_PATH
        ds = np.load(data_path, allow_pickle=True)

        all_eeg = ds["eeg"]  # (N, 36, 3000)
        all_subjects = ds["subjects"]
        all_labels = ds["labels"]

        # Filter by subjects
        if subjects is not None:
            mask = np.isin(all_subjects, subjects)
            all_eeg = all_eeg[mask]
            all_labels = all_labels[mask]
            all_subjects = all_subjects[mask]

        # Exclude non-EEG channels
        if exclude_channels is None:
            exclude_channels = NON_EEG_INDICES
        if exclude_channels:
            keep = [i for i in range(all_eeg.shape[1]) if i not in exclude_channels]
            all_eeg = all_eeg[:, keep, :]

        # Convert to float32
        all_eeg = all_eeg.astype(np.float32)

        # Z-score normalize per channel per trial
        if normalize:
            mean = all_eeg.mean(axis=2, keepdims=True)
            std = all_eeg.std(axis=2, keepdims=True) + 1e-8
            all_eeg = (all_eeg - mean) / std

        # Store in RAM as tensors
        self.eeg = torch.from_numpy(all_eeg)
        self.labels = torch.from_numpy(all_labels.astype(np.int64))
        self.subjects = all_subjects
        self.n_channels = self.eeg.shape[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.eeg[idx], self.labels[idx]


def get_subject_split(val_subjects, all_subjects=None):
    """
    Return (train_subjects, val_subjects) ensuring no overlap.

    Args:
        val_subjects: List of subject IDs for validation.
        all_subjects: List of all available subject IDs. If None, uses default.
    """
    if all_subjects is None:
        all_subjects = ["S01", "S02", "S04", "S05", "S06", "S07", "S08", "S10", "S12", "S13"]
    train_subjects = [s for s in all_subjects if s not in val_subjects]
    return train_subjects, val_subjects
