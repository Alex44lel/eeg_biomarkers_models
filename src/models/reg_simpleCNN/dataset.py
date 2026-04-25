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

# Label-source registry. Keys are the --dataset CLI values consumed by train_cv.
#   pk       : PK-model posterior-mean y1 curve (default, built by reg_graphTrip/build_dmt_dataset.py)
#   biexp    : per-subject bi-exponential fit (built by reg_simpleCNN/build_biexp_dataset.py)
#   pk_k{K}  : polyphase-downsampled view of `pk` with factor K ∈ {2,3,4}.
#              K * N_orig rows; each row has length 3000//K and a `k_idx`
#              field in [0, K). Subject IDs are preserved so LOSO grouping
#              keeps all K splits of a subject together.
#              Built by reg_simpleCNN/build_downsampled_dataset.py.
DATASET_PATHS = {
    "pk":     DEFAULT_DATA_PATH,
    "biexp":  PROJECT_ROOT / "data" / "eeg_dmt_regression_biexp.npz",
    "pk_k2":  PROJECT_ROOT / "data" / "eeg_dmt_regression_k2.npz",
    "pk_k3":  PROJECT_ROOT / "data" / "eeg_dmt_regression_k3.npz",
    "pk_k4":  PROJECT_ROOT / "data" / "eeg_dmt_regression_k4.npz",
    "pk_k5":  PROJECT_ROOT / "data" / "eeg_dmt_regression_k5.npz",
    "pk_k10": PROJECT_ROOT / "data" / "eeg_dmt_regression_k10.npz",
}

# Subjects with PK posterior-mean y1 labels (matches reg_graphTrip / build_dmt_dataset.py)
ALL_SUBJECTS = ["S01", "S02", "S05", "S06", "S07", "S10", "S12", "S13"]


def resolve_dataset_path(dataset=None, data_path=None):
    """Return a Path to the .npz to load.

    Priority: explicit data_path > dataset key > DEFAULT_DATA_PATH.
    """
    if data_path is not None:
        return Path(data_path)
    if dataset is not None:
        if dataset not in DATASET_PATHS:
            raise ValueError(
                f"Unknown dataset key {dataset!r}. "
                f"Available: {sorted(DATASET_PATHS)}"
            )
        return Path(DATASET_PATHS[dataset])
    return Path(DEFAULT_DATA_PATH)


class EEGDataset(Dataset):
    """
    EEG dataset for plasma DMT regression.

    The source .npz already has the 32 pure EEG channels — no exclusion needed.

    Args:
        data_path: Path to .npz file. If given, takes precedence over `dataset`.
        dataset:   Registry key in DATASET_PATHS ('pk' or 'biexp'). Ignored if
                   data_path is set. Defaults to DEFAULT_DATA_PATH when both
                   are None.
        subjects:  List of subject IDs to include (e.g. ['S01', 'S02']).
        normalize: If True, z-score normalise per channel per trial.
    """

    def __init__(self, data_path=None, subjects=None, normalize=True,
                 dataset=None):
        data_path = resolve_dataset_path(dataset=dataset, data_path=data_path)
        ds = np.load(data_path, allow_pickle=True)

        all_eeg = ds["eeg_data"]        # (N, 32, L)
        all_subjects = ds["subjects"]   # (N,) subject IDs
        all_labels = ds["labels"]       # (N,) plasma ng/mL
        all_times = ds["times"]         # (N,) minutes post-dose

        # Optional polyphase-downsampling metadata (present in
        # eeg_dmt_regression_k{K}.npz produced by build_downsampled_dataset).
        all_k_idx = ds["k_idx"] if "k_idx" in ds.files else None
        all_orig_id = ds["orig_trial_id"] if "orig_trial_id" in ds.files else None
        self.k_factor = int(ds["k_factor"]) if "k_factor" in ds.files else 1

        # Filter by subjects
        if subjects is not None:
            mask = np.isin(all_subjects, subjects)
            all_eeg = all_eeg[mask]
            all_labels = all_labels[mask]
            all_subjects = all_subjects[mask]
            all_times = all_times[mask]
            if all_k_idx is not None:
                all_k_idx = all_k_idx[mask]
            if all_orig_id is not None:
                all_orig_id = all_orig_id[mask]

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
        self.k_idx = all_k_idx              # None when source is non-polyphase
        self.orig_trial_id = all_orig_id    # None when source is non-polyphase
        self.n_channels = self.eeg.shape[1]
        self.signal_length = self.eeg.shape[2]

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
