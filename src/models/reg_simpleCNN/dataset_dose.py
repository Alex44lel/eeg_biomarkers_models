"""
Dose-aware wrapper around EEGDataset.

Joins per-subject initial DMT dose (mg) from data/plasma_clean.csv and yields
(eeg, label, dose_mg) per sample. The original EEGDataset is not modified.

Dose mapping (for the 8 subjects used in ALL_SUBJECTS):
    S01=7, S02=7, S05=14, S06=14, S07=20, S10=20, S12=20, S13=20
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .dataset import EEGDataset, ALL_SUBJECTS

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLASMA_CSV = PROJECT_ROOT / "data" / "plasma_clean.csv"


def load_subject_dose_map(csv_path=None):
    """Return {subject_id: dose_mg (float)} from plasma_clean.csv.

    Each subject has exactly one non-null dose across its rows; we take the
    first non-null and assert consistency.
    """
    csv_path = csv_path or PLASMA_CSV
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["dose_mg"])
    mapping = {}
    for subj, grp in df.groupby("subject"):
        unique = grp["dose_mg"].unique()
        if len(unique) != 1:
            raise ValueError(f"Subject {subj} has multiple dose values: {unique}")
        mapping[subj] = float(unique[0])
    return mapping


class EEGDoseDataset(Dataset):
    """EEGDataset + per-sample dose_mg tensor. Returns (eeg, label, dose)."""

    def __init__(self, data_path=None, subjects=None, normalize=True,
                 csv_path=None):
        self.base = EEGDataset(data_path=data_path, subjects=subjects,
                               normalize=normalize)
        dose_map = load_subject_dose_map(csv_path)
        missing = [s for s in np.unique(self.base.subjects) if s not in dose_map]
        if missing:
            raise ValueError(f"No dose found for subjects: {missing}")
        dose_arr = np.asarray(
            [dose_map[s] for s in self.base.subjects], dtype=np.float32
        )
        self.doses = torch.from_numpy(dose_arr)
        self.n_channels = self.base.n_channels
        self.labels = self.base.labels
        self.subjects = self.base.subjects
        self.times = self.base.times

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        eeg, label, _sidx = self.base[idx]
        return eeg, label, self.doses[idx]
