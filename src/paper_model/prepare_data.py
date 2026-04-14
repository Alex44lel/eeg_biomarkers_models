import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
EXCLUDED_SUBJECTS = ["S03", "S04", "S09", "S11"]


def load_and_prepare(exclude_subjects=None):
    """Load LZc data and prepare it for the two-compartment model.

    Times are shifted so t=0 is injection time for each subject.
    Only post-injection data (t >= 0) is returned.

    Returns dict with:
        t_model       : np.array — time since injection in minutes (N_obs,)
        lzc          : np.array — raw LZc values in [0, 1] (N_obs,)
        subject_idx  : np.array — integer subject index per obs (N_obs,)
        subject_names: list     — ordered subject labels
    """
    if exclude_subjects is None:
        exclude_subjects = EXCLUDED_SUBJECTS

    lzc_df = pd.read_csv(RESULTS_DIR / "lzc" / "lzc_results.csv")
    offsets_df = pd.read_csv(RESULTS_DIR / "lzc" / "injection_offsets.csv")

    df = lzc_df.merge(offsets_df, on="subject")

    if exclude_subjects:
        df = df[~df["subject"].isin(exclude_subjects)]

    # Shift time so t=0 = injection, keep only post-injection
    df["t_model"] = df["time_min"] - df["injection_time_min"]
    df = df[df["t_model"] >= 0].copy()

    # Use raw LZc (already in [0, 1] from LZ76 normalization)
    df["lzc"] = df["lzc_raw"]

    # Build integer subject index
    subject_names = sorted(df["subject"].unique())
    subj_to_idx = {s: i for i, s in enumerate(subject_names)}
    df["subject_idx"] = df["subject"].map(subj_to_idx)

    return {
        "t_model": df["t_model"].values.astype(np.float64),
        "lzc": df["lzc"].values.astype(np.float64),
        "subject_idx": df["subject_idx"].values.astype(np.int64),
        "subject_names": subject_names,
    }


def load_plasma_data(subject_names, exclude_subjects=None):
    """Load plasma DMT concentrations aligned with the model's subject indices.

    Times are relative to injection (t=0 = injection).

    Args:
        subject_names: ordered list of subject labels (from load_and_prepare).
        exclude_subjects: subjects to exclude (defaults to EXCLUDED_SUBJECTS).

    Returns dict with:
        plasma_t       : np.array — time since injection in minutes
        plasma_conc    : np.array — plasma concentration (ng/mL)
        plasma_subj_idx: np.array — integer subject index per observation
    """
    if exclude_subjects is None:
        exclude_subjects = EXCLUDED_SUBJECTS

    pdf = pd.read_csv(DATA_DIR / "plasma_clean.csv")
    pdf = pdf[pdf["condition"] == "dmt"]
    pdf = pdf[pdf["time_point"] != 0]  # exclude pre-injection baseline

    if exclude_subjects:
        pdf = pdf[~pdf["subject"].isin(exclude_subjects)]

    # Keep only subjects present in the model
    pdf = pdf[pdf["subject"].isin(subject_names)]

    # time_min is already relative to injection (t=0 = injection)
    subj_to_idx = {s: i for i, s in enumerate(subject_names)}
    pdf["subject_idx"] = pdf["subject"].map(subj_to_idx)

    # Normalize plasma concentrations to [0, 1]
    plasma_max = pdf["plasma_conc"].max()
    print(f"  Plasma normalization factor: {plasma_max:.1f} ng/mL")

    return {
        "plasma_t": pdf["time_min"].values.astype(np.float64),
        "plasma_conc": (pdf["plasma_conc"] / plasma_max).values.astype(np.float64),
        "plasma_subj_idx": pdf["subject_idx"].values.astype(np.int64),
        "plasma_max_ngml": plasma_max,
    }
