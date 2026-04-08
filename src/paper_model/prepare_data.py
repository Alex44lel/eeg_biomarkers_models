import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
EXCLUDED_SUBJECTS = ["S03", "S09", "S11"]


def load_and_prepare(exclude_subjects=None):
    """Load LZc data and prepare it for the two-compartment model.

    Returns dict with:
        t_model       : np.array —  times in minutes (N_obs,)
        lzc          : np.array — LZc values as normalized % / 10 (N_obs,)
        subject_idx  : np.array — integer subject index per obs (N_obs,)
        subject_names: list     — ordered subject labels
    """
    if exclude_subjects is None:
        exclude_subjects = EXCLUDED_SUBJECTS

    lzc_df = pd.read_csv(RESULTS_DIR / "lzc" / "lzc_results.csv")
    offsets_df = pd.read_csv(RESULTS_DIR / "lzc" / "injection_offsets.csv")

    df = lzc_df.merge(offsets_df, on="subject")
    df["t_model"] = df["time_min"]

    if exclude_subjects:
        df = df[~df["subject"].isin(exclude_subjects)]

    # The paper plots lzc_normalized directly (percentage change from baseline).
    df["lzc"] = (df["lzc_normalized"] + 5) / 10

    # Build integer subject index (use all data, not just post-injection)
    subject_names = sorted(df["subject"].unique())
    subj_to_idx = {s: i for i, s in enumerate(subject_names)}
    df["subject_idx"] = df["subject"].map(subj_to_idx)

    # Per-subject injection time (aligned with subject_names order)
    inj_map = df.groupby("subject")["injection_time_min"].first()
    injection_times = [inj_map[s] for s in subject_names]

    return {
        "t_model": df["t_model"].values.astype(np.float64),
        "lzc": df["lzc"].values.astype(np.float64),
        "subject_idx": df["subject_idx"].values.astype(np.int64),
        "subject_names": subject_names,
        "injection_times": injection_times,
    }


def load_plasma_data(subject_names, exclude_subjects=None):
    """Load plasma DMT concentrations aligned with the model's subject indices.

    Args:
        subject_names: ordered list of subject labels (from load_and_prepare).
        exclude_subjects: subjects to exclude (defaults to EXCLUDED_SUBJECTS).

    Returns dict with:
        plasma_t       : np.array — time points in minutes
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

    # Shift plasma times: time_min=0 is injection, add injection offset
    # so times align with the LZc recording-start reference
    offsets_df = pd.read_csv(RESULTS_DIR / "lzc" / "injection_offsets.csv")
    offset_map = offsets_df.set_index("subject")["injection_time_min"]
    pdf["time_min"] = pdf["time_min"] + pdf["subject"].map(offset_map)

    subj_to_idx = {s: i for i, s in enumerate(subject_names)}
    pdf["subject_idx"] = pdf["subject"].map(subj_to_idx)

    return {
        "plasma_t": pdf["time_min"].values.astype(np.float64),
        "plasma_conc": pdf["plasma_conc"].values.astype(np.float64),
        "plasma_subj_idx": pdf["subject_idx"].values.astype(np.int64),
    }
