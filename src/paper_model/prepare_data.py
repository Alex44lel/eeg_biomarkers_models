import numpy as np
import pandas as pd
from pathlib import Path

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

    return {
        "t_model": df["t_model"].values.astype(np.float64),
        "lzc": df["lzc"].values.astype(np.float64),
        "subject_idx": df["subject_idx"].values.astype(np.int64),
        "subject_names": subject_names,
    }
