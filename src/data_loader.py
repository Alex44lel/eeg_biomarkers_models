"""Load and preprocess plasma concentration data for the DMT condition."""
from pathlib import Path

import pandas as pd


def load_data(csv_path: str | Path = "data/plasma_clean.csv") -> dict:
    """
    Load the plasma_clean.csv and return flat arrays for PyMC.

    Only the DMT condition is used. The pre-dose baseline (time_point == 0)
    is excluded from the likelihood because P(t=0) = P0 (injection bolus)
    would contradict the near-zero baseline measurements.

    Returns
    -------
    dict with keys:
        df               — filtered DataFrame (DMT, post-dose only)
        t_obs_flat       — observed times, shape (N_total,)
        P_obs_flat       — observed plasma concentrations, shape (N_total,)
        subj_idx_flat    — integer subject index, shape (N_total,)
        is_imputed_flat  — boolean imputed flag, shape (N_total,)
        subject_ids      — list of subject ID strings
        n_subjects       — int
        dose_per_subject — array of dose_mg per subject, shape (n_subjects,)
        t_max            — float, maximum observed time
    """
    df = pd.read_csv(csv_path)

    # DMT condition only, exclude pre-dose row (time_point == 0)
    dmt = (
        df[(df["condition"] == "dmt") & (df["time_point"] > 0)]
        .copy()
        .reset_index(drop=True)
    )

    subject_ids = sorted(dmt["subject"].unique())
    n_subjects = len(subject_ids)
    subj_to_idx = {s: i for i, s in enumerate(subject_ids)}
    dmt["subj_idx"] = dmt["subject"].map(subj_to_idx)

    t_obs_flat = dmt["time_min"].values.astype(float)
    P_obs_flat = dmt["plasma_conc"].values.astype(float)
    subj_idx_flat = dmt["subj_idx"].values.astype(int)
    is_imputed_flat = dmt["is_imputed"].values.astype(bool)

    dose_per_subject = (
        dmt.groupby("subj_idx")["dose_mg"].first().sort_index().values
    )

    return {
        "df": dmt,
        "t_obs_flat": t_obs_flat,
        "P_obs_flat": P_obs_flat,
        "subj_idx_flat": subj_idx_flat,
        "is_imputed_flat": is_imputed_flat,
        "subject_ids": subject_ids,
        "n_subjects": n_subjects,
        "dose_per_subject": dose_per_subject,
        "t_max": float(dmt["time_min"].max()),
    }
