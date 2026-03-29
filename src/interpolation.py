"""
Posterior predictive interpolation of plasma concentration curves.

For each subject, draws smooth P(t) curves from the posterior and
computes the mean + 94% HDI at 500 evenly-spaced time points.

Usage:
    python -m src.interpolation
    # or
    python src/interpolation.py
"""
from src.pk_model import plasma_concentration_np
from src.data_loader import load_data
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import arviz as az

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def interpolate_plasma(
    trace_path: str | Path = "results/trace.nc",
    csv_path: str | Path = "data/plasma_clean.csv",
    n_grid: int = 1200,  # if 20 minutes this means a measurement per second
    hdi_prob: float = 0.94,
) -> pd.DataFrame:
    """
    Compute smooth posterior predictive plasma concentration curves.

    Algorithm
    ---------
    1. Load posterior samples of k0, k1, k2, P0 per subject.
    2. For each subject, evaluate P(t_grid) for every sample.
    3. Summarise with mean + HDI over the sample dimension.

    Returns
    -------
    DataFrame with columns:
        subject, time_min, plasma_mean, plasma_hdi_low, plasma_hdi_high
    """
    data = load_data(csv_path)
    trace = az.from_netcdf(str(trace_path))

    # upper limit set to 20 minutes as it is where we will cut the eeg signals
    t_grid = np.linspace(0.0, 20, n_grid)

    # Posterior arrays: shape (n_chains, n_draws, n_subjects)
    k0_post = trace.posterior["k0"].values
    k1_post = trace.posterior["k1"].values
    k2_post = trace.posterior["k2"].values
    P0_post = trace.posterior["P0"].values

    n_chains, n_draws, n_subj = k0_post.shape
    # Flatten chain × draw → samples
    k0_flat = k0_post.reshape(-1, n_subj)
    k1_flat = k1_post.reshape(-1, n_subj)
    k2_flat = k2_post.reshape(-1, n_subj)
    P0_flat = P0_post.reshape(-1, n_subj)
    n_samples = k0_flat.shape[0]

    records = []
    for i, subj in enumerate(data["subject_ids"]):
        print(f"  Interpolating {subj}  ({i + 1}/{n_subj})...")

        # Vectorise over samples: broadcast t_grid (n_grid,) against
        # per-sample scalars (n_samples,) using extra axis.
        t = t_grid[np.newaxis, :]            # (1, n_grid)
        k0 = k0_flat[:, i, np.newaxis]       # (n_samples, 1)
        k1 = k1_flat[:, i, np.newaxis]
        k2 = k2_flat[:, i, np.newaxis]
        P0 = P0_flat[:, i, np.newaxis]

        # P_mat shape: (n_samples, n_grid)
        P_mat = plasma_concentration_np(t, P0, k0, k1, k2)

        P_mean = P_mat.mean(axis=0)                        # (n_grid,)
        hdi = az.hdi(P_mat, hdi_prob=hdi_prob)             # (n_grid, 2)
        P_low = hdi[:, 0]
        P_high = hdi[:, 1]

        for j in range(n_grid):
            records.append(
                {
                    "subject": subj,
                    "time_min": t_grid[j],
                    "plasma_mean": P_mean[j],
                    "plasma_hdi_low": P_low[j],
                    "plasma_hdi_high": P_high[j],
                }
            )

    return pd.DataFrame(records)


def main():
    out_path = Path("results/plasma_interpolated.csv")
    out_path.parent.mkdir(exist_ok=True)

    print("Computing posterior predictive curves...")
    df = interpolate_plasma()
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}  ({len(df)} rows)")
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()
