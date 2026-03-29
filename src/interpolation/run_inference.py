"""
Run MCMC sampling for the two-compartment PK model and save results.

Usage:
    python -m src.interpolation.run_inference
"""
from src.interpolation.pk_model import build_model
from src.interpolation.data_loader import load_data
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import pymc as pm


def main():
    results_dir = Path("results")
    diag_dir = results_dir / "figures" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    print("Loading data...")
    data = load_data("data/plasma_clean.csv")
    print(f"  {data['n_subjects']} subjects, {len(data['t_obs_flat'])} post-dose observations")

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    print("Building model...")
    model = build_model(data)

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    print("Sampling (2 chains × 1000 draws + 1000 tune)...")
    with model:
        trace = pm.sample(
            draws=1000,
            tune=1000,
            chains=2,
            target_accept=0.9,
            return_inferencedata=True,
            progressbar=True,
        )

    # ------------------------------------------------------------------ #
    # Save trace
    # ------------------------------------------------------------------ #
    trace_path = results_dir / "trace.nc"  # trace strores the 2000 samples for every parameter
    az.to_netcdf(trace, str(trace_path))
    print(f"\nTrace saved → {trace_path}")

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    param_vars = [
        "log_k0_mu", "log_k1_mu", "log_k2_mu", "log_P0_mu",
        "log_k0_sigma", "log_k1_sigma", "log_k2_sigma", "log_P0_sigma",
        "plasma_sigma", "k0", "k1", "k2", "P0",
    ]

    summary = az.summary(trace, var_names=param_vars)
    summary_path = diag_dir / "summary.csv"
    summary.to_csv(summary_path)
    print(f"Summary saved → {summary_path}\n")
    print(summary.to_string())

    n_div = int(trace.sample_stats.diverging.sum())
    print(f"\nDivergences: {n_div}")
    if n_div > 0:
        print("  Consider increasing target_accept or reparametrising.")

    # Trace plots for population-level parameters
    az.plot_trace(
        trace,
        var_names=[
            "log_k0_mu", "log_k1_mu", "log_k2_mu", "log_P0_mu",
            "log_k0_sigma", "log_k1_sigma", "log_k2_sigma", "log_P0_sigma",
            "plasma_sigma",
        ],
        combined=False,
    )
    plt.savefig(diag_dir / "trace_plots.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Trace plots saved → {diag_dir / 'trace_plots.png'}")

    return trace, data


if __name__ == "__main__":
    main()
