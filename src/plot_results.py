"""
Visualise posterior predictive plasma concentration curves and parameter posteriors.

Produces:
    results/figures/posterior_predictions.png
    results/figures/parameter_posteriors.png
    results/figures/hyperparameter_posteriors.png

Usage:
    python -m src.plot_results
    # or
    python src/plot_results.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_data


# Colour palette
_C = {
    "curve": "#1f77b4",
    "band": "#aec7e8",
    "obs": "#d62728",
    "imputed": "#ff7f0e",
}


def _ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Posterior predictions per subject
# ---------------------------------------------------------------------------

def plot_posterior_predictions(
    interp_path: str | Path = "results/plasma_interpolated.csv",
    csv_path: str | Path = "data/plasma_clean.csv",
    out_path: str | Path = "results/figures/posterior_predictions.png",
):
    """One subplot per subject: observed points + posterior mean + 94% HDI."""
    data = load_data(csv_path)
    df_interp = pd.read_csv(interp_path)

    # Raw data including baseline (for plotting t=0)
    df_raw = pd.read_csv(csv_path)
    df_raw = df_raw[df_raw["condition"] == "dmt"]

    n_subj = data["n_subjects"]
    ncols = 4
    nrows = (n_subj + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    axes = axes.flatten()

    for i, subj in enumerate(data["subject_ids"]):
        ax = axes[i]
        curve = df_interp[df_interp["subject"] == subj]
        obs_all = df_raw[df_raw["subject"] == subj]
        dose = data["dose_per_subject"][i]

        # HDI shading
        ax.fill_between(
            curve["time_min"],
            curve["plasma_hdi_low"],
            curve["plasma_hdi_high"],
            color=_C["band"],
            alpha=0.6,
            label="94% HDI",
        )
        # Posterior mean
        ax.plot(
            curve["time_min"],
            curve["plasma_mean"],
            color=_C["curve"],
            lw=2,
            label="Posterior mean",
        )
        # Observed (non-imputed)
        obs_real = obs_all[~obs_all["is_imputed"]]
        obs_imp = obs_all[obs_all["is_imputed"]]
        ax.scatter(
            obs_real["time_min"], obs_real["plasma_conc"],
            color=_C["obs"], zorder=5, s=40, label="Observed",
        )
        if not obs_imp.empty:
            ax.scatter(
                obs_imp["time_min"], obs_imp["plasma_conc"],
                color=_C["imputed"], marker="D", zorder=5, s=40, label="Imputed",
            )

        ax.set_title(f"{subj}  ({dose:.0f} mg)", fontsize=10)
        ax.set_xlabel("Time (min)", fontsize=8)
        ax.set_ylabel("ng/mL", fontsize=8)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=7)

    # Hide surplus axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Build legend from all subplots so imputed marker is included
    # even when it only appears in some subjects.
    seen = {}
    for ax in axes[:n_subj]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen[l] = h
    fig.legend(seen.values(), seen.keys(), loc="lower right", fontsize=8,
               ncol=2, bbox_to_anchor=(0.98, 0.02))

    fig.suptitle("Two-compartment PK model — posterior predictions (DMT)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    _ensure_dir(Path(out_path))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# 2. Subject-level parameter posteriors (violin)
# ---------------------------------------------------------------------------

def plot_parameter_posteriors(
    trace_path: str | Path = "results/trace.nc",
    csv_path: str | Path = "data/plasma_clean.csv",
    out_path: str | Path = "results/figures/parameter_posteriors.png",
):
    """Violin plots of k0, k1, k2, P0 per subject."""
    data = load_data(csv_path)
    trace = az.from_netcdf(str(trace_path))
    subject_ids = data["subject_ids"]

    # Rate constants on log scale (log-normal model → log is the natural space).
    # P0 on linear scale (values are in ng/mL and easier to read linearly).
    params = [
        ("k0", "k₀ (min⁻¹)", True),
        ("k1", "k₁ (min⁻¹)", True),
        ("k2", "k₂ (min⁻¹)", True),
        ("P0", "P₀ (ng/mL)",  False),
    ]

    fig, axes = plt.subplots(1, len(params), figsize=(16, 4))

    for ax, (param, ylabel, log_scale) in zip(axes, params):
        samples = trace.posterior[param].values   # (chains, draws, subjects)
        flat = samples.reshape(-1, len(subject_ids))  # (n_samples, n_subjects)

        if log_scale:
            # Plot log10 of the samples so violin widths are meaningful
            plot_data = np.log10(np.clip(flat, 1e-6, None))
            ax.set_ylabel(f"log₁₀ {ylabel}", fontsize=9)
        else:
            plot_data = flat
            ax.set_ylabel(ylabel, fontsize=9)

        parts = ax.violinplot(plot_data, positions=range(len(subject_ids)),
                              showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(_C["curve"])
            pc.set_alpha(0.5)
        parts["cmedians"].set_color("black")

        ax.set_xticks(range(len(subject_ids)))
        ax.set_xticklabels(subject_ids, rotation=45, ha="right", fontsize=7)
        ax.set_title(param, fontsize=11)

    fig.suptitle("Posterior distributions — subject-level parameters", fontsize=12)
    fig.tight_layout()

    _ensure_dir(Path(out_path))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# 3. Hyperparameter posteriors
# ---------------------------------------------------------------------------

def plot_hyperparameters(
    trace_path: str | Path = "results/trace.nc",
    out_path: str | Path = "results/figures/hyperparameter_posteriors.png",
):
    """Posterior density plots for population-level hyperparameters."""
    trace = az.from_netcdf(str(trace_path))
    hyperparams = [
        "log_k0_mu", "log_k1_mu", "log_k2_mu", "log_P0_mu",
        "log_k0_sigma", "log_k1_sigma", "log_k2_sigma", "log_P0_sigma",
        "plasma_sigma",
    ]

    axes = az.plot_posterior(trace, var_names=hyperparams, grid=(3, 3), figsize=(15, 10))
    fig = axes.flatten()[0].get_figure()
    fig.suptitle("Population-level hyperparameter posteriors (log scale for k/P0)",
                 fontsize=13, y=1.01)
    fig.tight_layout()

    _ensure_dir(Path(out_path))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("Plotting posterior predictions...")
    plot_posterior_predictions()

    print("Plotting parameter posteriors...")
    plot_parameter_posteriors()

    print("Plotting hyperparameter posteriors...")
    plot_hyperparameters()


if __name__ == "__main__":
    main()
