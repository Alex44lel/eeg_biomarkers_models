import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from pathlib import Path
from .partial_pooling_model import compute_posterior_predictive, compute_posterior_predictive_y1


def plot_predictions(trace, data, output_path, t_max=18.0, n_grid=200):
    """Plot per-subject model predictions vs observed data (Figure 26 replica).

    Args:
        trace: ArviZ InferenceData with posterior samples.
        data: dict from prepare_data.load_and_prepare().
        output_path: Path to save the figure.
        t_max: maximum time on x-axis.
        n_grid: number of points in the fine time grid per subject.
    """
    subject_names = data["subject_names"]
    n_subjects = len(subject_names)

    # Build a fine time grid for each subject
    t_fine = np.linspace(0.01, t_max, n_grid)
    t_grid = np.tile(t_fine, n_subjects)
    subject_idx_grid = np.repeat(np.arange(n_subjects), n_grid)

    # Compute posterior predictions (use predictive with noise for original plot)
    _, predictions = compute_posterior_predictive(
        trace, t_grid, subject_idx_grid, n_subjects
    )
    # Layout: 4 rows x 3 cols
    n_cols = 3
    n_rows = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, subj_name in enumerate(subject_names):
        ax = axes[i]

        # Observed data for this subject
        mask_obs = data["subject_idx"] == i
        t_obs = data["t_model"][mask_obs]
        lzc_obs = data["lzc"][mask_obs]

        # Predicted curves for this subject
        mask_grid = subject_idx_grid == i
        t_subj = t_grid[mask_grid]
        preds_subj = predictions[:, mask_grid]  # (n_samples, n_grid)

        mean_pred = preds_subj.mean(axis=0)
        hdi_bounds = az.hdi(preds_subj, hdi_prob=0.94)
        hdi_low = hdi_bounds[:, 0]
        hdi_high = hdi_bounds[:, 1]

        ax.scatter(t_obs, lzc_obs, s=8, color="purple", alpha=0.5, zorder=2)
        ax.plot(t_subj, mean_pred, color="black", linewidth=1.5, zorder=3)
        ax.fill_between(t_subj, hdi_low, hdi_high, color="grey", alpha=0.3, zorder=1)
        ax.set_title(subj_name)
        ax.set_xlim(-0.5, t_max)
        ax.set_ylim(0, 3.2)

    # Hide unused subplots
    for j in range(n_subjects, len(axes)):
        axes[j].set_visible(False)

    # Shared labels
    for ax in axes[n_cols * (n_rows - 1):n_cols * n_rows]:
        if ax.get_visible():
            ax.set_xlabel("Fake time (min)")
    for r in range(n_rows):
        axes[r * n_cols].set_ylabel("LZc")

    fig.suptitle(
        "Two Compartment: Partial Pooling Model Predictions vs Observed Data",
        fontsize=14, y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved predictions plot to {output_path}")


def _plot_dual_axis(predictions, subject_names, subject_idx_grid, t_grid,
                    plasma_df, title, output_path, ylabel="model units",
                    t_max=18.0):
    """Shared helper for y1 plots with plasma overlay on secondary y-axis."""
    n_subjects = len(subject_names)
    n_cols = 3
    n_rows = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12), sharex=True)
    axes = axes.flatten()

    for i, subj_name in enumerate(subject_names):
        ax = axes[i]

        mask_grid = subject_idx_grid == i
        t_subj = t_grid[mask_grid]
        preds_subj = predictions[:, mask_grid]

        mean_pred = preds_subj.mean(axis=0)
        hdi_bounds = az.hdi(preds_subj, hdi_prob=0.94)
        hdi_low = hdi_bounds[:, 0]
        hdi_high = hdi_bounds[:, 1]

        ax.plot(t_subj, mean_pred, color="black", linewidth=1.5, zorder=3, label="Model mean")
        ax.fill_between(t_subj, hdi_low, hdi_high, color="grey", alpha=0.3, zorder=1, label="94% HDI")
        ax.set_title(subj_name)
        ax.set_xlim(-0.5, t_max)

        # Real plasma data on secondary y-axis
        subj_plasma = plasma_df[plasma_df["subject"] == subj_name]
        if not subj_plasma.empty:
            ax2 = ax.twinx()
            ax2.scatter(
                subj_plasma["time_min"], subj_plasma["plasma_conc"],
                s=20, color="red", marker="x", zorder=4, label="Plasma (ng/mL)",
            )
            ax2.set_ylabel("Plasma (ng/mL)", fontsize=8, color="red")
            ax2.tick_params(axis="y", labelcolor="red", labelsize=7)

        # Legend (combine handles from both axes)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            if not subj_plasma.empty:
                h2, l2 = ax2.get_legend_handles_labels()
                handles += h2
                labels += l2
            ax.legend(handles, labels, fontsize=6, loc="upper right")

    for j in range(n_subjects, len(axes)):
        axes[j].set_visible(False)

    for ax in axes[n_cols * (n_rows - 1):n_cols * n_rows]:
        if ax.get_visible():
            ax.set_xlabel("Fake time (min)")
    for r in range(n_rows):
        axes[r * n_cols].set_ylabel(ylabel)

    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def _load_plasma_df():
    """Load and filter plasma DMT data for plotting."""
    plasma_path = Path(__file__).resolve().parents[2] / "data" / "plasma_clean.csv"
    plasma_df = pd.read_csv(plasma_path)
    return plasma_df[plasma_df["condition"] == "dmt"]


def plot_predictions_y1(trace, data, output_path_curve, output_path_predictive,
                        t_max=18.0, n_grid=200):
    """Generate two y1 plots: posterior curves and posterior predictive (with noise)."""
    subject_names = data["subject_names"]
    n_subjects = len(subject_names)
    plasma_df = _load_plasma_df()

    t_fine = np.linspace(0.01, t_max, n_grid)
    t_grid = np.tile(t_fine, n_subjects)
    subject_idx_grid = np.repeat(np.arange(n_subjects), n_grid)

    curve_samples, predictive_samples = compute_posterior_predictive_y1(
        trace, t_grid, subject_idx_grid, n_subjects
    )

    _plot_dual_axis(curve_samples, subject_names, subject_idx_grid, t_grid, plasma_df,
                    "Plasma DMT (y1) — Posterior Curves", output_path_curve,
                    ylabel="y1 (model units)", t_max=t_max)

    _plot_dual_axis(predictive_samples, subject_names, subject_idx_grid, t_grid, plasma_df,
                    "Plasma DMT (y1) — Posterior Predictive", output_path_predictive,
                    ylabel="y1 (model units)", t_max=t_max)


def _plot_y2(predictions, subject_names, subject_idx_grid, t_grid,
             data, plasma_df, title, output_path, t_max=18.0):
    """Helper for y2 plots: primary = model predictions, secondary = observed LZc scale,
    plus red DMT plasma points (no axis)."""
    n_subjects = len(subject_names)
    n_cols = 3
    n_rows = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12), sharex=True)
    axes = axes.flatten()
    twin_axes = {}

    for i, subj_name in enumerate(subject_names):
        ax = axes[i]

        # Model predictions on primary axis
        mask_grid = subject_idx_grid == i
        t_subj = t_grid[mask_grid]
        preds_subj = predictions[:, mask_grid]

        mean_pred = preds_subj.mean(axis=0)
        hdi_bounds = az.hdi(preds_subj, hdi_prob=0.94)
        hdi_low = hdi_bounds[:, 0]
        hdi_high = hdi_bounds[:, 1]

        ax.plot(t_subj, mean_pred, color="black", linewidth=1.5, zorder=3, label="Model mean")
        ax.fill_between(t_subj, hdi_low, hdi_high, color="grey", alpha=0.3, zorder=1, label="94% HDI")
        ax.set_title(subj_name)
        ax.set_xlim(-0.5, t_max)

        # Observed LZc on secondary y-axis
        mask_obs = data["subject_idx"] == i
        t_obs = data["t_model"][mask_obs]
        lzc_obs = data["lzc"][mask_obs]

        ax2 = ax.twinx()
        twin_axes[i] = ax2
        ax2.scatter(t_obs, lzc_obs, s=8, color="purple", alpha=0.5, zorder=2,
                    label="LZc observed")

        # Red DMT plasma points on a hidden third axis (no ticks/labels)
        subj_plasma = plasma_df[plasma_df["subject"] == subj_name]
        if not subj_plasma.empty:
            ax3 = ax.twinx()
            ax3.spines["right"].set_visible(False)
            ax3.set_yticks([])
            ax3.scatter(
                subj_plasma["time_min"], subj_plasma["plasma_conc"],
                s=20, color="red", marker="x", zorder=4, label="Plasma DMT",
            )

        # Legend on first subplot (combine all axes)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            handles += h2
            labels += l2
            if not subj_plasma.empty:
                h3, l3 = ax3.get_legend_handles_labels()
                handles += h3
                labels += l3
            ax.legend(handles, labels, fontsize=6, loc="upper right")

    for j in range(n_subjects, len(axes)):
        axes[j].set_visible(False)

    for ax in axes[n_cols * (n_rows - 1):n_cols * n_rows]:
        if ax.get_visible():
            ax.set_xlabel("Fake time (min)")
    for r in range(n_rows):
        axes[r * n_cols].set_ylabel("y2 (model units)")
        right_idx = r * n_cols + n_cols - 1
        if right_idx in twin_axes:
            twin_axes[right_idx].set_ylabel("LZc (observed)", fontsize=8)

    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_predictions_y2(trace, data, output_path_curve, output_path_predictive,
                        t_max=18.0, n_grid=200):
    """Generate two y2 plots: posterior curves and posterior predictive (with noise)."""
    subject_names = data["subject_names"]
    n_subjects = len(subject_names)
    plasma_df = _load_plasma_df()

    t_fine = np.linspace(0.01, t_max, n_grid)
    t_grid = np.tile(t_fine, n_subjects)
    subject_idx_grid = np.repeat(np.arange(n_subjects), n_grid)

    curve_samples, predictive_samples = compute_posterior_predictive(
        trace, t_grid, subject_idx_grid, n_subjects
    )

    _plot_y2(curve_samples, subject_names, subject_idx_grid, t_grid,
             data, plasma_df, "Brain LZc (y2) — Posterior Curves",
             output_path_curve, t_max)

    _plot_y2(predictive_samples, subject_names, subject_idx_grid, t_grid,
             data, plasma_df, "Brain LZc (y2) — Posterior Predictive",
             output_path_predictive, t_max)
