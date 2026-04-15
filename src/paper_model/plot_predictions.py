import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from pathlib import Path
from .partial_pooling_model import (
    compute_posterior_predictive, compute_posterior_predictive_y1,
    compute_posterior_predictive_y2_raw,
)


def plot_predictions(trace, data, output_path, t_max=20.0, n_grid=200):
    """Plot per-subject model predictions vs observed data (Figure 26 replica).

    Args:
        trace: ArviZ InferenceData with posterior samples.
        data: dict from prepare_data.load_and_prepare().
        output_path: Path to save the figure.
        t_max: maximum time on x-axis (minutes since injection).
        n_grid: number of points in the fine time grid per subject.
    """
    subject_names = data["subject_names"]
    n_subjects = len(subject_names)

    # Build a fine time grid for each subject (t=0 is injection)
    t_fine = np.linspace(0, t_max, n_grid)
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
        ax.set_xlim(0, t_max)
        ax.set_ylim(0, 3.2)

    # Hide unused subplots
    for j in range(n_subjects, len(axes)):
        axes[j].set_visible(False)

    # Show tick labels on all visible subplots
    for ax in axes[:n_subjects]:
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("Time since injection (min)")
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
                    t_max=20.0):
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
        ax.set_xlim(0, t_max)

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

    for ax in axes[:n_subjects]:
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("Time since injection (min)")
    for r in range(n_rows):
        axes[r * n_cols].set_ylabel(ylabel)

    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def _plot_y1_ngml(predictions, subject_names, subject_idx_grid, t_grid,
                  plasma_df, title, output_path, t_max=20.0):
    """Plot y1 model predictions and plasma data on same axis (ng/mL)."""
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

        # Plasma data on same axis
        subj_plasma = plasma_df[plasma_df["subject"] == subj_name]
        if not subj_plasma.empty:
            ax.scatter(subj_plasma["time_min"], subj_plasma["plasma_conc"],
                       s=30, color="red", marker="x", zorder=4, label="Plasma observed")

        ax.set_title(subj_name)
        ax.set_xlim(0, t_max)

        if i == 0:
            ax.legend(fontsize=6, loc="upper right")

    for j in range(n_subjects, len(axes)):
        axes[j].set_visible(False)

    for ax in axes[:n_subjects]:
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("Time since injection (min)")
    for r in range(n_rows):
        axes[r * n_cols].set_ylabel("Plasma DMT (ng/mL)")

    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def _load_plasma_df():
    """Load and filter plasma DMT data for plotting (t=0 = injection)."""
    base = Path(__file__).resolve().parents[2]
    plasma_df = pd.read_csv(base / "data" / "plasma_clean.csv")
    plasma_df = plasma_df[plasma_df["condition"] == "dmt"]
    # time_min is already relative to injection (t=0 = injection)
    return plasma_df


def plot_predictions_y1(trace, data, output_path_curve, output_path_predictive,
                        t_max=20.0, n_grid=200):
    """Generate two y1 plots: posterior curves and posterior predictive (with noise)."""
    subject_names = data["subject_names"]
    n_subjects = len(subject_names)
    plasma_df = _load_plasma_df()

    t_fine = np.linspace(0, t_max, n_grid)
    t_grid = np.tile(t_fine, n_subjects)
    subject_idx_grid = np.repeat(np.arange(n_subjects), n_grid)

    curve_samples, predictive_samples = compute_posterior_predictive_y1(
        trace, t_grid, subject_idx_grid, n_subjects
    )

    _plot_y1_ngml(curve_samples, subject_names, subject_idx_grid, t_grid, plasma_df,
                  "Plasma DMT (y1) — Posterior Curves", output_path_curve, t_max=t_max)

    _plot_y1_ngml(predictive_samples, subject_names, subject_idx_grid, t_grid, plasma_df,
                  "Plasma DMT (y1) — Posterior Predictive", output_path_predictive, t_max=t_max)


def plot_predictions_y2(trace, data, output_path_curve, output_path_predictive,
                        t_max=20.0, n_grid=200):
    """Generate two y2 plots: brain drug concentration in ng/mL."""
    subject_names = data["subject_names"]
    n_subjects = len(subject_names)

    t_fine = np.linspace(0, t_max, n_grid)
    t_grid = np.tile(t_fine, n_subjects)
    subject_idx_grid = np.repeat(np.arange(n_subjects), n_grid)

    y2_raw = compute_posterior_predictive_y2_raw(
        trace, t_grid, subject_idx_grid, n_subjects
    )

    for preds, title, output_path in [
        (y2_raw, "Brain DMT (y2) — Posterior Curves", output_path_curve),
        (y2_raw, "Brain DMT (y2) — Posterior Predictive", output_path_predictive),
    ]:
        n_cols = 3
        n_rows = 4
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12), sharex=True)
        axes = axes.flatten()

        for i, subj_name in enumerate(subject_names):
            ax = axes[i]
            mask_grid = subject_idx_grid == i
            t_subj = t_grid[mask_grid]
            preds_subj = preds[:, mask_grid]

            mean_pred = preds_subj.mean(axis=0)
            hdi_bounds = az.hdi(preds_subj, hdi_prob=0.94)

            ax.plot(t_subj, mean_pred, color="black", linewidth=1.5, zorder=3, label="Model mean")
            ax.fill_between(t_subj, hdi_bounds[:, 0], hdi_bounds[:, 1],
                            color="grey", alpha=0.3, zorder=1, label="94% HDI")
            ax.set_title(subj_name)
            ax.set_xlim(0, t_max)

            if i == 0:
                ax.legend(fontsize=6, loc="upper right")

        for j in range(n_subjects, len(axes)):
            axes[j].set_visible(False)
        for ax in axes[:n_subjects]:
            ax.tick_params(labelbottom=True)
            ax.set_xlabel("Time since injection (min)")
        for r in range(n_rows):
            axes[r * n_cols].set_ylabel("Brain DMT (ng/mL)")

        fig.suptitle(title, fontsize=14, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {output_path}")


def plot_predictions_lzc(trace, data, output_path_curve, output_path_predictive,
                         t_max=20.0, n_grid=200):
    """Plot predicted LZc (Hill equation output) vs observed LZc per subject."""
    subject_names = data["subject_names"]
    n_subjects = len(subject_names)

    t_fine = np.linspace(0, t_max, n_grid)
    t_grid = np.tile(t_fine, n_subjects)
    subject_idx_grid = np.repeat(np.arange(n_subjects), n_grid)

    curve_samples, predictive_samples = compute_posterior_predictive(
        trace, t_grid, subject_idx_grid, n_subjects
    )

    for preds, title, output_path in [
        (curve_samples, "Predicted LZc (Hill) — Posterior Curves", output_path_curve),
        (predictive_samples, "Predicted LZc (Hill) — Posterior Predictive", output_path_predictive),
    ]:
        n_cols = 3
        n_rows = 4
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12), sharex=True)
        axes = axes.flatten()

        for i, subj_name in enumerate(subject_names):
            ax = axes[i]

            # Observed LZc
            mask_obs = data["subject_idx"] == i
            t_obs = data["t_model"][mask_obs]
            lzc_obs = data["lzc"][mask_obs]
            ax.scatter(t_obs, lzc_obs, s=8, color="purple", alpha=0.5, zorder=2,
                       label="LZc observed")

            # Model predictions
            mask_grid = subject_idx_grid == i
            t_subj = t_grid[mask_grid]
            preds_subj = preds[:, mask_grid]

            mean_pred = preds_subj.mean(axis=0)
            hdi_bounds = az.hdi(preds_subj, hdi_prob=0.94)

            ax.plot(t_subj, mean_pred, color="black", linewidth=1.5, zorder=3,
                    label="Model mean")
            ax.fill_between(t_subj, hdi_bounds[:, 0], hdi_bounds[:, 1],
                            color="grey", alpha=0.3, zorder=1, label="94% HDI")
            ax.set_title(subj_name)
            ax.set_xlim(0, t_max)
            ax.set_ylim(0, 1.05)

            if i == 0:
                ax.legend(fontsize=6, loc="upper right")

        for j in range(n_subjects, len(axes)):
            axes[j].set_visible(False)
        for ax in axes[:n_subjects]:
            ax.tick_params(labelbottom=True)
            ax.set_xlabel("Time since injection (min)")
        for r in range(n_rows):
            axes[r * n_cols].set_ylabel("LZc")

        fig.suptitle(title, fontsize=14, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {output_path}")
