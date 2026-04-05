import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from .partial_pooling_model import compute_posterior_predictive


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

    # Compute posterior predictions
    predictions = compute_posterior_predictive(
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

    # Shared labels
    for ax in axes[n_cols * (n_rows - 1):n_cols * n_rows]:
        if ax.get_visible():
            ax.set_xlabel("Time post-injection (min)")
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
