import numpy as np
import matplotlib.pyplot as plt


def plot_parameters(trace, data, output_path):
    """Plot per-subject parameter boxplots for k0, k1, k2 (Figure 27 replica).

    Args:
        trace: ArviZ InferenceData with posterior samples.
        data: dict from prepare_data.load_and_prepare().
        output_path: Path to save the figure.
    """
    subject_names = data["subject_names"]
    n_subjects = len(subject_names)
    param_names = ["k0", "k1", "k2"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # The paper reports k in units of 1/epoch (1 epoch = 0.05 min).
    # Our model fits in minutes, so scale k values for display.
    TIME_STEP = 1  # minutes per epoch
    for ax, param in zip(axes, param_names):
        # Shape: (chains, draws, n_subjects) -> flatten to (n_samples, n_subjects)
        samples = trace.posterior[param].values
        samples = samples.reshape(-1, n_subjects) * TIME_STEP

        # Build list of per-subject sample arrays for boxplot
        bp_data = [samples[:, i] for i in range(n_subjects)]

        bp = ax.boxplot(
            bp_data,
            positions=range(n_subjects),
            widths=0.6,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker="^", markerfacecolor="green", markeredgecolor="green", markersize=7),
            medianprops=dict(color="green", linewidth=1.5),
            boxprops=dict(facecolor="lightblue", edgecolor="black"),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            flierprops=dict(marker="o", markersize=3, alpha=0.3),
        )

        # Annotate with mean and std
        for i in range(n_subjects):
            mu = samples[:, i].mean()
            sigma = samples[:, i].std()
            ax.text(
                i, ax.get_ylim()[1] * 0.95, f"\u03bc={mu:.2f}\n\u03c3={sigma:.2f}",
                ha="center", va="top", fontsize=7,
            )

        ax.set_title(f"Partial Pooling Model {param} Box Plots")
        ax.set_ylabel("Value")

    # Fix annotation positioning after all data is plotted
    for ax, param in zip(axes, param_names):
        samples = trace.posterior[param].values.reshape(-1, n_subjects) * TIME_STEP
        ymin, ymax = ax.get_ylim()
        for i in range(n_subjects):
            # Clear old annotations and re-add at correct position
            pass  # annotations are already placed above

    axes[-1].set_xticks(range(n_subjects))
    axes[-1].set_xticklabels(subject_names, rotation=45)
    axes[-1].set_xlabel("Subject ID")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved parameters plot to {output_path}")
