import numpy as np
import matplotlib.pyplot as plt


def _fmt(x):
    """Format a number: use scientific notation for very small/large values."""
    if x == 0:
        return "0"
    if abs(x) < 0.001 or abs(x) >= 10000:
        return f"{x:.3e}"
    return f"{x:.4f}"


def plot_parameter_table(trace, data, output_path):
    """Save an image with a summary table of key parameter means and variances.

    Per-subject parameters show the mean and variance of the per-subject
    posterior means (i.e. across-subject variability).
    Global parameters show the posterior mean and posterior variance.

    Args:
        trace: ArviZ InferenceData with posterior samples.
        data: dict from prepare_data.load_and_prepare().
        output_path: Path to save the table image.
    """
    n_subjects = len(data["subject_names"])

    per_subject_params = ["k0", "k1", "k2", "y_init"]
    global_params = ["EC50", "n_hill", "lz_sigma", "plasma_sigma"]

    rows = []

    # Section: per-subject parameters
    rows.append(["Per-subject parameters", "", ""])
    for param in per_subject_params:
        if param not in trace.posterior:
            continue
        samples = trace.posterior[param].values.reshape(-1, n_subjects)
        subject_means = samples.mean(axis=0)
        rows.append([f"  {param}", _fmt(subject_means.mean()),
                     _fmt(subject_means.var())])

    # Section: global parameters
    rows.append(["Global parameters", "", ""])
    for param in global_params:
        if param not in trace.posterior:
            continue
        samples = trace.posterior[param].values.reshape(-1)
        rows.append([f"  {param}", _fmt(samples.mean()), _fmt(samples.var())])

    n_cols = 3
    fig, ax = plt.subplots(figsize=(7, 0.5 + 0.4 * len(rows)))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=["Parameter", "Mean", "Variance"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    # Style header row
    for j in range(n_cols):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Style section headers and data rows
    for i, row in enumerate(rows, start=1):
        is_section = row[1] == ""
        for j in range(n_cols):
            if is_section:
                table[i, j].set_facecolor("#B4C7E7")
                table[i, j].set_text_props(fontweight="bold")
            else:
                color = "#D9E2F3" if i % 2 == 0 else "white"
                table[i, j].set_facecolor(color)

    fig.suptitle("Key Parameter Summary (posterior)", fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved parameter table to {output_path}")


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
