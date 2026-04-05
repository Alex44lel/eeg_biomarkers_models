"""
Plot LZc and DMT plasma concentration per subject.

Produces a grid of subplots (one per subject) with:
- Raw LZc values per 3-second trial (pink dots, left y-axis)
- DMT plasma concentration (black dots, right y-axis)
- Baseline reference line (red dashed at y=0)

This replicates Figure 10 from the reference paper.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[2]
    results_dir = project_root / "results" / "lzc"

    # Load LZc results
    lzc_df = pd.read_csv(results_dir / "lzc_results.csv")

    # Load injection time offsets (for aligning plasma times)
    offsets_df = pd.read_csv(results_dir / "injection_offsets.csv")
    offsets = dict(zip(offsets_df["subject"], offsets_df["injection_time_min"]))

    # Load plasma data (DMT condition only)
    plasma_df = pd.read_csv(project_root / "data" / "plasma_clean.csv")
    plasma_dmt = plasma_df[plasma_df["condition"] == "dmt"].copy()

    # All subjects from either dataset
    subjects_with_lzc = set(lzc_df["subject"].unique())
    all_plasma_subjects = set(plasma_dmt["subject"].unique())
    all_subjects = sorted(subjects_with_lzc | all_plasma_subjects)

    n_subjects = len(all_subjects)
    n_cols = 3
    n_rows = (n_subjects + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(14, 3.5 * n_rows),
        squeeze=False,
    )
    fig.suptitle(
        "LZc and DMT Plasma Concentration across Time, Per Subject",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    for idx, subj in enumerate(all_subjects):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        ax.set_title(subj, fontsize=11)
        ax.set_xlabel("Time (minutes)", fontsize=9)
        ax.set_ylabel("LZc", fontsize=9)
        ax.set_ylim(-5, 20)
        ax.set_xlim(0, 18)

        # Plot raw LZc (all 3-second trial values, no resampling)
        subj_lzc = lzc_df[lzc_df["subject"] == subj]
        if len(subj_lzc) > 0:
            ax.scatter(
                subj_lzc["time_min"],
                subj_lzc["lzc_normalized"],
                color="lightpink", s=8, alpha=0.6,
                edgecolors="none", zorder=2,
            )

        # Baseline reference line
        ax.axhline(y=0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)

        # Right axis: Plasma concentration (shifted by injection offset)
        ax2 = ax.twinx()
        ax2.set_ylabel("Plasma (nm)", fontsize=9)
        ax2.set_ylim(0, 600)

        subj_plasma = plasma_dmt[plasma_dmt["subject"] == subj]
        if len(subj_plasma) > 0:
            # Shift plasma times: plasma time_min is relative to injection,
            # LZc time is sequential trial time starting from baseline
            inj_offset = offsets.get(subj, 0)
            ax2.scatter(
                subj_plasma["time_min"] + inj_offset,
                subj_plasma["plasma_conc"],
                color="black", s=30, zorder=3,
            )

    # Hide unused subplots
    for idx in range(n_subjects, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lightpink",
               markersize=8, label="LZc(LHS)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               markersize=8, label="Plasma(RHS)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out_path = results_dir / "lzc_plasma_per_subject.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
