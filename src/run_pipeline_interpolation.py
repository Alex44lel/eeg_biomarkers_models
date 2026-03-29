"""
End-to-end pipeline: inference → interpolation → plots.

Run from the project root:
    python -m src.run_pipeline_interpolation
"""
from src.interpolation.run_inference import main as run_inference
from src.interpolation.interpolation import main as run_interpolation
from src.interpolation.plot_results import main as run_plots


if __name__ == "__main__":
    print("=" * 60)
    print("Step 1/3 — MCMC sampling")
    print("=" * 60)
    run_inference()

    print("\n" + "=" * 60)
    print("Step 2/3 — Posterior predictive interpolation")
    print("=" * 60)
    run_interpolation()

    print("\n" + "=" * 60)
    print("Step 3/3 — Plots")
    print("=" * 60)
    run_plots()

    print("\nDone. Results written to results/")
