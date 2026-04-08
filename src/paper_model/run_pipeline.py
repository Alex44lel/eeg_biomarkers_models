"""
Two-compartment partial pooling PK model pipeline.

Usage:
    python -m src.paper_model.run_pipeline [--draws 1000] [--chains 2] [--load-trace path]
"""

import argparse
from pathlib import Path

import arviz as az

from .prepare_data import load_and_prepare, load_plasma_data
from .partial_pooling_model import build_model, fit_model
from .plot_predictions import plot_predictions, plot_predictions_y1, plot_predictions_y2
from .plot_parameters import plot_parameters

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "paper_model"


def main():
    parser = argparse.ArgumentParser(description="Partial pooling two-compartment PK model")
    parser.add_argument("--draws", type=int, default=1000, help="MCMC draws per chain")
    parser.add_argument("--chains", type=int, default=2, help="Number of MCMC chains")
    parser.add_argument("--load-trace", type=str, default=None, help="Path to existing trace .nc file")
    parser.add_argument("--observe-plasma", action="store_true",
                        help="Use real plasma DMT concentrations as observed data for y1")
    parser.add_argument("--plasma-only", action="store_true",
                        help="Fit model using only plasma DMT data (no LZc likelihood)")
    parser.add_argument("--only-after-injection", action="store_true",
                        help="Restrict LZc data to post-injection samples only")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Prepare data
    print("Loading and preparing data...")
    data = load_and_prepare()
    full_data = data  # keep unfiltered copy for plotting
    print(f"  {len(data['subject_names'])} subjects, {len(data['t_model'])} observations")

    # Filter to post-injection only if requested (for model fitting only)
    if args.only_after_injection:
        import numpy as np
        inj_times = np.array(data["injection_times"])
        mask = data["t_model"] >= inj_times[data["subject_idx"]]
        data = {
            **data,
            "t_model": data["t_model"][mask],
            "lzc": data["lzc"][mask],
            "subject_idx": data["subject_idx"][mask],
        }
        print(f"  After filtering to post-injection: {mask.sum()} / {len(mask)} observations kept")

    # Load plasma data if requested
    plasma_data = None
    observe_lzc = True
    if args.plasma_only:
        print("Loading plasma DMT data (plasma-only mode, no LZc likelihood)...")
        plasma_data = load_plasma_data(data["subject_names"])
        observe_lzc = False
        print(f"  {len(plasma_data['plasma_conc'])} plasma observations")
    elif args.observe_plasma:
        print("Loading plasma DMT data as observed variables...")
        plasma_data = load_plasma_data(data["subject_names"])
        print(f"  {len(plasma_data['plasma_conc'])} plasma observations")

    # Step 2: Build and fit model (or load existing trace)
    if args.load_trace:
        print(f"Loading trace from {args.load_trace}...")
        trace = az.from_netcdf(args.load_trace)
    else:
        print(f"Building model and sampling ({args.draws} draws, {args.chains} chains)...")
        model = build_model(
            data["t_model"], data["lzc"], data["subject_idx"],
            len(data["subject_names"]),
            plasma_data=plasma_data,
            observe_lzc=observe_lzc,
        )
        trace = fit_model(model, draws=args.draws, chains=args.chains)

        # Save trace
        trace_path = RESULTS_DIR / "partial_pooling_trace.nc"
        trace.to_netcdf(str(trace_path))
        print(f"  Trace saved to {trace_path}")

    # Step 3: Plot predictions (Figure 26) — use full_data so pre-injection points appear
    print("Generating predictions plot...")
    plot_predictions(trace, full_data, RESULTS_DIR / "partial_pooling_predictions.png")

    # Step 3b: Plot y1 (plasma DMT) — curves and predictive
    print("Generating y1 (plasma DMT) plots...")
    plot_predictions_y1(
        trace, full_data,
        RESULTS_DIR / "partial_pooling_y1_curves.png",
        RESULTS_DIR / "partial_pooling_y1_predictive.png",
    )

    # Step 3c: Plot y2 (brain LZc) — curves and predictive with plasma overlay
    print("Generating y2 (brain LZc) dual-axis plots...")
    plot_predictions_y2(
        trace, full_data,
        RESULTS_DIR / "partial_pooling_y2_curves.png",
        RESULTS_DIR / "partial_pooling_y2_predictive.png",
    )

    # Step 4: Plot parameters (Figure 27)
    print("Generating parameters plot...")
    plot_parameters(trace, full_data, RESULTS_DIR / "partial_pooling_parameters.png")

    print("Done!")


if __name__ == "__main__":
    main()
