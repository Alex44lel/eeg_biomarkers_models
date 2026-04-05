"""
Run the LZc computation and plotting pipeline.

Run from the project root:
    python -m src.run_lzc_pipeline
"""
from src.lzc.compute_lzc import main as compute_lzc
from src.lzc.plot_lzc import main as plot_lzc


if __name__ == "__main__":
    print("=" * 60)
    print("Step 1/2 — Compute LZc from EEG data")
    print("=" * 60)
    compute_lzc()

    print("\n" + "=" * 60)
    print("Step 2/2 — Plot LZc + Plasma")
    print("=" * 60)
    plot_lzc()

    print("\nDone. Results written to results/lzc/")
