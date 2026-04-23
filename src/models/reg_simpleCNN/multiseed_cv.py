"""
Multi-seed LOSO cross-validation for the SimpleCNN apr19 best config (+0.358).

Runs the same LOSO sweep N times with different seeds to get honest error bars
on cv_mean_val_r2 and per-fold stability. Reuses run_fold from train_cv.

MLflow layout:
    parent run   (one per invocation, groups the whole multi-seed experiment)
      seed_00  (child, seed=S0)
        fold_01_val_S01  (grandchild)
        ...
      seed_01  (child, seed=S1)
        ...

Parent-level metrics:
    seed_{S}_cv_mean_val_r2, seed_{S}_cv_std_val_r2, seed_{S}_pooled_val_r2, ...
    across_seed_mean_cv_r2, across_seed_std_cv_r2, across_seed_min_cv_r2, ...
    per_fold_{subject}_mean_r2, per_fold_{subject}_std_r2   (stability)
    per_fold_{subject}_pct_positive                         (% of seeds with r²>0)

Artifacts on the parent run:
    predictions_all_seeds.npz  (y_true, y_pred, times, subject, seed for every trial)
    per_subject_r2_stability.png
    across_seed_r2_summary.png

Usage:
    python -m src.models.reg_simpleCNN.multiseed_cv \
        --seeds 42 123 7 2024 0 \
        --run_name multiseed_apr19_best
"""

import argparse
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from mlflow.tracking import MlflowClient

from .dataset import ALL_SUBJECTS, DATASET_PATHS, EEGDataset
from .train_cv import compute_regression_metrics, run_fold

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-seed LOSO CV for SimpleCNN apr19 best config"
    )
    # Seeds — either explicit list or N seeds drawn from a base
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[42, 123, 7, 2024, 0],
                   help="List of seeds to run. Default: 5 seeds.")
    # Training hyperparams — defaults match the +0.358 config (run55 / 149453d)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--loss", type=str, default="smoothl1",
                   choices=["mse", "smoothl1"])
    p.add_argument("--huber_beta", type=float, default=10.0)
    p.add_argument("--mixup_alpha", type=float, default=0.0)
    p.add_argument("--k1", type=int, default=15)
    p.add_argument("--k2", type=int, default=7)
    p.add_argument("--k3", type=int, default=7)
    p.add_argument("--description", type=str, default="")
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--dataset", type=str, default="pk",
                   choices=sorted(DATASET_PATHS.keys()))
    p.add_argument("--experiment_name", type=str,
                   default="SimpleCNN_DMT_regression_CV")
    p.add_argument("--run_name", type=str, default="multiseed_apr19_best")
    p.add_argument("--log_model", action="store_true")
    return p.parse_args()


def set_seed(seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def plot_per_subject_stability(per_subject, out_path):
    """Box/strip plot: for each subject, distribution of val_r2 across seeds."""
    subjects = sorted(per_subject.keys())
    data = [per_subject[s] for s in subjects]
    means = [float(np.mean(v)) for v in data]

    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(subjects)), 5))
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    bp = ax.boxplot(data, tick_labels=subjects, widths=0.55,
                    patch_artist=True, showfliers=False)
    for box in bp["boxes"]:
        box.set(facecolor="lightsteelblue", edgecolor="navy")
    for med in bp["medians"]:
        med.set(color="navy", linewidth=1.5)
    for i, vals in enumerate(data, start=1):
        x_jit = np.random.normal(i, 0.05, size=len(vals))
        ax.scatter(x_jit, vals, color="darkorange", s=20, alpha=0.85, zorder=3)
    for i, m in enumerate(means, start=1):
        ax.scatter([i], [m], color="red", marker="D", s=35, zorder=4,
                   label=("mean" if i == 1 else None))
    ax.set_xlabel("Held-out subject")
    ax.set_ylabel("val R²")
    ax.set_title(f"Per-subject val R² across {len(data[0])} seeds")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_across_seed_summary(seed_results, out_path):
    """Bar plot of cv_mean_val_r2 per seed with horizontal mean±std band."""
    seeds = [r["seed"] for r in seed_results]
    cv_mean_r2 = [r["cv_mean_val_r2"] for r in seed_results]
    pooled_r2 = [r["pooled_val_r2"] for r in seed_results]

    mean = float(np.mean(cv_mean_r2))
    std = float(np.std(cv_mean_r2))

    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(seeds)), 5))
    x = np.arange(len(seeds))
    ax.bar(x - 0.2, cv_mean_r2, width=0.4, color="steelblue",
           label="cv_mean_val_r² (across 8 folds)")
    ax.bar(x + 0.2, pooled_r2, width=0.4, color="darkorange",
           label="pooled_val_r²")
    ax.axhline(mean, color="navy", linestyle="--", linewidth=1.2,
               label=f"cv_mean_val_r² mean = {mean:+.3f}")
    ax.axhspan(mean - std, mean + std, color="navy", alpha=0.12,
               label=f"±1σ (σ = {std:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Seed")
    ax.set_ylabel("R²")
    ax.set_title(f"Across-seed stability of the +0.358 config ({len(seeds)} seeds)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cv_subjects = args.subjects if args.subjects else ALL_SUBJECTS
    n_folds = len(cv_subjects)

    print("=" * 70)
    print("  SimpleCNN DMT Plasma Regression  |  MULTI-SEED LOSO CV")
    print("=" * 70)
    print(f"Device:        {device}")
    print(f"Dataset:       {args.dataset}")
    print(f"Subjects ({n_folds}): {cv_subjects}")
    print(f"Seeds ({len(args.seeds)}): {args.seeds}")
    print(f"Hyperparams:   lr={args.lr}  bs={args.batch_size}  "
          f"dropout={args.dropout}  wd={args.weight_decay}")
    print(f"               loss={args.loss}  huber_beta={args.huber_beta}  "
          f"mixup_alpha={args.mixup_alpha}")
    print(f"Training:      max_epochs={args.epochs}  patience={args.patience}")
    print("=" * 70, flush=True)

    mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))
    mlflow.set_experiment(args.experiment_name)

    t_global = time.time()
    mlf_client = MlflowClient()
    shared_hparams = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "n_folds": n_folds,
        "cv_subjects": ",".join(cv_subjects),
        "device": str(device),
        "task": "regression",
        "cv_scheme": "leave-one-subject-out",
        "early_stop_metric": "val_r2",
        "early_stop_direction": "maximize",
        "dataset": args.dataset,
        "loss": args.loss,
        "huber_beta": args.huber_beta,
        "mixup_alpha": args.mixup_alpha,
        "multiseed": "true",
        "n_seeds": len(args.seeds),
        "seeds": ",".join(str(s) for s in args.seeds),
    }

    with mlflow.start_run(run_name=args.run_name) as parent_run:
        parent_run_id = parent_run.info.run_id
        mlflow.log_params(shared_hparams)
        mlflow.set_tag("multiseed_group", args.run_name)

        # seed -> list of fold-result dicts (across all 8 subjects)
        all_seed_results = []
        # subject -> list of r² values (one per seed)
        per_subject_r2 = {s: [] for s in cv_subjects}
        # rows for the big npz: (seed, subject, time, y_true, y_pred)
        npz_seed, npz_subj, npz_time, npz_true, npz_pred = [], [], [], [], []

        for seed_idx, seed in enumerate(args.seeds):
            set_seed(seed, device)
            print("\n" + "#" * 70)
            print(f"#  SEED {seed_idx + 1}/{len(args.seeds)}  (seed = {seed})")
            print("#" * 70, flush=True)

            t_seed = time.time()
            args.seed = seed  # run_fold / logs pick up args.seed via the namespace

            with mlflow.start_run(nested=True,
                                  run_name=f"seed_{seed_idx:02d}_s{seed}") as seed_run:
                seed_run_id = seed_run.info.run_id
                mlflow.log_params({**shared_hparams, "seed": seed})

                fold_results = []
                for i, subj in enumerate(cv_subjects, start=1):
                    t_fold = time.time()
                    with mlflow.start_run(nested=True,
                                          run_name=f"fold_{i:02d}_val_{subj}"):
                        mlflow.log_params({**shared_hparams, "seed": seed})
                        res = run_fold(
                            args, subj, i, n_folds, device,
                            parent_run_id=seed_run_id,
                            mlf_client=mlf_client,
                            cv_subjects=cv_subjects,
                        )
                        fold_results.append(res)

                    per_subject_r2[subj].append(res["best_val_r2"])
                    # mirror scalar on seed-level run for easy MLflow UI table
                    mlf_client.log_metric(seed_run_id,
                                          f"fold_{subj}_best_val_r2",
                                          res["best_val_r2"])
                    mlf_client.log_metric(seed_run_id,
                                          f"fold_{subj}_best_val_mae",
                                          res["best_val_mae"])
                    mlf_client.log_metric(seed_run_id,
                                          f"fold_{subj}_best_val_rmse",
                                          res["best_val_rmse"])
                    mlf_client.log_metric(seed_run_id,
                                          f"fold_{subj}_best_epoch",
                                          res["best_epoch"])

                    # stash predictions for the npz
                    # val_loader uses shuffle=False, so y_true/y_pred order
                    # matches EEGDataset order → times align.
                    n = len(res["y_true"])
                    val_times = EEGDataset(subjects=[subj],
                                           dataset=args.dataset).times
                    npz_seed.extend([seed] * n)
                    npz_subj.extend([subj] * n)
                    npz_time.extend(list(val_times))
                    npz_true.extend(list(res["y_true"]))
                    npz_pred.extend(list(res["y_pred"]))

                    print(f"  >>> seed {seed} fold {i}/{n_folds} ({subj}) in "
                          f"{time.time() - t_fold:.1f}s  "
                          f"val_r2={res['best_val_r2']:+.4f}", flush=True)

                # --- seed-level aggregate (across the 8 folds) ---
                r2s = np.array([r["best_val_r2"] for r in fold_results])
                maes = np.array([r["best_val_mae"] for r in fold_results])
                rmses = np.array([r["best_val_rmse"] for r in fold_results])
                y_true_all = np.concatenate([r["y_true"] for r in fold_results])
                y_pred_all = np.concatenate([r["y_pred"] for r in fold_results])
                pooled = compute_regression_metrics(y_true_all, y_pred_all)

                seed_summary = {
                    "cv_mean_val_r2":  float(r2s.mean()),
                    "cv_std_val_r2":   float(r2s.std()),
                    "cv_min_val_r2":   float(r2s.min()),
                    "cv_max_val_r2":   float(r2s.max()),
                    "cv_mean_val_mae": float(maes.mean()),
                    "cv_mean_val_rmse": float(rmses.mean()),
                    "pooled_val_r2":   pooled["r2"],
                    "pooled_val_mae":  pooled["mae"],
                    "pooled_val_rmse": pooled["rmse"],
                }
                mlflow.log_metrics(seed_summary)

                # mirror seed aggregates onto parent for easy comparison
                for k, v in seed_summary.items():
                    mlf_client.log_metric(parent_run_id, f"seed_{seed}_{k}", v)

                seed_summary["seed"] = seed
                seed_summary["fold_results"] = fold_results
                all_seed_results.append(seed_summary)

                print(f"\n  Seed {seed}: cv_mean_val_r2={seed_summary['cv_mean_val_r2']:+.4f}"
                      f" ± {seed_summary['cv_std_val_r2']:.4f} "
                      f"(pooled {seed_summary['pooled_val_r2']:+.4f})  "
                      f"[{time.time() - t_seed:.1f}s]", flush=True)

        # ==================================================================
        #  Cross-seed aggregates
        # ==================================================================
        cv_means = np.array([r["cv_mean_val_r2"] for r in all_seed_results])
        pooled_r2s = np.array([r["pooled_val_r2"] for r in all_seed_results])
        cv_stds = np.array([r["cv_std_val_r2"] for r in all_seed_results])

        print("\n" + "=" * 70)
        print("  ACROSS-SEED SUMMARY (+0.358 config)")
        print("=" * 70)
        print(f"{'seed':>8} {'cv_mean_r2':>12} {'cv_std_r2':>11} "
              f"{'pooled_r2':>11} {'cv_min':>9} {'cv_max':>9}")
        print("-" * 65)
        for r in all_seed_results:
            print(f"{r['seed']:>8} {r['cv_mean_val_r2']:>+12.4f} "
                  f"{r['cv_std_val_r2']:>11.4f} {r['pooled_val_r2']:>+11.4f} "
                  f"{r['cv_min_val_r2']:>+9.4f} {r['cv_max_val_r2']:>+9.4f}")
        print("-" * 65)
        print(f"{'MEAN':>8} {cv_means.mean():>+12.4f} {cv_stds.mean():>11.4f} "
              f"{pooled_r2s.mean():>+11.4f}")
        print(f"{'STD':>8} {cv_means.std():>12.4f} "
              f"{'':11} {pooled_r2s.std():>11.4f}")
        print(f"{'MIN':>8} {cv_means.min():>+12.4f} "
              f"{'':11} {pooled_r2s.min():>+11.4f}")
        print(f"{'MAX':>8} {cv_means.max():>+12.4f} "
              f"{'':11} {pooled_r2s.max():>+11.4f}")

        across_metrics = {
            "across_seed_mean_cv_r2": float(cv_means.mean()),
            "across_seed_std_cv_r2":  float(cv_means.std()),
            "across_seed_min_cv_r2":  float(cv_means.min()),
            "across_seed_max_cv_r2":  float(cv_means.max()),
            "across_seed_mean_pooled_r2": float(pooled_r2s.mean()),
            "across_seed_std_pooled_r2":  float(pooled_r2s.std()),
            "across_seed_min_pooled_r2":  float(pooled_r2s.min()),
            "across_seed_max_pooled_r2":  float(pooled_r2s.max()),
            "across_seed_mean_cv_std":    float(cv_stds.mean()),
        }
        mlflow.log_metrics(across_metrics)

        # per-subject stability across seeds
        print("\n" + "=" * 70)
        print("  PER-SUBJECT STABILITY (val R² across seeds)")
        print("=" * 70)
        print(f"{'subject':>8} {'mean_r2':>10} {'std_r2':>10} "
              f"{'min_r2':>10} {'max_r2':>10} {'pct_pos':>8}")
        print("-" * 60)
        for subj in cv_subjects:
            vals = np.array(per_subject_r2[subj])
            mean = float(vals.mean())
            std = float(vals.std())
            mn = float(vals.min())
            mx = float(vals.max())
            pct_pos = float((vals > 0).mean() * 100)
            print(f"{subj:>8} {mean:>+10.4f} {std:>10.4f} "
                  f"{mn:>+10.4f} {mx:>+10.4f} {pct_pos:>7.0f}%")
            mlflow.log_metrics({
                f"per_fold_{subj}_mean_r2": mean,
                f"per_fold_{subj}_std_r2": std,
                f"per_fold_{subj}_min_r2": mn,
                f"per_fold_{subj}_max_r2": mx,
                f"per_fold_{subj}_pct_positive": pct_pos,
            })

        # ==================================================================
        #  Artifacts
        # ==================================================================
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)

            npz_path = tdir / "predictions_all_seeds.npz"
            np.savez(
                npz_path,
                seeds=np.array(npz_seed, dtype=np.int64),
                subjects=np.array(npz_subj),
                times=np.array(npz_time, dtype=np.float32),
                y_true=np.array(npz_true, dtype=np.float32),
                y_pred=np.array(npz_pred, dtype=np.float32),
            )
            mlflow.log_artifact(str(npz_path))

            stab_path = tdir / "per_subject_r2_stability.png"
            plot_per_subject_stability(per_subject_r2, stab_path)
            mlflow.log_artifact(str(stab_path))

            seed_path = tdir / "across_seed_r2_summary.png"
            plot_across_seed_summary(all_seed_results, seed_path)
            mlflow.log_artifact(str(seed_path))

        # ==================================================================
        #  Machine-parseable one-line summary
        # ==================================================================
        print("\n" + "=" * 70)
        print("  RESULT (multi-seed):")
        print(
            f"  RESULT "
            f"across_seed_mean_cv_r2={across_metrics['across_seed_mean_cv_r2']:+.4f} "
            f"across_seed_std_cv_r2={across_metrics['across_seed_std_cv_r2']:.4f} "
            f"across_seed_min_cv_r2={across_metrics['across_seed_min_cv_r2']:+.4f} "
            f"across_seed_max_cv_r2={across_metrics['across_seed_max_cv_r2']:+.4f} "
            f"across_seed_mean_pooled_r2={across_metrics['across_seed_mean_pooled_r2']:+.4f} "
            f"n_seeds={len(args.seeds)}"
        )
        print("=" * 70)
        print(f"  Total wall time: {time.time() - t_global:.1f}s")
        print(f"  Parent run id:   {parent_run_id}")
        print(f"  View:            mlflow ui --backend-store-uri mlruns/")
        print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
