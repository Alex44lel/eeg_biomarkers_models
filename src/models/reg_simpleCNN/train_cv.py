"""
Leave-One-Subject-Out (LOSO) cross-validation for SimpleCNN EEG regression.

Each fold holds out one subject as the validation set while training on the
remaining subjects. All metrics (MAE, RMSE, MSE, R²) are reported per-fold,
aggregated (mean/std/min/max across folds) and pooled (concatenating every
held-out prediction). Early stopping uses val R² (higher is better).

Everything is logged to MLflow: one parent run with nested child runs per fold.
Designed for autoresearch pipelines — log output is verbose and structured so
it can be parsed to decide the next experiment.

Usage (run from project root):
    python -m src.models.reg_simpleCNN.train_cv \
        --lr 1e-3 --batch_size 64 --epochs 300 --patience 20
"""

import argparse
import time
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from .model import SimpleCNN
from .dataset import EEGDataset, ALL_SUBJECTS

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args():
    p = argparse.ArgumentParser(
        description="LOSO cross-validation for SimpleCNN DMT plasma regression"
    )
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=40,
                   help="Early-stop patience on val R² (higher is better)")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--loss", type=str, default="smoothl1",
                   choices=["mse", "smoothl1"],
                   help="Training loss. smoothl1 is Huber-like, robust to outliers.")
    p.add_argument("--huber_beta", type=float, default=10.0,
                   help="SmoothL1 beta (ng/mL). Errors smaller use L2, larger use L1.")
    p.add_argument("--mixup_alpha", type=float, default=0.0,
                   help="Mixup Beta(a,a) sampling. 0 disables.")
    p.add_argument("--subjects", nargs="+", default=None,
                   help="Restrict CV to these subject IDs (default: ALL_SUBJECTS)")
    p.add_argument("--experiment_name", type=str,
                   default="SimpleCNN_DMT_regression_CV")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_model", action="store_true",
                   help="If set, also log each fold's best PyTorch model to MLflow")
    return p.parse_args()


def compute_regression_metrics(y_true, y_pred):
    """MAE, RMSE, MSE, R²."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "mse": mse, "r2": r2}


def plot_predicted_vs_actual(y_true, y_pred, title, metrics, artifact_subdir,
                              extra_targets=None):
    """Scatter predicted vs actual. Logs to active MLflow run and optionally to
    extra (client, run_id, artifact_subdir) targets (e.g. parent run)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=15, c="steelblue")
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual plasma DMT (ng/mL)")
    ax.set_ylabel("Predicted plasma DMT (ng/mL)")
    ax.set_title(f"{title}\n"
                 f"MAE={metrics['mae']:.1f}  RMSE={metrics['rmse']:.1f}  "
                 f"R²={metrics['r2']:.3f}")
    ax.legend()
    plt.tight_layout()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        fig.savefig(f.name, dpi=150)
        mlflow.log_artifact(f.name, artifact_subdir)
        if extra_targets:
            for client, run_id, subdir in extra_targets:
                client.log_artifact(run_id, f.name, subdir)
    plt.close(fig)


def plot_dmt_evolution(times, y_true, y_pred, title, artifact_subdir):
    """Line plot of true vs predicted DMT evolution for one fold.

    Axes: y = time (min post-dose), x = plasma DMT (ng/mL). Points sorted
    by time so the connecting lines trace the PK trajectory. Logged as an
    MLflow artifact on the active run.
    """
    times = np.asarray(times)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    order = np.argsort(times)
    t, yt, yp = times[order], y_true[order], y_pred[order]

    fig, ax = plt.subplots(figsize=(6, 7))
    ax.plot(yt, t, "-o", color="steelblue", markersize=3, linewidth=1.3,
            label="True")
    ax.plot(yp, t, "-o", color="darkorange", markersize=3, linewidth=1.3,
            alpha=0.85, label="Predicted")
    ax.set_ylabel("Time (min post-dose)")
    ax.set_xlabel("Plasma DMT (ng/mL)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        fig.savefig(f.name, dpi=150)
        mlflow.log_artifact(f.name, artifact_subdir)
    plt.close(fig)


def train_one_epoch(model, loader, criterion, optimizer, device,
                    mixup_alpha=0.0, ema_model=None, ema_decay=0.999):
    model.train()
    total_loss, n_samples = 0.0, 0
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        if mixup_alpha and mixup_alpha > 0 and X.size(0) > 1:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            idx = torch.randperm(X.size(0), device=X.device)
            X = lam * X + (1.0 - lam) * X[idx]
            y = lam * y + (1.0 - lam) * y[idx]
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        if ema_model is not None:
            with torch.no_grad():
                for pe, p in zip(ema_model.parameters(), model.parameters()):
                    pe.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)
                for be, b in zip(ema_model.buffers(), model.buffers()):
                    be.data.copy_(b.data)
        bs = X.size(0)
        total_loss += loss.item() * bs
        n_samples += bs
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return total_loss / n_samples, compute_regression_metrics(all_labels, all_preds)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n_samples = 0.0, 0
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        preds = model(X)
        loss = criterion(preds, y)
        bs = X.size(0)
        total_loss += loss.item() * bs
        n_samples += bs
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return total_loss / n_samples, compute_regression_metrics(all_labels, all_preds), all_labels, all_preds


def run_fold(args, val_subject, fold_idx, n_folds, device,
             parent_run_id=None, mlf_client=None, cv_subjects=None):
    """Train one LOSO fold. Returns dict with best-model metrics and held-out preds.

    If parent_run_id + mlf_client are provided, per-epoch training curves are
    also logged on the parent run under keys fold_{subj}_<metric> with
    step=epoch, and the scatter plot is mirrored into the parent's
    per_fold_scatter/ artifacts so curves and plots are all viewable on the
    parent run."""
    pool = cv_subjects if cv_subjects is not None else ALL_SUBJECTS
    train_subjects = [s for s in pool if s != val_subject]

    print("\n" + "=" * 70)
    print(f"  FOLD {fold_idx}/{n_folds}  |  val_subject = {val_subject}")
    print("=" * 70, flush=True)
    print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Val subject:                 {val_subject}")

    train_ds = EEGDataset(subjects=train_subjects)
    val_ds = EEGDataset(subjects=[val_subject])
    print(f"Train samples: {len(train_ds):5d} | Val samples: {len(val_ds):5d}")
    print(f"Label range (train): [{float(train_ds.labels.min()):.2f}, "
          f"{float(train_ds.labels.max()):.2f}] ng/mL  "
          f"(mean={float(train_ds.labels.mean()):.2f})")
    print(f"Label range (val):   [{float(val_ds.labels.min()):.2f}, "
          f"{float(val_ds.labels.max()):.2f}] ng/mL  "
          f"(mean={float(val_ds.labels.mean()):.2f})")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = SimpleCNN(in_channels=train_ds.n_channels, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    ema_model = SimpleCNN(in_channels=train_ds.n_channels, dropout=args.dropout).to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad_(False)
    ema_decay = 0.999  # per-batch

    if args.loss == "smoothl1":
        criterion = nn.SmoothL1Loss(beta=args.huber_beta)
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    mlflow.log_params({
        "val_subject": val_subject,
        "train_subjects": ",".join(train_subjects),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "in_channels": train_ds.n_channels,
        "n_params": n_params,
        "fold_idx": fold_idx,
        "n_folds": n_folds,
    })

    best_val_r2 = -float("inf")
    best_epoch = 0
    best_val_loss = float("inf")
    best_val_metrics = None
    patience_counter = 0
    best_state = None

    print(f"\nTraining (max_epochs={args.epochs}, patience={args.patience} on val_r2)")
    print("-" * 70, flush=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            mixup_alpha=args.mixup_alpha,
            ema_model=ema_model, ema_decay=ema_decay,
        )
        val_loss, val_metrics, _, _ = evaluate(ema_model, val_loader, criterion, device)
        elapsed = time.time() - t0

        epoch_metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mae": train_metrics["mae"],
            "val_mae": val_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "val_rmse": val_metrics["rmse"],
            "train_mse": train_metrics["mse"],
            "val_mse": val_metrics["mse"],
            "train_r2": train_metrics["r2"],
            "val_r2": val_metrics["r2"],
        }
        mlflow.log_metrics(epoch_metrics, step=epoch)

        if mlf_client is not None and parent_run_id is not None:
            for k, v in epoch_metrics.items():
                mlf_client.log_metric(
                    parent_run_id, f"fold_{val_subject}_{k}", v, step=epoch
                )

        improved = val_metrics["r2"] > best_val_r2
        marker = "*" if improved else " "
        print(f"  Ep {epoch:3d}/{args.epochs} {marker} | "
              f"loss {train_loss:7.3f}/{val_loss:7.3f} | "
              f"MAE {train_metrics['mae']:6.2f}/{val_metrics['mae']:6.2f} | "
              f"RMSE {train_metrics['rmse']:6.2f}/{val_metrics['rmse']:6.2f} | "
              f"R² {train_metrics['r2']:+.3f}/{val_metrics['r2']:+.3f} | "
              f"{elapsed:4.1f}s  "
              f"[best val_r2 {best_val_r2:+.3f}@ep{best_epoch}, "
              f"pat {patience_counter}/{args.patience}]",
              flush=True)

        if improved:
            best_val_r2 = val_metrics["r2"]
            best_epoch = epoch
            best_val_loss = val_loss
            best_val_metrics = val_metrics
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in ema_model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stop at epoch {epoch} "
                      f"(no val_r2 improvement for {args.patience} epochs). "
                      f"Best: ep{best_epoch}, val_r2={best_val_r2:+.4f}")
                break

    # Restore best-R² EMA weights and do a clean final eval
    if best_state is not None:
        ema_model.load_state_dict(best_state)
        ema_model.to(device)

    final_val_loss, final_val_metrics, y_true, y_pred = evaluate(
        ema_model, val_loader, criterion, device
    )

    print("-" * 70)
    print(f"  Fold {fold_idx}/{n_folds} BEST:")
    print(f"    epoch       = {best_epoch}")
    print(f"    val_loss    = {final_val_loss:.4f}")
    print(f"    val_mae     = {final_val_metrics['mae']:.4f} ng/mL")
    print(f"    val_rmse    = {final_val_metrics['rmse']:.4f} ng/mL")
    print(f"    val_mse     = {final_val_metrics['mse']:.4f}")
    print(f"    val_r2      = {final_val_metrics['r2']:+.4f}")
    print(f"  Predicted vs actual (val, first 5):")
    for i in range(min(5, len(y_true))):
        print(f"    y_true={y_true[i]:8.2f}  y_pred={y_pred[i]:8.2f}  "
              f"err={y_pred[i] - y_true[i]:+8.2f}")
    print(flush=True)

    mlflow.log_metrics({
        "best_epoch": best_epoch,
        "best_val_loss": final_val_loss,
        "best_val_mae": final_val_metrics["mae"],
        "best_val_rmse": final_val_metrics["rmse"],
        "best_val_mse": final_val_metrics["mse"],
        "best_val_r2": final_val_metrics["r2"],
    })

    extra = None
    if mlf_client is not None and parent_run_id is not None:
        extra = [(mlf_client, parent_run_id,
                  f"per_fold_scatter/fold_{fold_idx:02d}_{val_subject}")]
    plot_predicted_vs_actual(
        y_true, y_pred,
        title=f"Fold {fold_idx}/{n_folds} (val={val_subject})",
        metrics=final_val_metrics,
        artifact_subdir="predicted_vs_actual",
        extra_targets=extra,
    )

    plot_dmt_evolution(
        times=val_ds.times,
        y_true=y_true,
        y_pred=y_pred,
        title=(f"DMT evolution — fold {fold_idx}/{n_folds} (val={val_subject})\n"
               f"MAE={final_val_metrics['mae']:.1f}  "
               f"RMSE={final_val_metrics['rmse']:.1f}  "
               f"R²={final_val_metrics['r2']:.3f}"),
        artifact_subdir="dmt_evolution",
    )

    if args.log_model:
        mlflow.pytorch.log_model(model, "model")

    return {
        "val_subject": val_subject,
        "best_epoch": best_epoch,
        "best_val_loss": final_val_loss,
        "best_val_mae": final_val_metrics["mae"],
        "best_val_rmse": final_val_metrics["rmse"],
        "best_val_mse": final_val_metrics["mse"],
        "best_val_r2": final_val_metrics["r2"],
        "y_true": y_true,
        "y_pred": y_pred,
        "n_val": len(val_ds),
    }


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cv_subjects = args.subjects if args.subjects else ALL_SUBJECTS
    n_folds = len(cv_subjects)

    print("=" * 70)
    print("  SimpleCNN DMT Plasma Regression  |  LOSO Cross-Validation")
    print("=" * 70)
    print(f"Device:        {device}")
    print(f"Subjects ({n_folds}): {cv_subjects}")
    print(f"Hyperparams:   lr={args.lr}  bs={args.batch_size}  "
          f"dropout={args.dropout}  weight_decay={args.weight_decay}")
    print(f"Training:      max_epochs={args.epochs}  patience={args.patience} "
          f"(early-stop metric: val_r2, higher is better)")
    print(f"Seed:          {args.seed}")
    print("=" * 70, flush=True)

    mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))
    mlflow.set_experiment(args.experiment_name)

    run_name = args.run_name or (
        f"cv_lr{args.lr}_bs{args.batch_size}_do{args.dropout}_wd{args.weight_decay}"
    )

    t_global = time.time()
    mlf_client = MlflowClient()
    with mlflow.start_run(run_name=run_name) as parent_run:
        parent_run_id = parent_run.info.run_id
        mlflow.log_params({
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "n_folds": n_folds,
            "cv_subjects": ",".join(cv_subjects),
            "device": str(device),
            "seed": args.seed,
            "task": "regression",
            "cv_scheme": "leave-one-subject-out",
            "early_stop_metric": "val_r2",
            "early_stop_direction": "maximize",
        })

        fold_results = []
        for i, subj in enumerate(cv_subjects, start=1):
            t_fold = time.time()
            with mlflow.start_run(nested=True, run_name=f"fold_{i:02d}_val_{subj}"):
                mlflow.log_params({
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "patience": args.patience,
                    "dropout": args.dropout,
                    "weight_decay": args.weight_decay,
                    "seed": args.seed,
                    "cv_scheme": "leave-one-subject-out",
                    "early_stop_metric": "val_r2",
                })
                res = run_fold(args, subj, i, n_folds, device,
                               parent_run_id=parent_run_id,
                               mlf_client=mlf_client,
                               cv_subjects=cv_subjects)
                fold_results.append(res)

            # Per-fold best (scalar) summary on PARENT run — single step, easy to read in UI table
            mlf_client.log_metric(parent_run_id, f"fold_{subj}_best_val_r2",
                                  res["best_val_r2"])
            mlf_client.log_metric(parent_run_id, f"fold_{subj}_best_val_mae",
                                  res["best_val_mae"])
            mlf_client.log_metric(parent_run_id, f"fold_{subj}_best_val_rmse",
                                  res["best_val_rmse"])
            mlf_client.log_metric(parent_run_id, f"fold_{subj}_best_val_mse",
                                  res["best_val_mse"])
            mlf_client.log_metric(parent_run_id, f"fold_{subj}_best_val_loss",
                                  res["best_val_loss"])
            mlf_client.log_metric(parent_run_id, f"fold_{subj}_best_epoch",
                                  res["best_epoch"])

            print(f"  >>> Fold {i}/{n_folds} ({subj}) finished in "
                  f"{time.time() - t_fold:.1f}s", flush=True)

        # -----  Per-fold summary table  -----
        print("\n" + "=" * 70)
        print("  CV FOLD SUMMARY")
        print("=" * 70)
        header = (f"{'fold':>4} {'subject':>8} {'n_val':>6} {'epochs':>7} "
                  f"{'val_r2':>9} {'val_mae':>9} {'val_rmse':>9} {'val_mse':>10}")
        print(header)
        print("-" * len(header))
        for i, r in enumerate(fold_results, start=1):
            print(f"{i:>4} {r['val_subject']:>8} {r['n_val']:>6} {r['best_epoch']:>7} "
                  f"{r['best_val_r2']:>+9.4f} {r['best_val_mae']:>9.4f} "
                  f"{r['best_val_rmse']:>9.4f} {r['best_val_mse']:>10.4f}")

        # -----  Aggregate stats (mean / std / min / max)  -----
        def stats(values):
            arr = np.asarray(values, dtype=float)
            return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())

        agg_keys = ["best_val_r2", "best_val_mae", "best_val_rmse",
                    "best_val_mse", "best_val_loss", "best_epoch"]
        agg = {k: stats([r[k] for r in fold_results]) for k in agg_keys}

        print("\n" + "=" * 70)
        print("  CV AGGREGATE (across folds)")
        print("=" * 70)
        print(f"{'metric':<16} {'mean':>11} {'std':>11} {'min':>11} {'max':>11}")
        print("-" * 62)
        for k, (mean, std, mn, mx) in agg.items():
            print(f"{k:<16} {mean:>11.4f} {std:>11.4f} {mn:>11.4f} {mx:>11.4f}")

        mlflow.log_metrics({
            "cv_mean_val_r2":    agg["best_val_r2"][0],
            "cv_std_val_r2":     agg["best_val_r2"][1],
            "cv_min_val_r2":     agg["best_val_r2"][2],
            "cv_max_val_r2":     agg["best_val_r2"][3],
            "cv_mean_val_mae":   agg["best_val_mae"][0],
            "cv_std_val_mae":    agg["best_val_mae"][1],
            "cv_mean_val_rmse":  agg["best_val_rmse"][0],
            "cv_std_val_rmse":   agg["best_val_rmse"][1],
            "cv_mean_val_mse":   agg["best_val_mse"][0],
            "cv_std_val_mse":    agg["best_val_mse"][1],
            "cv_mean_val_loss":  agg["best_val_loss"][0],
            "cv_std_val_loss":   agg["best_val_loss"][1],
            "cv_mean_best_epoch": agg["best_epoch"][0],
            "cv_std_best_epoch":  agg["best_epoch"][1],
        })

        # -----  Pooled metrics (concatenate all held-out predictions)  -----
        all_true = np.concatenate([r["y_true"] for r in fold_results])
        all_pred = np.concatenate([r["y_pred"] for r in fold_results])
        pooled = compute_regression_metrics(all_true, all_pred)

        print("\n" + "=" * 70)
        print(f"  POOLED (concatenated held-out predictions, N={len(all_true)})")
        print("=" * 70)
        print(f"  pooled_val_r2:   {pooled['r2']:+.4f}")
        print(f"  pooled_val_mae:  {pooled['mae']:.4f}")
        print(f"  pooled_val_rmse: {pooled['rmse']:.4f}")
        print(f"  pooled_val_mse:  {pooled['mse']:.4f}")

        mlflow.log_metrics({
            "pooled_val_r2":   pooled["r2"],
            "pooled_val_mae":  pooled["mae"],
            "pooled_val_rmse": pooled["rmse"],
            "pooled_val_mse":  pooled["mse"],
        })

        plot_predicted_vs_actual(
            all_true, all_pred,
            title=f"LOSO CV pooled ({n_folds} folds, N={len(all_true)})",
            metrics=pooled,
            artifact_subdir="predicted_vs_actual_pooled",
        )

        # -----  Machine-parseable one-line summary (easy for autoresearch to grep)  -----
        print("\n" + "=" * 70)
        print("  RESULT (one-line, for autoresearch):")
        print(
            f"  RESULT cv_mean_val_r2={agg['best_val_r2'][0]:+.4f} "
            f"cv_std_val_r2={agg['best_val_r2'][1]:.4f} "
            f"cv_mean_val_mae={agg['best_val_mae'][0]:.4f} "
            f"cv_mean_val_rmse={agg['best_val_rmse'][0]:.4f} "
            f"pooled_val_r2={pooled['r2']:+.4f} "
            f"pooled_val_mae={pooled['mae']:.4f} "
            f"pooled_val_rmse={pooled['rmse']:.4f} "
            f"mean_best_epoch={agg['best_epoch'][0]:.1f}"
        )
        print("=" * 70)
        print(f"  Total wall time: {time.time() - t_global:.1f}s")
        print(f"  Parent run id:   {parent_run_id}")
        print(f"  View:            mlflow ui --backend-store-uri mlruns/")
        print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
