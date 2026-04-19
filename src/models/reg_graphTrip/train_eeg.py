"""
Training script for graphTrip EEG DMT plasma regression.

Predicts plasma DMT concentration (ng/mL) from 3-second EEG trials
using a graph neural network with VGAE reconstruction loss.

Usage (run from project root):
    python -m src.models.reg_graphTrip.train_eeg --val_subjects S12 S13 \
        --lr 1e-3 --batch_size 16 --epochs 100
"""

import argparse
import time
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import mlflow
import mlflow.pytorch

from .eeg_graph_dataset import EEGGraphDataset, get_subject_split
from .eeg_classifier import EEGGraphClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args():
    p = argparse.ArgumentParser(description="Train graphTrip EEG DMT regression")
    p.add_argument("--val_subjects", nargs="+", default=["S12", "S13"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--use_coords", action="store_true",
                    help="Use electrode 10-10 coordinates as conditional node features")
    p.add_argument("--task_weight", type=float, default=1.0,
                    help="Weight for regression loss relative to VGAE loss")
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--experiment_name", type=str, default="graphTrip_DMT_regression")
    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


def compute_regression_metrics(y_true, y_pred):
    """Compute MAE, RMSE, R² for regression."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "mse": mse, "r2": r2}


def plot_predicted_vs_actual(y_true, y_pred, val_subjects, metrics):
    """Scatter plot of predicted vs actual plasma concentration."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=15, c="steelblue")

    # Identity line
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")

    ax.set_xlabel("Actual plasma DMT (ng/mL)")
    ax.set_ylabel("Predicted plasma DMT (ng/mL)")
    ax.set_title(f"Val subjects: {','.join(val_subjects)}\n"
                 f"MAE={metrics['mae']:.1f}  RMSE={metrics['rmse']:.1f}  R²={metrics['r2']:.3f}")
    ax.legend()
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        fig.savefig(f.name, dpi=150)
        mlflow.log_artifact(f.name, "predicted_vs_actual")
    plt.close(fig)


@torch.no_grad()
def plot_latent_space(model, train_loader, val_loader, val_subjects, device):
    """t-SNE of graph-level latent features, colored by plasma concentration."""
    model.eval()

    all_feats, all_labels, all_splits = [], [], []
    for split_name, loader in [("train", train_loader), ("val", val_loader)]:
        for batch in loader:
            batch = batch.to(device)
            _, _, graph_feats = model(batch)
            all_feats.append(graph_feats.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
            all_splits.extend([split_name] * batch.num_graphs)

    all_feats = np.concatenate(all_feats)
    all_labels = np.concatenate(all_labels)
    all_splits = np.array(all_splits)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    proj = tsne.fit_transform(all_feats)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Color by concentration
    sc = axes[0].scatter(proj[:, 0], proj[:, 1], c=all_labels, cmap="viridis",
                         alpha=0.5, s=10)
    axes[0].set_title("All data - by concentration (ng/mL)")
    fig.colorbar(sc, ax=axes[0], label="ng/mL")

    # Color by split
    for split, color in [("train", "tab:gray"), ("val", "tab:orange")]:
        mask = all_splits == split
        axes[1].scatter(proj[mask, 0], proj[mask, 1], c=color, label=split, alpha=0.5, s=10)
    axes[1].set_title("All data - by split")
    axes[1].legend()

    # Val only, colored by concentration
    val_mask = all_splits == "val"
    sc2 = axes[2].scatter(proj[val_mask, 0], proj[val_mask, 1],
                          c=all_labels[val_mask], cmap="viridis", alpha=0.6, s=15)
    axes[2].set_title(f"Val only ({','.join(val_subjects)}) - by concentration")
    fig.colorbar(sc2, ax=axes[2], label="ng/mL")

    for ax in axes:
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

    fig.suptitle("Latent space (graph-level features -> t-SNE)", fontsize=13)
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        fig.savefig(f.name, dpi=150)
        mlflow.log_artifact(f.name, "latent_space")
    plt.close(fig)
    print("Saved latent space plot to MLflow artifacts.")


def train_one_epoch(model, loader, criterion, optimizer, device, task_weight):
    model.train()
    total_loss = 0.0
    total_task_loss = 0.0
    total_vgae_loss = 0.0
    all_preds, all_labels = [], []
    n_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        output, vgae_data, _ = model(batch)
        task_loss = criterion(output, batch.y)
        vgae_loss = model.vgae_loss(vgae_data)
        loss = vgae_loss + task_weight * task_loss

        loss.backward()
        optimizer.step()

        bs = batch.num_graphs
        total_loss += loss.item() * bs
        total_task_loss += task_loss.item() * bs
        total_vgae_loss += vgae_loss.item() * bs
        n_samples += bs

        all_preds.append(output.detach().cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = compute_regression_metrics(all_labels, all_preds)
    return total_loss / n_samples, total_task_loss / n_samples, total_vgae_loss / n_samples, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, task_weight):
    model.eval()
    total_loss = 0.0
    total_task_loss = 0.0
    total_vgae_loss = 0.0
    all_preds, all_labels = [], []
    n_samples = 0

    for batch in loader:
        batch = batch.to(device)
        output, vgae_data, _ = model(batch)
        task_loss = criterion(output, batch.y)
        vgae_loss = model.vgae_loss(vgae_data)
        loss = vgae_loss + task_weight * task_loss

        bs = batch.num_graphs
        total_loss += loss.item() * bs
        total_task_loss += task_loss.item() * bs
        total_vgae_loss += vgae_loss.item() * bs
        n_samples += bs

        all_preds.append(output.cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = compute_regression_metrics(all_labels, all_preds)
    return total_loss / n_samples, total_task_loss / n_samples, total_vgae_loss / n_samples, metrics, all_labels, all_preds


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_subjects, val_subjects = get_subject_split(args.val_subjects)
    print(f"Train subjects: {train_subjects}")
    print(f"Val subjects:   {val_subjects}")
    print(f"Use coords:     {args.use_coords}")

    train_ds = EEGGraphDataset(subjects=train_subjects, use_coords=args.use_coords)
    val_ds = EEGGraphDataset(subjects=val_subjects, use_coords=args.use_coords)
    print(f"Train graphs: {len(train_ds)} | Val graphs: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Model (num_classes=1 for regression)
    num_cond_attrs = 3 if args.use_coords else 0
    model = EEGGraphClassifier(
        num_node_attr=5,
        num_cond_attrs=num_cond_attrs,
        num_edge_attr=1,
        hidden_dim=args.hidden_dim,
        node_emb_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        num_classes=1,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # MLflow
    mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))
    mlflow.set_experiment(args.experiment_name)

    run_name = args.run_name or (
        f"lr{args.lr}_bs{args.batch_size}_do{args.dropout}"
        f"_tw{args.task_weight}_{'coords' if args.use_coords else 'nocoords'}"
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "use_coords": args.use_coords,
            "task_weight": args.task_weight,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "val_subjects": ",".join(args.val_subjects),
            "train_subjects": ",".join(train_subjects),
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "device": str(device),
            "task": "regression",
        })

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            train_loss, train_task, train_vgae, train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device, args.task_weight
            )
            val_loss, val_task, val_vgae, val_metrics, _, _ = evaluate(
                model, val_loader, criterion, device, args.task_weight
            )

            elapsed = time.time() - t0

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_task_loss": train_task,
                "val_task_loss": val_task,
                "train_vgae_loss": train_vgae,
                "val_vgae_loss": val_vgae,
                "train_mae": train_metrics["mae"],
                "val_mae": val_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "val_rmse": val_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "val_r2": val_metrics["r2"],
            }, step=epoch)

            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"MAE: {train_metrics['mae']:.1f}/{val_metrics['mae']:.1f} | "
                  f"R²: {train_metrics['r2']:.3f}/{val_metrics['r2']:.3f} | "
                  f"{elapsed:.1f}s")

            # Early stopping on total val loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                    break

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model.to(device)

        val_loss, val_task, val_vgae, val_metrics, y_true, y_pred = evaluate(
            model, val_loader, criterion, device, args.task_weight
        )

        print(f"\n--- Best model (val_loss={best_val_loss:.4f}) ---")
        print(f"Val MAE:  {val_metrics['mae']:.1f} ng/mL")
        print(f"Val RMSE: {val_metrics['rmse']:.1f} ng/mL")
        print(f"Val R²:   {val_metrics['r2']:.3f}")

        mlflow.log_metrics({
            "best_val_loss": best_val_loss,
            "best_val_mae": val_metrics["mae"],
            "best_val_rmse": val_metrics["rmse"],
            "best_val_r2": val_metrics["r2"],
        })

        # Predicted vs actual scatter plot
        plot_predicted_vs_actual(y_true, y_pred, args.val_subjects, val_metrics)

        # Latent space t-SNE plot
        plot_latent_space(model, train_loader, val_loader, args.val_subjects, device)

        # Save model
        mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    main()
