"""
Training script for SimpleCNN EEG classifier.

Usage:
    python train.py --val_subjects S12 S13 --lr 1e-3 --batch_size 64 --epochs 100 --patience 15 --dropout 0.3
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tempfile

import mlflow
import mlflow.pytorch

from model import SimpleCNN
from dataset import EEGDataset, get_subject_split

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args():
    p = argparse.ArgumentParser(description="Train SimpleCNN on EEG DMT data")
    p.add_argument("--val_subjects", nargs="+", default=["S12", "S13"],
                    help="Subjects for validation (rest go to training)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15,
                    help="Early stopping patience (epochs without val loss improvement)")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--experiment_name", type=str, default="SimpleCNN_EEG")
    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


def compute_metrics(y_true, y_pred):
    """Compute precision, recall (sensitivity), specificity, NPV."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "npv": npv,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def plot_confusion_matrix(metrics, val_subjects, run_name):
    """Save confusion matrix as image and log to MLflow."""
    tp, tn, fp, fn = metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]
    cm = np.array([[tn, fp], [fn, tp]])
    acc = metrics["accuracy"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pre (0)", "Post (1)"])
    ax.set_yticklabels(["Pre (0)", "Post (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Val subjects: {','.join(val_subjects)}\nVal accuracy: {acc:.4f}")

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=16)

    fig.colorbar(im)
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        fig.savefig(f.name, dpi=150)
        mlflow.log_artifact(f.name, "confusion_matrix")
    plt.close(fig)


@torch.no_grad()
def plot_latent_space(model, train_loader, val_loader, val_subjects, device):
    """Extract 128-dim features, project with t-SNE, and plot train+val colored by class."""
    model.eval()

    all_feats, all_labels, all_splits = [], [], []
    for split_name, loader in [("train", train_loader), ("val", val_loader)]:
        for X, y in loader:
            feats = model.extract_features(X.to(device)).cpu().numpy()
            all_feats.append(feats)
            all_labels.append(y.numpy())
            all_splits.extend([split_name] * len(y))

    all_feats = np.concatenate(all_feats)
    all_labels = np.concatenate(all_labels)
    all_splits = np.array(all_splits)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    proj = tsne.fit_transform(all_feats)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: All data colored by class
    for lbl, name, color in [(0, "Pre-inj", "tab:blue"), (1, "Post-inj", "tab:red")]:
        mask = all_labels == lbl
        axes[0].scatter(proj[mask, 0], proj[mask, 1], c=color, label=name, alpha=0.5, s=10)
    axes[0].set_title("All data — by class")
    axes[0].legend()

    # Plot 2: Colored by split (train vs val)
    for split, color in [("train", "tab:gray"), ("val", "tab:orange")]:
        mask = all_splits == split
        axes[1].scatter(proj[mask, 0], proj[mask, 1], c=color, label=split, alpha=0.5, s=10)
    axes[1].set_title("All data — by split")
    axes[1].legend()

    # Plot 3: Val only, colored by class
    val_mask = all_splits == "val"
    for lbl, name, color in [(0, "Pre-inj", "tab:blue"), (1, "Post-inj", "tab:red")]:
        mask = val_mask & (all_labels == lbl)
        axes[2].scatter(proj[mask, 0], proj[mask, 1], c=color, label=name, alpha=0.6, s=15)
    axes[2].set_title(f"Val only ({','.join(val_subjects)}) — by class")
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

    fig.suptitle("Latent space (128-dim features → t-SNE)", fontsize=13)
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        fig.savefig(f.name, dpi=150)
        mlflow.log_artifact(f.name, "latent_space")
    plt.close(fig)
    print("Saved latent space plot to MLflow artifacts.")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)

        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_subjects, val_subjects = get_subject_split(args.val_subjects)
    print(f"Train subjects: {train_subjects}")
    print(f"Val subjects:   {val_subjects}")

    train_ds = EEGDataset(subjects=train_subjects)
    val_ds = EEGDataset(subjects=val_subjects)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    n_channels = train_ds.n_channels

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Model
    model = SimpleCNN(in_channels=n_channels, dropout=args.dropout).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # MLflow
    mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))
    mlflow.set_experiment(args.experiment_name)

    run_name = args.run_name or f"lr{args.lr}_bs{args.batch_size}_do{args.dropout}_wd{args.weight_decay}"

    with mlflow.start_run(run_name=run_name):
        # Log params
        mlflow.log_params({
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "val_subjects": ",".join(args.val_subjects),
            "train_subjects": ",".join(train_subjects),
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "in_channels": n_channels,
            "device": str(device),
        })

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

            elapsed = time.time() - t0

            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "val_precision": val_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "val_recall": val_metrics["recall"],
                "train_specificity": train_metrics["specificity"],
                "val_specificity": val_metrics["specificity"],
                "train_npv": train_metrics["npv"],
                "val_npv": val_metrics["npv"],
            }, step=epoch)

            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train loss: {train_loss:.4f} acc: {train_metrics['accuracy']:.3f} | "
                  f"Val loss: {val_loss:.4f} acc: {val_metrics['accuracy']:.3f} | "
                  f"{elapsed:.1f}s")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                    break

        # Restore best model and do final evaluation
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model.to(device)

        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        print(f"\n--- Best model (val_loss={best_val_loss:.4f}) ---")
        print(f"Val accuracy:    {val_metrics['accuracy']:.4f}")
        print(f"Val precision:   {val_metrics['precision']:.4f}")
        print(f"Val recall:      {val_metrics['recall']:.4f}")
        print(f"Val specificity: {val_metrics['specificity']:.4f}")
        print(f"Val NPV:         {val_metrics['npv']:.4f}")
        print(f"Confusion matrix: TP={val_metrics['tp']} TN={val_metrics['tn']} "
              f"FP={val_metrics['fp']} FN={val_metrics['fn']}")

        mlflow.log_metrics({
            "best_val_loss": best_val_loss,
            "best_val_accuracy": val_metrics["accuracy"],
            "best_val_precision": val_metrics["precision"],
            "best_val_recall": val_metrics["recall"],
            "best_val_specificity": val_metrics["specificity"],
            "best_val_npv": val_metrics["npv"],
            "best_val_tp": val_metrics["tp"],
            "best_val_tn": val_metrics["tn"],
            "best_val_fp": val_metrics["fp"],
            "best_val_fn": val_metrics["fn"],
        })

        # Save confusion matrix
        plot_confusion_matrix(val_metrics, args.val_subjects, run_name)

        # Save latent space t-SNE plot
        plot_latent_space(model, train_loader, val_loader, args.val_subjects, device)

        # Save model
        mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    main()
