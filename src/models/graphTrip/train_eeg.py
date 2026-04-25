"""
Training script for graphTrip EEG classifier.

Usage:
    python train_eeg.py --val_subjects S12 S13 --lr 1e-3 --batch_size 16 --epochs 100 \
        --patience 15 --dropout 0.1 --use_coords --cls_weight 1.0
"""

import argparse
import time
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import dagshub
import mlflow
import mlflow.pytorch

from eeg_graph_dataset import EEGGraphDataset, get_subject_split
from eeg_classifier import EEGGraphClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args():
    p = argparse.ArgumentParser(description="Train graphTrip EEG classifier")
    p.add_argument("--val_subjects", nargs="+", default=["S12", "S13"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--use_coords", action="store_true",
                    help="Use electrode 10-10 coordinates as conditional node features")
    p.add_argument("--cls_weight", type=float, default=1.0,
                    help="Weight for classification loss relative to VGAE loss")
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--experiment_name", type=str, default="graphTrip_EEG")
    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
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


def plot_confusion_matrix(metrics, val_subjects):
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

    for lbl, name, color in [(0, "Pre-inj", "tab:blue"), (1, "Post-inj", "tab:red")]:
        mask = all_labels == lbl
        axes[0].scatter(proj[mask, 0], proj[mask, 1], c=color, label=name, alpha=0.5, s=10)
    axes[0].set_title("All data - by class")
    axes[0].legend()

    for split, color in [("train", "tab:gray"), ("val", "tab:orange")]:
        mask = all_splits == split
        axes[1].scatter(proj[mask, 0], proj[mask, 1], c=color, label=split, alpha=0.5, s=10)
    axes[1].set_title("All data - by split")
    axes[1].legend()

    val_mask = all_splits == "val"
    for lbl, name, color in [(0, "Pre-inj", "tab:blue"), (1, "Post-inj", "tab:red")]:
        mask = val_mask & (all_labels == lbl)
        axes[2].scatter(proj[mask, 0], proj[mask, 1], c=color, label=name, alpha=0.6, s=15)
    axes[2].set_title(f"Val only ({','.join(val_subjects)}) - by class")
    axes[2].legend()

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


def train_one_epoch(model, loader, criterion, optimizer, device, cls_weight):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_vgae_loss = 0.0
    all_preds, all_labels = [], []
    n_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits, vgae_data, _ = model(batch)
        cls_loss = criterion(logits, batch.y)
        vgae_loss = model.vgae_loss(vgae_data)
        loss = vgae_loss + cls_weight * cls_loss

        loss.backward()
        optimizer.step()

        bs = batch.num_graphs
        total_loss += loss.item() * bs
        total_cls_loss += cls_loss.item() * bs
        total_vgae_loss += vgae_loss.item() * bs
        n_samples += bs

        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_labels, all_preds)
    return total_loss / n_samples, total_cls_loss / n_samples, total_vgae_loss / n_samples, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, cls_weight):
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_vgae_loss = 0.0
    all_preds, all_labels = [], []
    n_samples = 0

    for batch in loader:
        batch = batch.to(device)
        logits, vgae_data, _ = model(batch)
        cls_loss = criterion(logits, batch.y)
        vgae_loss = model.vgae_loss(vgae_data)
        loss = vgae_loss + cls_weight * cls_loss

        bs = batch.num_graphs
        total_loss += loss.item() * bs
        total_cls_loss += cls_loss.item() * bs
        total_vgae_loss += vgae_loss.item() * bs
        n_samples += bs

        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_labels, all_preds)
    return total_loss / n_samples, total_cls_loss / n_samples, total_vgae_loss / n_samples, metrics


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

    # Model
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
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # MLflow
    dagshub.init(repo_owner="Alex44lel", repo_name="eeg_biomarkers_models", mlflow=True)
    mlflow.set_experiment(args.experiment_name)

    run_name = args.run_name or (
        f"lr{args.lr}_bs{args.batch_size}_do{args.dropout}"
        f"_cw{args.cls_weight}_{'coords' if args.use_coords else 'nocoords'}"
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
            "cls_weight": args.cls_weight,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "val_subjects": ",".join(args.val_subjects),
            "train_subjects": ",".join(train_subjects),
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "device": str(device),
        })

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            train_loss, train_cls, train_vgae, train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device, args.cls_weight
            )
            val_loss, val_cls, val_vgae, val_metrics = evaluate(
                model, val_loader, criterion, device, args.cls_weight
            )

            elapsed = time.time() - t0

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_cls_loss": train_cls,
                "val_cls_loss": val_cls,
                "train_vgae_loss": train_vgae,
                "val_vgae_loss": val_vgae,
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
                  f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"CLS: {train_cls:.4f}/{val_cls:.4f} | "
                  f"Acc: {train_metrics['accuracy']:.3f}/{val_metrics['accuracy']:.3f} | "
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

        val_loss, val_cls, val_vgae, val_metrics = evaluate(
            model, val_loader, criterion, device, args.cls_weight
        )

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

        # Confusion matrix plot
        plot_confusion_matrix(val_metrics, args.val_subjects)

        # Latent space t-SNE plot
        plot_latent_space(model, train_loader, val_loader, args.val_subjects, device)

        # Save model
        mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    main()
