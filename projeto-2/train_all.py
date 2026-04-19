"""
Unified training script with:
- Train / Val / Test split (70/15/15)
- Per-epoch training curves (loss + val metrics)
- Both CLIP and ConvNeXt features
- Plots: loss curves, val accuracy curves, confusion matrices
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_DIR = "D:/media/machine-learning-master-class/projeto-2/dataset"
OUTPUT_DIR = "D:/media/machine-learning-master-class/projeto-2/output"
SEED = 42
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# === Model ===

class MLPProbe(nn.Module):
    def __init__(self, in_dim, hidden1=256, hidden2=128, n_type_classes=6, dropout=0.3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_people = nn.Linear(hidden2, 1)
        self.head_type = nn.Linear(hidden2, n_type_classes)

    def forward(self, x):
        h = self.backbone(x)
        return self.head_people(h).squeeze(-1), self.head_type(h)


class LinearProbe(nn.Module):
    def __init__(self, in_dim, n_type_classes=6):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1 + n_type_classes)
        self.n_type = n_type_classes

    def forward(self, x):
        out = self.linear(x)
        return out[:, 0], out[:, 1:]


# === Data ===

def load_split(feature_file):
    X = np.load(feature_file)
    y_type = np.load(f"{DATASET_DIR}/y_type.npy")
    y_people = np.load(f"{DATASET_DIR}/y_people.npy")
    with open(f"{DATASET_DIR}/class_names.json") as f:
        class_names = json.load(f)

    # 70/15/15 stratified split
    X_trainval, X_test, yt_trainval, yt_test, yp_trainval, yp_test = train_test_split(
        X, y_type, y_people, test_size=0.15, random_state=SEED, stratify=y_type
    )
    X_train, X_val, yt_train, yt_val, yp_train, yp_val = train_test_split(
        X_trainval, yt_trainval, yp_trainval, test_size=0.176, random_state=SEED, stratify=yt_trainval
        # 0.176 of 85% ~ 15% of total
    )

    def make_loader(X, yt, yp, shuffle=False):
        return DataLoader(
            TensorDataset(torch.from_numpy(X), torch.from_numpy(yt), torch.from_numpy(yp)),
            batch_size=BATCH_SIZE, shuffle=shuffle,
        )

    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    return (
        make_loader(X_train, yt_train, yp_train, shuffle=True),
        make_loader(X_val, yt_val, yp_val),
        make_loader(X_test, yt_test, yp_test),
        class_names,
    )


# === Training ===

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for X_b, yt_b, yp_b in loader:
        X_b, yt_b, yp_b = X_b.to(DEVICE), yt_b.to(DEVICE), yp_b.to(DEVICE)
        pl, tl = model(X_b)
        loss = F.binary_cross_entropy_with_logits(pl, yp_b) + F.cross_entropy(tl, yt_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_b.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def compute_metrics(model, loader):
    model.eval()
    all_pl, all_tl, all_yp, all_yt = [], [], [], []
    total_loss = 0
    for X_b, yt_b, yp_b in loader:
        X_b, yt_b, yp_b = X_b.to(DEVICE), yt_b.to(DEVICE), yp_b.to(DEVICE)
        pl, tl = model(X_b)
        loss = F.binary_cross_entropy_with_logits(pl, yp_b) + F.cross_entropy(tl, yt_b)
        total_loss += loss.item() * X_b.size(0)
        all_pl.append(pl.cpu()); all_tl.append(tl.cpu())
        all_yp.append(yp_b.cpu()); all_yt.append(yt_b.cpu())

    pl = torch.cat(all_pl); tl = torch.cat(all_tl)
    yp = torch.cat(all_yp); yt = torch.cat(all_yt)

    pp = (pl > 0).long()
    tp = tl.argmax(dim=1)

    p_acc = accuracy_score(yp, pp)
    p_f1 = precision_recall_fscore_support(yp, pp, average="binary", zero_division=0)[2]
    p_auc = roc_auc_score(yp, pl) if yp.sum() > 0 else 0.0
    t_acc = accuracy_score(yt, tp)
    t_prec, t_rec, t_f1, _ = precision_recall_fscore_support(yt, tp, average="weighted", zero_division=0)

    return {
        "loss": total_loss / len(loader.dataset),
        "people_acc": p_acc, "people_f1": p_f1, "people_auc": p_auc,
        "type_acc": t_acc, "type_f1": t_f1, "type_prec": t_prec, "type_rec": t_rec,
    }


@torch.no_grad()
def full_report(model, loader, class_names, label=""):
    model.eval()
    all_pl, all_tl, all_yp, all_yt = [], [], [], []
    for X_b, yt_b, yp_b in loader:
        pl, tl = model(X_b.to(DEVICE))
        all_pl.append(pl.cpu()); all_tl.append(tl.cpu())
        all_yp.append(yp_b); all_yt.append(yt_b)

    pl = torch.cat(all_pl); tl = torch.cat(all_tl)
    yp = torch.cat(all_yp); yt = torch.cat(all_yt)
    pp = (pl > 0).long(); tp = tl.argmax(dim=1)

    p_acc = accuracy_score(yp, pp)
    p_prec, p_rec, p_f1, _ = precision_recall_fscore_support(yp, pp, average="binary", zero_division=0)
    p_auc = roc_auc_score(yp, pl)
    t_acc = accuracy_score(yt, tp)
    t_prec, t_rec, t_f1, _ = precision_recall_fscore_support(yt, tp, average="weighted", zero_division=0)

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"\n  has_people:  Acc={p_acc:.4f}  Prec={p_prec:.4f}  Rec={p_rec:.4f}  F1={p_f1:.4f}  AUC={p_auc:.4f}")
    print(f"  image_type:  Acc={t_acc:.4f}  Prec={t_prec:.4f}  Rec={t_rec:.4f}  F1={t_f1:.4f}")
    print(f"\n{classification_report(yt, tp, target_names=class_names, zero_division=0)}")

    return {
        "people_acc": p_acc, "people_f1": p_f1, "people_auc": p_auc,
        "type_acc": t_acc, "type_f1": t_f1,
        "cm": confusion_matrix(yt, tp),
        "y_true": yt.numpy(), "y_pred": tp.numpy(),
        "people_true": yp.numpy(), "people_pred": pp.numpy(), "people_logits": pl.numpy(),
    }


def run_experiment(name, feature_file, in_dim, model_class, **model_kwargs):
    print(f"\n{'#' * 60}")
    print(f"  EXPERIMENT: {name}")
    print(f"{'#' * 60}")

    train_loader, val_loader, test_loader, class_names = load_split(feature_file)

    model = model_class(in_dim=in_dim, n_type_classes=len(class_names), **model_kwargs).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "val_type_acc": [], "val_type_f1": [],
               "val_people_acc": [], "val_people_f1": []}

    best_val_f1 = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer)
        scheduler.step()

        val_m = compute_metrics(model, val_loader)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_m["loss"])
        history["val_type_acc"].append(val_m["type_acc"])
        history["val_type_f1"].append(val_m["type_f1"])
        history["val_people_acc"].append(val_m["people_acc"])
        history["val_people_f1"].append(val_m["people_f1"])

        # Best model by val type_f1
        if val_m["type_f1"] > best_val_f1:
            best_val_f1 = val_m["type_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f}  val_loss={val_m['loss']:.4f}  "
                  f"val_type_f1={val_m['type_f1']:.4f}  val_people_f1={val_m['people_f1']:.4f}")

    # Load best model
    model.load_state_dict(best_state)
    print(f"\n  Best val type_f1: {best_val_f1:.4f}")

    train_report = full_report(model, train_loader, class_names, f"{name} - TRAIN")
    val_report = full_report(model, val_loader, class_names, f"{name} - VAL")
    test_report = full_report(model, test_loader, class_names, f"{name} - TEST")

    return {
        "name": name, "params": n_params, "history": history,
        "train": train_report, "val": val_report, "test": test_report,
        "class_names": class_names,
    }


# === Plotting ===

def plot_training_curves(experiments):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Curves", fontsize=16, fontweight="bold")

    metrics = [
        ("train_loss", "Train Loss"),
        ("val_loss", "Val Loss"),
        ("val_type_f1", "Val Type F1"),
        ("val_people_f1", "Val People F1"),
    ]

    for ax, (key, title) in zip(axes.flat, metrics):
        for exp in experiments:
            ax.plot(exp["history"]["epoch"], exp["history"][key], label=exp["name"], linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/training_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_confusion_matrices(experiments):
    n = len(experiments)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, exp in zip(axes, experiments):
        cm = exp["test"]["cm"]
        class_names = exp["class_names"]
        # Normalize
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                    xticklabels=[c[:10] for c in class_names],
                    yticklabels=[c[:10] for c in class_names],
                    ax=ax, cbar=False, square=True)
        ax.set_title(f"{exp['name']}\nTest Acc={exp['test']['type_acc']:.3f}", fontsize=12)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/confusion_matrices.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_comparison_bar(experiments):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    names = [e["name"] for e in experiments]
    x = np.arange(len(names))
    w = 0.25

    # Type metrics
    ax = axes[0]
    acc = [e["test"]["type_acc"] for e in experiments]
    f1 = [e["test"]["type_f1"] for e in experiments]
    bars1 = ax.bar(x - w/2, acc, w, label="Accuracy", color="#4C72B0")
    bars2 = ax.bar(x + w/2, f1, w, label="F1 (weighted)", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("image_type (Test Set)")
    ax.set_ylim(0.6, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    # People metrics
    ax = axes[1]
    acc = [e["test"]["people_acc"] for e in experiments]
    f1 = [e["test"]["people_f1"] for e in experiments]
    auc = [e["test"]["people_auc"] for e in experiments]
    bars1 = ax.bar(x - w, acc, w, label="Accuracy", color="#4C72B0")
    bars2 = ax.bar(x, f1, w, label="F1", color="#55A868")
    bars3 = ax.bar(x + w, auc, w, label="AUC", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("has_people (Test Set)")
    ax.set_ylim(0.6, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/comparison_bar.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")

    clip_features = f"{DATASET_DIR}/X_embeddings.npy"
    convnext_features = f"{DATASET_DIR}/X_convnext.npy"

    experiments = []

    # 1. Linear Probe (CLIP)
    exp1 = run_experiment(
        "Linear (CLIP)", clip_features, in_dim=512,
        model_class=LinearProbe,
    )
    experiments.append(exp1)

    # 2. MLP (CLIP)
    exp2 = run_experiment(
        "MLP (CLIP)", clip_features, in_dim=512,
        model_class=MLPProbe,
    )
    experiments.append(exp2)

    # 3. MLP (ConvNeXt)
    exp3 = run_experiment(
        "MLP (ConvNeXt)", convnext_features, in_dim=768,
        model_class=MLPProbe,
    )
    experiments.append(exp3)

    # Final comparison table
    print(f"\n{'#' * 60}")
    print(f"  FINAL COMPARISON (test set, best val checkpoint)")
    print(f"{'#' * 60}")
    print(f"  {'':20s} {'Lin(CLIP)':>10s} {'MLP(CLIP)':>10s} {'MLP(CNeXt)':>10s}")
    for k in ["type_acc", "type_f1", "people_acc", "people_f1", "people_auc"]:
        vals = [e["test"][k] for e in experiments]
        print(f"  {k:20s} {vals[0]:10.4f} {vals[1]:10.4f} {vals[2]:10.4f}")
    print(f"  {'params':20s} {experiments[0]['params']:10,d} {experiments[1]['params']:10,d} {experiments[2]['params']:10,d}")

    # Plots
    print("\nGenerating plots...")
    plot_training_curves(experiments)
    plot_confusion_matrices(experiments)
    plot_comparison_bar(experiments)

    # Save all metrics
    summary = {}
    for exp in experiments:
        summary[exp["name"]] = {
            "params": exp["params"],
            "test": {k: v for k, v in exp["test"].items() if k not in ("cm", "y_true", "y_pred", "people_true", "people_pred", "people_logits")},
            "val": {k: v for k, v in exp["val"].items() if k not in ("cm", "y_true", "y_pred", "people_true", "people_pred", "people_logits")},
        }
    with open(f"{OUTPUT_DIR}/all_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll done. Results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
