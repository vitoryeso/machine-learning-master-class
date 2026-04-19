"""
End-to-end fine-tuning: CLIP ViT-B/32 and ConvNeXt-Tiny with dual-head MLP.
- Backbone LR: 1e-5 | Head LR: 1e-3
- Data augmentation: RandomResizedCrop, HorizontalFlip, ColorJitter
- Early stopping by val type_f1 (patience=20)
- Runs on GPU (CUDA)

Usage: C:/Python312/python.exe finetune_all.py
"""
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score,
)
from PIL import Image, UnidentifiedImageError
import open_clip
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
PATHS_FILE   = "D:/media/machine-learning-master-class/projeto-2/all_paths.txt"
DATASET_DIR  = "D:/media/machine-learning-master-class/projeto-2/dataset"
OUTPUT_DIR   = "D:/media/machine-learning-master-class/projeto-2/output"
CKPT_DIR     = "D:/media/machine-learning-master-class/projeto-2/checkpoints"

# ── Hyperparams ────────────────────────────────────────────────────
SEED         = 42
BATCH_SIZE   = 64
LR_BACKBONE  = 1e-5
LR_HEADS     = 1e-3
EPOCHS       = 50
PATIENCE     = 15      # early stop patience
SUBSET_FRAC  = 0.40   # fraction of train set to use (reduces epoch time ~2.5x)
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ── Data augmentation ──────────────────────────────────────────────
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(236, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Dataset ────────────────────────────────────────────────────────
class ImageDataset(Dataset):
    def __init__(self, paths, y_type, y_people, transform):
        self.paths = paths
        self.y_type = y_type
        self.y_people = y_people
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            img = self.transform(img)
        except (UnidentifiedImageError, OSError):
            img = torch.zeros(3, 224, 224)
        return img, self.y_type[idx], self.y_people[idx]


def load_splits():
    with open(PATHS_FILE, encoding="utf-8") as f:
        all_paths = [l.strip() for l in f if l.strip()]
    y_type   = np.load(f"{DATASET_DIR}/y_type.npy")
    y_people = np.load(f"{DATASET_DIR}/y_people.npy")
    with open(f"{DATASET_DIR}/class_names.json") as f:
        class_names = json.load(f)

    idx = np.arange(len(all_paths))
    idx_tv, idx_test = train_test_split(idx, test_size=0.15, random_state=SEED, stratify=y_type)
    idx_train, idx_val = train_test_split(idx_tv, test_size=0.176, random_state=SEED, stratify=y_type[idx_tv])

    # Stratified subset of train set to keep epochs fast
    if SUBSET_FRAC < 1.0:
        idx_train, _ = train_test_split(idx_train, train_size=SUBSET_FRAC,
                                         random_state=SEED, stratify=y_type[idx_train])

    paths = np.array(all_paths)
    yt = torch.from_numpy(y_type)
    yp = torch.from_numpy(y_people)

    def make_loader(idxs, transform, shuffle):
        ds = ImageDataset(paths[idxs].tolist(), yt[idxs], yp[idxs], transform)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                          num_workers=4, pin_memory=True, persistent_workers=True)

    print(f"  Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")
    return (
        make_loader(idx_train, TRAIN_TRANSFORM, shuffle=True),
        make_loader(idx_val,   VAL_TRANSFORM,   shuffle=False),
        make_loader(idx_test,  VAL_TRANSFORM,   shuffle=False),
        class_names,
    )


# ── Models ────────────────────────────────────────────────────────
class CLIPFinetune(nn.Module):
    def __init__(self, n_classes=6, dropout=0.3):
        super().__init__()
        self.backbone, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        dim = 512
        self.head_backbone = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head_people = nn.Linear(128, 1)
        self.head_type   = nn.Linear(128, n_classes)

    def encode(self, x):
        # open_clip encode_image returns features before projection in some versions
        feats = self.backbone.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float()

    def forward(self, x):
        h = self.encode(x)
        h = self.head_backbone(h)
        return self.head_people(h).squeeze(-1), self.head_type(h)

    def param_groups(self):
        return [
            {"params": self.backbone.parameters(),    "lr": LR_BACKBONE},
            {"params": self.head_backbone.parameters(),"lr": LR_HEADS},
            {"params": self.head_people.parameters(),  "lr": LR_HEADS},
            {"params": self.head_type.parameters(),    "lr": LR_HEADS},
        ]


class ConvNeXtFinetune(nn.Module):
    def __init__(self, n_classes=6, dropout=0.3):
        super().__init__()
        base = models.convnext_tiny(weights="DEFAULT")
        # Remove classifier, keep features
        self.backbone = nn.Sequential(
            base.features,
            base.avgpool,
            nn.Flatten(),
        )
        dim = 768
        self.head_backbone = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head_people = nn.Linear(128, 1)
        self.head_type   = nn.Linear(128, n_classes)

    def forward(self, x):
        h = self.backbone(x)
        h = self.head_backbone(h)
        return self.head_people(h).squeeze(-1), self.head_type(h)

    def param_groups(self):
        return [
            {"params": self.backbone.parameters(),    "lr": LR_BACKBONE},
            {"params": self.head_backbone.parameters(),"lr": LR_HEADS},
            {"params": self.head_people.parameters(),  "lr": LR_HEADS},
            {"params": self.head_type.parameters(),    "lr": LR_HEADS},
        ]


# ── Training ──────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scaler):
    model.train()
    total_loss = 0
    for imgs, yt, yp in loader:
        imgs = imgs.to(DEVICE)
        yt   = yt.to(DEVICE)
        yp   = yp.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            pl, tl = model(imgs)
            loss = F.binary_cross_entropy_with_logits(pl, yp) + F.cross_entropy(tl, yt)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def compute_metrics(model, loader):
    model.eval()
    all_pl, all_tl, all_yp, all_yt = [], [], [], []
    total_loss = 0
    for imgs, yt, yp in loader:
        imgs = imgs.to(DEVICE); yt = yt.to(DEVICE); yp = yp.to(DEVICE)
        with torch.amp.autocast("cuda"):
            pl, tl = model(imgs)
            loss = F.binary_cross_entropy_with_logits(pl, yp) + F.cross_entropy(tl, yt)
        total_loss += loss.item() * imgs.size(0)
        all_pl.append(pl.cpu()); all_tl.append(tl.cpu())
        all_yp.append(yp.cpu()); all_yt.append(yt.cpu())

    pl = torch.cat(all_pl); tl = torch.cat(all_tl)
    yp = torch.cat(all_yp); yt = torch.cat(all_yt)
    pp = (pl > 0).long(); tp = tl.argmax(1)

    return {
        "loss":       total_loss / len(loader.dataset),
        "type_acc":   accuracy_score(yt, tp),
        "type_f1":    precision_recall_fscore_support(yt, tp, average="weighted", zero_division=0)[2],
        "people_f1":  precision_recall_fscore_support(yp, pp, average="binary",   zero_division=0)[2],
        "people_auc": roc_auc_score(yp, pl) if yp.sum() > 0 else 0.0,
    }


@torch.no_grad()
def full_report(model, loader, class_names, label=""):
    model.eval()
    all_pl, all_tl, all_yp, all_yt = [], [], [], []
    for imgs, yt, yp in loader:
        with torch.amp.autocast("cuda"):
            pl, tl = model(imgs.to(DEVICE))
        all_pl.append(pl.cpu()); all_tl.append(tl.cpu())
        all_yp.append(yp); all_yt.append(yt)

    pl = torch.cat(all_pl); tl = torch.cat(all_tl)
    yp = torch.cat(all_yp); yt = torch.cat(all_yt)
    pp = (pl > 0).long(); tp = tl.argmax(1)

    p_acc = accuracy_score(yp, pp)
    _, _, p_f1, _ = precision_recall_fscore_support(yp, pp, average="binary", zero_division=0)
    p_auc = roc_auc_score(yp, pl)
    t_acc = accuracy_score(yt, tp)
    _, _, t_f1, _ = precision_recall_fscore_support(yt, tp, average="weighted", zero_division=0)

    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    print(f"  has_people: Acc={p_acc:.4f}  F1={p_f1:.4f}  AUC={p_auc:.4f}")
    print(f"  image_type: Acc={t_acc:.4f}  F1={t_f1:.4f}")
    print(f"\n{classification_report(yt, tp, target_names=class_names, zero_division=0)}")

    return {
        "people_acc": p_acc, "people_f1": float(p_f1), "people_auc": float(p_auc),
        "type_acc":   t_acc, "type_f1":   float(t_f1),
        "cm": confusion_matrix(yt, tp),
    }


def run_finetune(name, model_class, class_names, train_loader, val_loader, test_loader):
    print(f"\n{'#'*60}\n  FINE-TUNE: {name}\n{'#'*60}")
    model = model_class(n_classes=len(class_names)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params total={n_params:,}  trainable={n_trainable:,}")

    # Sanitize name for filesystem (replace /, \, spaces, parens)
    safe_name = (name.lower()
                 .replace(" ", "_").replace("/", "-").replace("\\", "-")
                 .replace("(", "").replace(")", ""))
    Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)
    ckpt_path = f"{CKPT_DIR}/{safe_name}.pt"

    optimizer = torch.optim.AdamW(model.param_groups(), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = torch.amp.GradScaler("cuda")

    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "val_type_f1": [], "val_people_f1": []}
    best_val_f1 = 0
    best_state  = None
    no_improve  = 0

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scaler)
        scheduler.step()
        val_m = compute_metrics(model, val_loader)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_m["loss"])
        history["val_type_f1"].append(val_m["type_f1"])
        history["val_people_f1"].append(val_m["people_f1"])

        if val_m["type_f1"] > best_val_f1:
            best_val_f1 = val_m["type_f1"]
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_path)  # save incrementally
            no_improve  = 0
        else:
            no_improve += 1

        elapsed = (time.time() - t0) / 60
        print(f"  Epoch {epoch:3d}: train={train_loss:.4f}  val_loss={val_m['loss']:.4f}"
              f"  type_f1={val_m['type_f1']:.4f}  people_f1={val_m['people_f1']:.4f}"
              f"  [{elapsed:.1f}min]")

        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    model.load_state_dict(best_state)
    print(f"\n  Best val type_f1: {best_val_f1:.4f}")

    print(f"  Checkpoint saved: {ckpt_path}")

    test_r = full_report(model, test_loader, class_names, f"{name} - TEST")
    return {"name": name, "params": n_params, "history": history, "test": test_r}


# ── Plots ─────────────────────────────────────────────────────────
def plot_results(experiments):
    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Fine-Tune Training Curves", fontsize=14, fontweight="bold")
    for ax, (key, title) in zip(axes, [
        ("val_loss", "Val Loss"),
        ("val_type_f1", "Val Type F1"),
        ("val_people_f1", "Val People F1"),
    ]):
        for exp in experiments:
            ax.plot(exp["history"]["epoch"], exp["history"][key], label=exp["name"], linewidth=1.8)
        ax.set_xlabel("Epoch"); ax.set_ylabel(title); ax.set_title(title)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/finetune_curves.png", dpi=150)
    plt.close()

    # Confusion matrices
    n = len(experiments)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 6))
    if n == 1: axes = [axes]
    for ax, exp in zip(axes, experiments):
        cm = exp["test"]["cm"]
        class_names = exp.get("class_names", [str(i) for i in range(cm.shape[0])])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                    xticklabels=[c[:10] for c in class_names],
                    yticklabels=[c[:10] for c in class_names],
                    ax=ax, cbar=False, square=True)
        ax.set_title(f"{exp['name']}\nTest Acc={exp['test']['type_acc']:.3f}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/finetune_confusion.png", dpi=150)
    plt.close()
    print(f"Plots saved to {OUTPUT_DIR}/")


# ── Main ──────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}  ({torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'cpu'})")

    train_loader, val_loader, test_loader, class_names = load_splits()
    print(f"Classes: {class_names}")

    experiments = []

    # 1. Fine-tune CLIP
    exp1 = run_finetune("FT CLIP ViT-B/32", CLIPFinetune, class_names,
                        train_loader, val_loader, test_loader)
    exp1["class_names"] = class_names
    experiments.append(exp1)

    # 2. Fine-tune ConvNeXt
    exp2 = run_finetune("FT ConvNeXt-Tiny", ConvNeXtFinetune, class_names,
                        train_loader, val_loader, test_loader)
    exp2["class_names"] = class_names
    experiments.append(exp2)

    # Summary
    print(f"\n{'#'*60}\n  FINE-TUNE RESULTS (test set)\n{'#'*60}")
    print(f"  {'':20s} {'FT CLIP':>10s} {'FT ConvNeXt':>12s}")
    for k in ["type_acc", "type_f1", "people_f1", "people_auc"]:
        vals = [e["test"][k] for e in experiments]
        print(f"  {k:20s} {vals[0]:10.4f} {vals[1]:12.4f}")

    # Save metrics
    summary = {e["name"]: {"params": e["params"],
               "test": {k: v for k, v in e["test"].items() if k != "cm"}}
               for e in experiments}
    with open(f"{OUTPUT_DIR}/finetune_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_results(experiments)
    print("\nAll done.")


if __name__ == "__main__":
    main()
