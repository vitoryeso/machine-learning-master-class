"""
ConvNeXt-Tiny as frozen feature extractor + MLP dual-head probe.
Pipeline: raw images -> ConvNeXt-Tiny (frozen, no classifier) -> 768-dim -> MLP -> 2 heads
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score,
)
from PIL import Image
from pathlib import Path

DATASET_DIR = "D:/media/machine-learning-master-class/projeto-2/dataset"
PATHS_FILE = "D:/media/cluster_output/paths.txt"
SEED = 42
BATCH_SIZE = 64
EXTRACT_BATCH = 32
LR = 1e-3
EPOCHS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURES_FILE = f"{DATASET_DIR}/X_convnext.npy"


# === Feature extraction ===

class ImagePathDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img), idx
        except Exception:
            # Return black image on error
            return torch.zeros(3, 224, 224), idx


def extract_features():
    """Extract 768-dim features from ConvNeXt-Tiny for all images."""
    print("Loading ConvNeXt-Tiny...")
    model = models.convnext_tiny(weights="DEFAULT")
    # Remove classifier, keep up to global pool
    model.classifier = nn.Sequential(
        model.classifier[0],  # LayerNorm2d
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    model = model.to(DEVICE).eval()

    transform = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with open(PATHS_FILE, "r", encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip()]

    ds = ImagePathDataset(paths, transform)
    loader = DataLoader(ds, batch_size=EXTRACT_BATCH, shuffle=False, num_workers=0)

    all_features = np.zeros((len(paths), 768), dtype=np.float32)
    print(f"Extracting features from {len(paths)} images...")

    with torch.no_grad():
        for batch_idx, (imgs, indices) in enumerate(loader):
            imgs = imgs.to(DEVICE)
            feats = model(imgs).cpu().numpy()
            for i, idx in enumerate(indices):
                all_features[idx] = feats[i]
            if (batch_idx + 1) % 25 == 0 or batch_idx == 0:
                done = min((batch_idx + 1) * EXTRACT_BATCH, len(paths))
                print(f"  [{done}/{len(paths)}]")

    np.save(FEATURES_FILE, all_features)
    print(f"Saved features: {all_features.shape} -> {FEATURES_FILE}")
    return all_features


# === MLP Probe (same architecture, 768-dim input) ===

class MLPProbe(nn.Module):
    def __init__(self, in_dim=768, hidden1=256, hidden2=128, n_type_classes=6, dropout=0.3):
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
def evaluate(model, loader, class_names, label=""):
    model.eval()
    all_pl, all_tl, all_yp, all_yt = [], [], [], []
    for X_b, yt_b, yp_b in loader:
        pl, tl = model(X_b.to(DEVICE))
        all_pl.append(pl.cpu()); all_tl.append(tl.cpu())
        all_yp.append(yp_b); all_yt.append(yt_b)

    pl = torch.cat(all_pl); tl = torch.cat(all_tl)
    yp = torch.cat(all_yp); yt = torch.cat(all_yt)

    pp = (pl > 0).long()
    p_acc = accuracy_score(yp, pp)
    p_prec, p_rec, p_f1, _ = precision_recall_fscore_support(yp, pp, average="binary", zero_division=0)
    p_auc = roc_auc_score(yp, pl)

    tp = tl.argmax(dim=1)
    t_acc = accuracy_score(yt, tp)
    t_prec, t_rec, t_f1, _ = precision_recall_fscore_support(yt, tp, average="weighted", zero_division=0)

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"\nHEAD 1: has_people")
    print(f"  Acc={p_acc:.4f}  Prec={p_prec:.4f}  Rec={p_rec:.4f}  F1={p_f1:.4f}  AUC={p_auc:.4f}")
    print(f"\nHEAD 2: image_type")
    print(f"  Acc={t_acc:.4f}  Prec={t_prec:.4f}  Rec={t_rec:.4f}  F1={t_f1:.4f}")
    print(f"\n{classification_report(yt, tp, target_names=class_names, zero_division=0)}")

    cm = confusion_matrix(yt, tp)
    header = "".join(f"{n[:8]:>10s}" for n in class_names)
    print(f"{'':>20s}{header}")
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>20s}{''.join(f'{v:10d}' for v in row)}")

    return {"people_acc": p_acc, "people_f1": p_f1, "people_auc": p_auc, "type_acc": t_acc, "type_f1": t_f1}


def main():
    print(f"Device: {DEVICE}")

    # Extract or load features
    try:
        X = np.load(FEATURES_FILE)
        print(f"Loaded cached features: {X.shape}")
    except FileNotFoundError:
        X = extract_features()

    y_type = np.load(f"{DATASET_DIR}/y_type.npy")
    y_people = np.load(f"{DATASET_DIR}/y_people.npy")
    with open(f"{DATASET_DIR}/class_names.json") as f:
        class_names = json.load(f)

    X_train, X_test, yt_train, yt_test, yp_train, yp_test = train_test_split(
        X, y_type, y_people, test_size=0.2, random_state=SEED, stratify=y_type
    )
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(yt_train), torch.from_numpy(yp_train)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(yt_test), torch.from_numpy(yp_test)),
        batch_size=BATCH_SIZE,
    )

    model = MLPProbe(in_dim=768, n_type_classes=len(class_names)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MLPProbe (ConvNeXt features): {n_params:,} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer)
        scheduler.step()
        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}: loss={loss:.4f}  lr={scheduler.get_last_lr()[0]:.6f}")

    # Evaluate
    print("\n" + "#" * 60)
    evaluate(model, train_loader, class_names, label="TRAIN SET (ConvNeXt)")
    metrics = evaluate(model, test_loader, class_names, label="TEST SET (ConvNeXt)")

    # 3-way comparison
    print("\n" + "#" * 60)
    print("  LINEAR (CLIP) vs MLP (CLIP) vs MLP (ConvNeXt)  [test set]")
    print("#" * 60)
    try:
        with open("D:/media/machine-learning-master-class/projeto-2/linear_probe_metrics.json") as f:
            lp = json.load(f)
        with open("D:/media/machine-learning-master-class/projeto-2/mlp_probe_metrics.json") as f:
            mlp = json.load(f)
        cn = metrics
        print(f"  {'':20s} {'Lin(CLIP)':>10s} {'MLP(CLIP)':>10s} {'MLP(CNeXt)':>10s}")
        for k in ["type_acc", "type_f1", "people_acc", "people_f1", "people_auc"]:
            print(f"  {k:20s} {lp[k]:10.4f} {mlp[k]:10.4f} {cn[k]:10.4f}")
    except FileNotFoundError:
        print("  (previous metrics not found)")

    torch.save(model.state_dict(), "D:/media/machine-learning-master-class/projeto-2/convnext_probe.pt")
    with open("D:/media/machine-learning-master-class/projeto-2/convnext_probe_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nModel + metrics saved.")


if __name__ == "__main__":
    main()
