"""
Linear probe: can a single linear layer predict image_type + has_people
from CLIP embeddings alone?

Multi-task output: 7 logits = 1 (people) + 6 (image_type)
Loss: BCEWithLogitsLoss(people) + CrossEntropyLoss(type)
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

DATASET_DIR = "dataset"
SEED = 42
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data():
    X = np.load(f"{DATASET_DIR}/X_embeddings.npy")
    y_type = np.load(f"{DATASET_DIR}/y_type.npy")
    y_people = np.load(f"{DATASET_DIR}/y_people.npy")
    with open(f"{DATASET_DIR}/class_names.json") as f:
        class_names = json.load(f)
    return X, y_type, y_people, class_names


class LinearProbe(nn.Module):
    """Single linear layer: 512 -> 7 logits (1 people + 6 type)."""
    def __init__(self, in_dim=512, n_type_classes=6):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1 + n_type_classes)
        self.n_type = n_type_classes

    def forward(self, x):
        out = self.linear(x)
        return out[:, 0], out[:, 1:]  # people_logit, type_logits


def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for X_batch, y_type_batch, y_people_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_type_batch = y_type_batch.to(DEVICE)
        y_people_batch = y_people_batch.to(DEVICE)

        people_logit, type_logits = model(X_batch)

        loss_people = F.binary_cross_entropy_with_logits(people_logit, y_people_batch)
        loss_type = F.cross_entropy(type_logits, y_type_batch)
        loss = loss_people + loss_type

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, class_names):
    model.eval()
    all_people_logits, all_type_logits = [], []
    all_y_people, all_y_type = [], []

    for X_batch, y_type_batch, y_people_batch in loader:
        X_batch = X_batch.to(DEVICE)
        people_logit, type_logits = model(X_batch)
        all_people_logits.append(people_logit.cpu())
        all_type_logits.append(type_logits.cpu())
        all_y_people.append(y_people_batch)
        all_y_type.append(y_type_batch)

    people_logits = torch.cat(all_people_logits)
    type_logits = torch.cat(all_type_logits)
    y_people = torch.cat(all_y_people)
    y_type = torch.cat(all_y_type)

    # People predictions
    people_pred = (people_logits > 0).long()
    people_acc = accuracy_score(y_people.numpy(), people_pred.numpy())
    people_prec, people_rec, people_f1, _ = precision_recall_fscore_support(
        y_people.numpy(), people_pred.numpy(), average="binary", zero_division=0
    )
    people_auc = roc_auc_score(y_people.numpy(), people_logits.numpy())

    # Type predictions
    type_pred = type_logits.argmax(dim=1)
    type_acc = accuracy_score(y_type.numpy(), type_pred.numpy())
    type_prec, type_rec, type_f1, _ = precision_recall_fscore_support(
        y_type.numpy(), type_pred.numpy(), average="weighted", zero_division=0
    )

    print("\n" + "=" * 60)
    print("HEAD 1: has_people (binary)")
    print(f"  Accuracy:  {people_acc:.4f}")
    print(f"  Precision: {people_prec:.4f}")
    print(f"  Recall:    {people_rec:.4f}")
    print(f"  F1:        {people_f1:.4f}")
    print(f"  AUC:       {people_auc:.4f}")

    print("\nHEAD 2: image_type (6 classes)")
    print(f"  Accuracy:  {type_acc:.4f}")
    print(f"  Precision: {type_prec:.4f}  (weighted)")
    print(f"  Recall:    {type_rec:.4f}  (weighted)")
    print(f"  F1:        {type_f1:.4f}  (weighted)")

    print("\n--- Classification Report (image_type) ---")
    print(classification_report(
        y_type.numpy(), type_pred.numpy(),
        target_names=class_names, zero_division=0
    ))

    print("--- Confusion Matrix (image_type) ---")
    cm = confusion_matrix(y_type.numpy(), type_pred.numpy())
    # Pretty print
    header = "".join(f"{n[:8]:>10s}" for n in class_names)
    print(f"{'pred>>':>20s}{header}")
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:10d}" for v in row)
        print(f"{class_names[i]:>20s}{row_str}")

    return {
        "people_acc": people_acc, "people_f1": people_f1, "people_auc": people_auc,
        "type_acc": type_acc, "type_f1": type_f1,
    }


def main():
    print(f"Device: {DEVICE}")
    X, y_type, y_people, class_names = load_data()
    print(f"Data: X={X.shape}, classes={class_names}")

    # Stratified split
    X_train, X_test, yt_train, yt_test, yp_train, yp_test = train_test_split(
        X, y_type, y_people, test_size=0.2, random_state=SEED, stratify=y_type
    )
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Tensors
    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(yt_train),
        torch.from_numpy(yp_train),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(yt_test),
        torch.from_numpy(yp_test),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Model
    model = LinearProbe(in_dim=512, n_type_classes=len(class_names)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"\nTraining linear probe ({sum(p.numel() for p in model.parameters())} params)...")
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}: loss={loss:.4f}")

    print("\n=== TRAIN SET ===")
    evaluate(model, train_loader, class_names)

    print("\n=== TEST SET ===")
    metrics = evaluate(model, test_loader, class_names)

    # Save
    torch.save(model.state_dict(), f"{DATASET_DIR}/../linear_probe.pt")
    with open(f"{DATASET_DIR}/../linear_probe_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nModel + metrics saved.")


if __name__ == "__main__":
    main()
