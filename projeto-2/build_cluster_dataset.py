"""
Build and train a supervised dataset from human-approved cluster annotations.

Input:
  output/hierarchical/hierarchy.json
  cluster_annotations.json
  embeddings_all.bin

Output:
  dataset_cluster/
    X.npy
    y.npy
    labels.json
    paths.txt
    split.json
    metrics.json
    confusion_matrix.png

Only clusters marked usable_for_classification=true are used. The final class is
annotation.label_final. Old deterministic labels are ignored.
"""
from __future__ import annotations

import argparse
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build classifier dataset from annotated clusters.")
    p.add_argument("--hierarchy", default="output/hierarchical/hierarchy.json")
    p.add_argument("--annotations", default="cluster_annotations.json")
    p.add_argument("--embeddings", default="embeddings_all.bin")
    p.add_argument("--paths", default="all_paths.txt")
    p.add_argument("--output", default="dataset_cluster")
    p.add_argument("--min-class-size", type=int, default=20)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--no-train", action="store_true", help="Only build arrays/splits; do not train baseline MLP.")
    return p.parse_args()


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_lines(path: str | Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_embeddings(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32, copy=False)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        dim = struct.unpack("<Q", f.read(8))[0]
        arr = np.frombuffer(f.read(n * dim * 4), dtype=np.float32)
    return arr.reshape(n, dim)


def flatten_leaves(hierarchy: Dict[str, Any]) -> List[Dict[str, Any]]:
    leaves: List[Dict[str, Any]] = []
    for macro in hierarchy.get("clusters", []):
        for leaf in macro.get("subclusters", []):
            leaves.append(leaf)
    return leaves


def collect_indices(
    hierarchy: Dict[str, Any],
    annotations: Dict[str, Any],
    min_class_size: int,
) -> Tuple[List[int], List[str], Dict[str, Any]]:
    by_label: Dict[str, List[int]] = defaultdict(list)
    used_clusters: Dict[str, Dict[str, Any]] = {}
    skipped_clusters: Dict[str, str] = {}

    for leaf in flatten_leaves(hierarchy):
        cid = leaf["id"]
        ann = annotations.get(cid, {})
        label = str(ann.get("label_final", "")).strip()
        usable = bool(ann.get("usable_for_classification", False))
        if not usable:
            skipped_clusters[cid] = "not usable_for_classification"
            continue
        if not label:
            skipped_clusters[cid] = "missing label_final"
            continue
        indices = [int(i) for i in leaf.get("image_indices", [])]
        by_label[label].extend(indices)
        used_clusters[cid] = {
            "label_final": label,
            "n": len(indices),
            "needs_split": bool(ann.get("needs_split", False)),
            "notes": ann.get("notes", ""),
        }

    kept_labels = {label for label, idxs in by_label.items() if len(idxs) >= min_class_size}
    low_support = {label: len(idxs) for label, idxs in by_label.items() if label not in kept_labels}

    final_indices: List[int] = []
    final_labels: List[str] = []
    for label in sorted(kept_labels):
        for idx in by_label[label]:
            final_indices.append(idx)
            final_labels.append(label)

    manifest = {
        "n_samples": len(final_indices),
        "class_counts": dict(Counter(final_labels)),
        "used_clusters": used_clusters,
        "skipped_clusters": skipped_clusters,
        "low_support_labels": low_support,
        "min_class_size": min_class_size,
    }
    return final_indices, final_labels, manifest


def make_split(y: np.ndarray, test_size: float, val_size: float) -> Dict[str, List[int]]:
    indices = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    relative_val = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val,
        stratify=y[train_val_idx],
        random_state=RANDOM_STATE,
    )
    return {
        "train": train_idx.astype(int).tolist(),
        "val": val_idx.astype(int).tolist(),
        "test": test_idx.astype(int).tolist(),
    }


def train_baseline(X: np.ndarray, y: np.ndarray, split: Dict[str, List[int]], class_names: List[str], out_dir: Path) -> Dict[str, Any]:
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=120,
        early_stopping=True,
        n_iter_no_change=12,
        random_state=RANDOM_STATE,
        verbose=True,
    )
    clf.fit(X[split["train"]], y[split["train"]])
    metrics: Dict[str, Any] = {"n_iter": int(clf.n_iter_), "classes": class_names}
    for name, idx in split.items():
        pred = clf.predict(X[idx])
        metrics[name] = {
            "accuracy": float(accuracy_score(y[idx], pred)),
            "macro_f1": float(f1_score(y[idx], pred, average="macro")),
            "weighted_f1": float(f1_score(y[idx], pred, average="weighted")),
            "report": classification_report(y[idx], pred, target_names=class_names, output_dict=True, zero_division=0),
        }
    cm = confusion_matrix(y[split["test"]], clf.predict(X[split["test"]]), labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(max(7, len(class_names) * 0.7), max(6, len(class_names) * 0.65)), dpi=140)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Cluster-label classifier — test confusion matrix")
    ax.set_xlabel("predito")
    ax.set_ylabel("real")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)
    return metrics


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    hierarchy = load_json(args.hierarchy)
    annotations = load_json(args.annotations)
    all_paths = load_lines(args.paths)
    X_all = load_embeddings(args.embeddings)

    selected_indices, labels_str, manifest = collect_indices(hierarchy, annotations, args.min_class_size)
    if not selected_indices:
        raise SystemExit(
            "No samples selected. Annotate clusters first with cluster_annotator.py and mark usable_for_classification=true."
        )
    if len(set(labels_str)) < 2:
        raise SystemExit("Need at least two usable classes after min_class_size filtering.")

    class_names = sorted(set(labels_str))
    label_to_id = {label: i for i, label in enumerate(class_names)}
    y = np.array([label_to_id[x] for x in labels_str], dtype=np.int64)
    X = X_all[np.array(selected_indices, dtype=np.int64)].astype(np.float32, copy=False)
    selected_paths = [all_paths[i] for i in selected_indices]
    split = make_split(y, args.test_size, args.val_size)

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    with open(out_dir / "paths.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(selected_paths) + "\n")
    save_json(out_dir / "labels.json", {"class_names": class_names, "label_to_id": label_to_id})
    save_json(out_dir / "split.json", split)
    save_json(out_dir / "manifest.json", manifest)

    print(f"Built dataset: X={X.shape}, classes={len(class_names)}")
    print("Class counts:", Counter(labels_str))

    if not args.no_train:
        metrics = train_baseline(X, y, split, class_names, out_dir)
        save_json(out_dir / "metrics.json", metrics)
        print("Baseline test:", metrics["test"])


if __name__ == "__main__":
    main()
