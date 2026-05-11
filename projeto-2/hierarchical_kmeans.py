"""
Hierarchical CLIP clustering for Projeto 2.

This replaces the old deterministic-label classification setup with a
zero-trust discovery pipeline:

1. Load CLIP image embeddings for the full dataset.
2. Scan K for a global KMeans and choose K from clustering metrics.
3. Fit the selected global KMeans.
4. For each global cluster, independently scan and choose a sub-K.
5. Save hierarchy data, metric tables, scatter plots, and contact sheets.

The script intentionally does NOT "invert" CLIP embeddings into text.
Cluster descriptions are built from representative images and metadata.

Example:
  python hierarchical_kmeans.py
  python hierarchical_kmeans.py --k-max 40 --sub-k-max 12
  python hierarchical_kmeans.py --limit 1200 --k-max 8 --output output/hierarchical_smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import struct
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hierarchical KMeans over CLIP embeddings.")
    p.add_argument("--embeddings", default="embeddings_all.bin")
    p.add_argument("--paths", default="all_paths.txt")
    p.add_argument("--viz-paths", default="all_paths_256.txt")
    p.add_argument("--metadata", default="metadata.json")
    p.add_argument("--output", default="output/hierarchical")
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=80)
    p.add_argument("--sub-k-min", type=int, default=2)
    p.add_argument("--sub-k-max", type=int, default=16)
    p.add_argument("--min-cluster-size", type=int, default=100)
    p.add_argument("--min-subcluster-size", type=int, default=30)
    p.add_argument("--sample-size", type=int, default=5000)
    p.add_argument("--plot-sample-size", type=int, default=8000)
    p.add_argument("--representatives", type=int, default=12)
    p.add_argument("--sheet-representatives", type=int, default=8)
    p.add_argument("--thumb-size", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--n-init", type=int, default=5)
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Debug/smoke-test limit. 0 means full dataset.",
    )
    p.add_argument(
        "--skip-umap",
        action="store_true",
        help="Do not try to generate UMAP even if umap-learn is installed.",
    )
    return p.parse_args()


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def json_dump(obj: Any, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_embeddings(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32, copy=False)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        dim = struct.unpack("<Q", f.read(8))[0]
        arr = np.frombuffer(f.read(n * dim * 4), dtype=np.float32)
    return arr.reshape(n, dim)


def load_lines(path: str | Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_metadata(path: str | Path, n: int) -> List[Dict[str, Any]]:
    if not Path(path).exists():
        return [{} for _ in range(n)]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if len(data) != n:
        print(f"[warn] metadata length {len(data)} != embeddings length {n}; using available slice.")
    out: List[Dict[str, Any]] = []
    for i in range(n):
        out.append(data[i] if i < len(data) and isinstance(data[i], dict) else {})
    return out


def maybe_limit(
    X: np.ndarray,
    paths: List[str],
    viz_paths: List[str],
    metadata: List[Dict[str, Any]],
    limit: int,
) -> Tuple[np.ndarray, List[str], List[str], List[Dict[str, Any]], np.ndarray]:
    n = len(X)
    original_indices = np.arange(n, dtype=np.int64)
    if limit and limit < n:
        rng = np.random.default_rng(RANDOM_STATE)
        sel = np.sort(rng.choice(n, size=limit, replace=False))
        return X[sel], [paths[i] for i in sel], [viz_paths[i] for i in sel], [metadata[i] for i in sel], sel
    return X, paths, viz_paths, metadata, original_indices


def sample_indices(n: int, sample_size: int) -> np.ndarray:
    if n <= sample_size:
        return np.arange(n)
    rng = np.random.default_rng(RANDOM_STATE)
    return np.sort(rng.choice(n, size=sample_size, replace=False))


def fit_kmeans(X: np.ndarray, k: int, batch_size: int, n_init: int) -> MiniBatchKMeans:
    return MiniBatchKMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        batch_size=min(batch_size, max(256, len(X))),
        n_init=n_init,
        max_iter=200,
        reassignment_ratio=0.01,
        verbose=0,
    ).fit(X)


def safe_metric_scores(X_eval: np.ndarray, labels_eval: np.ndarray) -> Dict[str, Optional[float]]:
    n_unique = len(set(labels_eval.tolist()))
    if n_unique < 2 or n_unique >= len(labels_eval):
        return {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}
    out: Dict[str, Optional[float]] = {}
    try:
        out["silhouette"] = float(silhouette_score(X_eval, labels_eval, metric="euclidean"))
    except Exception as e:
        print(f"[warn] silhouette failed: {e}")
        out["silhouette"] = None
    try:
        out["davies_bouldin"] = float(davies_bouldin_score(X_eval, labels_eval))
    except Exception as e:
        print(f"[warn] davies_bouldin failed: {e}")
        out["davies_bouldin"] = None
    try:
        out["calinski_harabasz"] = float(calinski_harabasz_score(X_eval, labels_eval))
    except Exception as e:
        print(f"[warn] calinski_harabasz failed: {e}")
        out["calinski_harabasz"] = None
    return out


def cluster_balance(counts: Sequence[int]) -> float:
    total = float(sum(counts))
    if total <= 0 or len(counts) <= 1:
        return 0.0
    probs = np.array(counts, dtype=np.float64) / total
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    return entropy / math.log(len(counts))


def scan_k(
    X: np.ndarray,
    k_values: Iterable[int],
    min_cluster_size: int,
    sample_size: int,
    batch_size: int,
    n_init: int,
    label: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    eval_idx = sample_indices(len(X), sample_size)
    X_eval = X[eval_idx]
    for k in k_values:
        if k < 2 or k >= len(X):
            continue
        if len(X) // k < max(2, min_cluster_size // 2):
            continue
        t0 = time.time()
        print(f"[{label}] scanning K={k} on n={len(X)}")
        km = fit_kmeans(X, k, batch_size, n_init)
        labels = km.labels_
        counts = np.bincount(labels, minlength=k).astype(int)
        labels_eval = labels[eval_idx]
        metrics = safe_metric_scores(X_eval, labels_eval)
        row: Dict[str, Any] = {
            "k": int(k),
            "inertia": float(km.inertia_),
            "min_cluster_size": int(counts.min()),
            "max_cluster_size": int(counts.max()),
            "cluster_sizes": counts.tolist(),
            "balance": float(cluster_balance(counts)),
            "valid_min_size": bool(counts.min() >= min_cluster_size),
            "elapsed_sec": round(time.time() - t0, 3),
        }
        row.update(metrics)
        rows.append(row)
    score_candidates(rows)
    return rows


def norm_values(values: List[Optional[float]], higher_is_better: bool = True) -> List[float]:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return [0.0 for _ in values]
    lo, hi = min(finite), max(finite)
    if abs(hi - lo) < 1e-12:
        return [0.5 if v is not None and math.isfinite(v) else 0.0 for v in values]
    out = []
    for v in values:
        if v is None or not math.isfinite(v):
            out.append(0.0)
        else:
            z = (v - lo) / (hi - lo)
            out.append(float(z if higher_is_better else 1.0 - z))
    return out


def score_candidates(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    sil = norm_values([r.get("silhouette") for r in rows], True)
    db = norm_values([r.get("davies_bouldin") for r in rows], False)
    ch = norm_values([r.get("calinski_harabasz") for r in rows], True)
    bal = norm_values([r.get("balance") for r in rows], True)
    inertia_inv = norm_values([r.get("inertia") for r in rows], False)
    # Composite emphasizes separation metrics but keeps a small balance/elbow proxy term.
    for i, r in enumerate(rows):
        min_penalty = 0.0 if r.get("valid_min_size") else -0.25
        r["score"] = float(
            0.36 * sil[i]
            + 0.24 * db[i]
            + 0.20 * ch[i]
            + 0.12 * bal[i]
            + 0.08 * inertia_inv[i]
            + min_penalty
        )
    rows.sort(key=lambda r: (r["score"], r.get("silhouette") or -999.0), reverse=True)
    for rank, r in enumerate(rows, start=1):
        r["rank"] = rank


def select_k(rows: List[Dict[str, Any]], min_cluster_size: int) -> Dict[str, Any]:
    valid = [r for r in rows if r.get("valid_min_size")]
    pool = valid or rows
    if not pool:
        raise RuntimeError("No K candidates were evaluated.")
    best = max(pool, key=lambda r: (r.get("score", -999), r.get("silhouette") or -999))
    return {
        "selected_k": int(best["k"]),
        "reason": (
            "best composite score among valid K values"
            if valid
            else f"no candidate met min_cluster_size={min_cluster_size}; selected best relaxed candidate"
        ),
        "metrics": {k: best.get(k) for k in [
            "score",
            "silhouette",
            "davies_bouldin",
            "calinski_harabasz",
            "inertia",
            "min_cluster_size",
            "max_cluster_size",
            "balance",
        ]},
        "top_candidates": rows[:10],
    }


def write_metrics_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = [
        "rank",
        "k",
        "score",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "inertia",
        "min_cluster_size",
        "max_cluster_size",
        "balance",
        "valid_min_size",
        "elapsed_sec",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda x: x["k"]):
            w.writerow({k: r.get(k) for k in fields})


def plot_k_scan(rows: List[Dict[str, Any]], selected_k: int, path: Path, title: str) -> None:
    rows_by_k = sorted(rows, key=lambda r: r["k"])
    ks = [r["k"] for r in rows_by_k]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=140)
    plots = [
        ("inertia", "Inertia", axes[0, 0]),
        ("silhouette", "Silhouette ↑", axes[0, 1]),
        ("davies_bouldin", "Davies-Bouldin ↓", axes[1, 0]),
        ("score", "Composite score ↑", axes[1, 1]),
    ]
    for key, ylabel, ax in plots:
        vals = [r.get(key) for r in rows_by_k]
        ax.plot(ks, vals, marker="o", linewidth=2)
        ax.axvline(selected_k, color="crimson", linestyle="--", label=f"selected K={selected_k}")
        ax.set_xlabel("K")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def top_counts(values: Iterable[Any], n: int = 6) -> List[Dict[str, Any]]:
    c = Counter(str(v) if v not in (None, "") else "(unknown)" for v in values)
    total = sum(c.values()) or 1
    return [
        {"value": value, "count": int(count), "pct": round(100.0 * count / total, 2)}
        for value, count in c.most_common(n)
    ]


def summarize_indices(indices: Sequence[int], paths: List[str], metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
    folders = [metadata[i].get("folder") or Path(paths[i]).parent.name for i in indices]
    exts = [metadata[i].get("ext") or Path(paths[i]).suffix.lower() for i in indices]
    buckets = [metadata[i].get("bucket") or "unknown" for i in indices]
    fsize = [float(metadata[i].get("fsize_kb", 0) or 0) for i in indices]
    return {
        "n": int(len(indices)),
        "top_folders": top_counts(folders),
        "extensions": top_counts(exts),
        "aspect_buckets": top_counts(buckets),
        "avg_fsize_kb": round(float(np.mean(fsize)), 2) if fsize else 0.0,
        "median_fsize_kb": round(float(np.median(fsize)), 2) if fsize else 0.0,
        "sample_files": [Path(paths[i]).name for i in list(indices)[:8]],
    }


def representative_indices(
    X: np.ndarray,
    indices: np.ndarray,
    center: np.ndarray,
    n: int,
) -> List[int]:
    Xc = X[indices]
    d = np.sum((Xc - center.reshape(1, -1)) ** 2, axis=1)
    order = np.argsort(d)[:n]
    return [int(indices[i]) for i in order]


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def open_thumb(path: str, size: int) -> Image.Image:
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((size, size), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (size, size), (245, 245, 245))
            x = (size - im.width) // 2
            y = (size - im.height) // 2
            canvas.paste(im, (x, y))
            return canvas
    except Exception:
        canvas = Image.new("RGB", (size, size), (40, 40, 40))
        d = ImageDraw.Draw(canvas)
        d.text((10, size // 2 - 10), "erro", fill=(255, 255, 255), font=load_font(18))
        return canvas


def wrap_text(text: str, max_chars: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    cur = ""
    for word in words:
        if len(cur) + len(word) + 1 <= max_chars:
            cur = f"{cur} {word}".strip()
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [""]


def make_contact_sheet(
    rows: List[Dict[str, Any]],
    viz_paths: List[str],
    out_path: Path,
    thumb_size: int = 150,
    max_images_per_row: int = 8,
    title: str = "",
) -> None:
    if not rows:
        return
    font_big = load_font(26)
    font = load_font(20)
    font_small = load_font(16)
    label_w = 420
    gap = 10
    row_h = thumb_size + 70
    title_h = 52 if title else 0
    width = label_w + max_images_per_row * (thumb_size + gap) + gap
    height = title_h + len(rows) * row_h + gap
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    y = 0
    if title:
        draw.rectangle([0, 0, width, title_h], fill=(15, 23, 42))
        draw.text((14, 10), title, fill=(255, 255, 255), font=font_big)
        y = title_h
    for row in rows:
        draw.rectangle([0, y, width, y + row_h - 3], fill=(248, 250, 252))
        label = row.get("label", "")
        summary = row.get("summary", "")
        draw.text((12, y + 10), label, fill=(0, 0, 0), font=font_big)
        for j, line in enumerate(wrap_text(summary, 36)[:4]):
            draw.text((12, y + 48 + j * 22), line, fill=(30, 41, 59), font=font_small)
        for col, idx in enumerate(row.get("indices", [])[:max_images_per_row]):
            x = label_w + col * (thumb_size + gap)
            thumb = open_thumb(viz_paths[idx], thumb_size)
            sheet.paste(thumb, (x, y + 10))
            draw.text((x + 2, y + thumb_size + 15), str(idx), fill=(60, 60, 60), font=font_small)
        y += row_h
    sheet.save(out_path)


def plot_pca_scatter(
    X: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    title: str,
    sample_size: int,
) -> None:
    idx = sample_indices(len(X), sample_size)
    Xs = X[idx]
    ls = labels[idx]
    xy = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(11, 9), dpi=140)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=ls, s=6, cmap="tab20", alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.grid(alpha=0.25)
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def maybe_plot_umap(
    X: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    title: str,
    sample_size: int,
    skip: bool,
) -> None:
    if skip:
        return
    try:
        import umap  # type: ignore
    except Exception:
        print("[info] umap-learn not installed; skipping UMAP plot.")
        return
    idx = sample_indices(len(X), sample_size)
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE, metric="euclidean")
    xy = reducer.fit_transform(X[idx])
    fig, ax = plt.subplots(figsize=(11, 9), dpi=140)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=labels[idx], s=6, cmap="tab20", alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.25)
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def format_top(items: List[Dict[str, Any]]) -> str:
    return ", ".join(f"{x['value']} {x['pct']:.0f}%" for x in items[:3])


def build_hierarchy(
    X: np.ndarray,
    paths: List[str],
    viz_paths: List[str],
    metadata: List[Dict[str, Any]],
    original_indices: np.ndarray,
    global_labels: np.ndarray,
    global_centers: np.ndarray,
    args: argparse.Namespace,
    out_dir: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    hierarchy: Dict[str, Any] = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_size": int(len(X)),
        "embedding_dim": int(X.shape[1]),
        "global_k": int(len(np.unique(global_labels))),
        "clusters": [],
    }
    macro_rows: List[Dict[str, Any]] = []
    global_sheet_rows: List[Dict[str, Any]] = []
    cluster_tree_lines: List[str] = [
        "Hierarchical KMeans cluster tree",
        f"N={len(X)} dim={X.shape[1]} K_global={hierarchy['global_k']}",
        "",
    ]

    for g in sorted(np.unique(global_labels).tolist()):
        gid = f"G{int(g):02d}"
        indices = np.where(global_labels == g)[0]
        summary = summarize_indices(indices.tolist(), paths, metadata)
        reps = representative_indices(X, indices, global_centers[g], args.representatives)
        macro = {
            "id": gid,
            "global_label": int(g),
            "image_indices": [int(original_indices[i]) for i in indices.tolist()],
            "local_indices": indices.astype(int).tolist(),
            "representative_indices": [int(original_indices[i]) for i in reps],
            "representative_local_indices": reps,
            "summary": summary,
            "sub_k": 1,
            "subclusters": [],
        }
        macro_rows.append({
            "cluster_id": gid,
            "n": summary["n"],
            "top_folders": format_top(summary["top_folders"]),
            "aspect_buckets": format_top(summary["aspect_buckets"]),
            "extensions": format_top(summary["extensions"]),
            "median_fsize_kb": summary["median_fsize_kb"],
        })
        global_sheet_rows.append({
            "label": f"{gid} · n={len(indices)}",
            "summary": f"folders: {format_top(summary['top_folders'])}; aspect: {format_top(summary['aspect_buckets'])}",
            "indices": reps,
        })
        cluster_tree_lines.append(f"{gid} n={len(indices)} folders=[{format_top(summary['top_folders'])}]")

        max_sub_k = min(args.sub_k_max, max(1, len(indices) // args.min_subcluster_size))
        if max_sub_k >= args.sub_k_min and len(indices) >= args.sub_k_min * args.min_subcluster_size:
            sub_rows = scan_k(
                X[indices],
                range(args.sub_k_min, max_sub_k + 1),
                args.min_subcluster_size,
                min(args.sample_size, len(indices)),
                args.batch_size,
                args.n_init,
                label=gid,
            )
            json_dump(sub_rows, out_dir / f"subcluster_metrics_{gid}.json")
            valid_sub_rows = [r for r in sub_rows if r.get("valid_min_size")]
            if valid_sub_rows:
                selected = select_k(sub_rows, args.min_subcluster_size)
                json_dump(selected, out_dir / f"selected_k_{gid}.json")
                sub_k = int(selected["selected_k"])
                sub_km = fit_kmeans(X[indices], sub_k, args.batch_size, args.n_init)
                sub_labels = sub_km.labels_
                macro["sub_k"] = sub_k
                plot_k_scan(sub_rows, sub_k, out_dir / f"subcluster_metrics_{gid}.png", f"{gid} subcluster K scan")
                if sub_k > 1:
                    plot_pca_scatter(
                        X[indices],
                        sub_labels,
                        out_dir / f"subcluster_scatter_{gid}.png",
                        f"{gid} subclusters (PCA)",
                        min(args.plot_sample_size, len(indices)),
                    )
            else:
                selected = {
                    "selected_k": 1,
                    "reason": f"no sub-K candidate satisfied min_subcluster_size={args.min_subcluster_size}; kept macrocluster as one leaf",
                    "top_candidates": sub_rows[:10],
                }
                json_dump(selected, out_dir / f"selected_k_{gid}.json")
                sub_k = 1
                sub_labels = np.zeros(len(indices), dtype=np.int64)
                sub_km = None
        else:
            sub_k = 1
            sub_labels = np.zeros(len(indices), dtype=np.int64)
            sub_km = None

        sub_sheet_rows: List[Dict[str, Any]] = []
        for s in range(sub_k):
            sid = f"{gid}_S{s:02d}"
            local_pos = np.where(sub_labels == s)[0]
            sub_global_indices = indices[local_pos]
            if sub_km is not None:
                center = sub_km.cluster_centers_[s]
                reps_sub_local = representative_indices(X, sub_global_indices, center, args.representatives)
            else:
                reps_sub_local = reps
            sub_summary = summarize_indices(sub_global_indices.tolist(), paths, metadata)
            leaf = {
                "id": sid,
                "parent_id": gid,
                "sub_label": int(s),
                "image_indices": [int(original_indices[i]) for i in sub_global_indices.tolist()],
                "local_indices": sub_global_indices.astype(int).tolist(),
                "representative_indices": [int(original_indices[i]) for i in reps_sub_local],
                "representative_local_indices": reps_sub_local,
                "summary": sub_summary,
            }
            macro["subclusters"].append(leaf)
            sub_sheet_rows.append({
                "label": f"{sid} · n={len(sub_global_indices)}",
                "summary": f"folders: {format_top(sub_summary['top_folders'])}; aspect: {format_top(sub_summary['aspect_buckets'])}",
                "indices": reps_sub_local,
            })
            cluster_tree_lines.append(
                f"  {sid} n={len(sub_global_indices)} folders=[{format_top(sub_summary['top_folders'])}]"
            )
        make_contact_sheet(
            sub_sheet_rows,
            viz_paths,
            out_dir / f"contact_sheet_{gid}.png",
            thumb_size=args.thumb_size,
            max_images_per_row=args.sheet_representatives,
            title=f"{gid}: subclusters",
        )
        hierarchy["clusters"].append(macro)
        cluster_tree_lines.append("")

    make_contact_sheet(
        global_sheet_rows,
        viz_paths,
        out_dir / "contact_sheet_global.png",
        thumb_size=args.thumb_size,
        max_images_per_row=args.sheet_representatives,
        title="Global clusters: nearest images to centroids",
    )
    with open(out_dir / "cluster_tree.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(cluster_tree_lines))
    return hierarchy, macro_rows


def write_macro_summary(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = ["cluster_id", "n", "top_folders", "aspect_buckets", "extensions", "median_fsize_kb"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output)

    print("[1/7] loading data")
    X = load_embeddings(args.embeddings)
    paths = load_lines(args.paths)
    if Path(args.viz_paths).exists():
        viz_paths = load_lines(args.viz_paths)
    else:
        viz_paths = paths[:]
    if not (len(paths) == len(viz_paths) == len(X)):
        raise ValueError(f"Length mismatch: X={len(X)} paths={len(paths)} viz_paths={len(viz_paths)}")
    metadata = load_metadata(args.metadata, len(X))
    X, paths, viz_paths, metadata, original_indices = maybe_limit(X, paths, viz_paths, metadata, args.limit)
    print(f"Loaded X={X.shape}, paths={len(paths)}")

    print("[2/7] scanning global K")
    rows = scan_k(
        X,
        range(args.k_min, args.k_max + 1),
        args.min_cluster_size,
        args.sample_size,
        args.batch_size,
        args.n_init,
        label="global",
    )
    json_dump(rows, out_dir / "k_scan_global.json")
    write_metrics_csv(rows, out_dir / "global_metrics_table.csv")

    print("[3/7] selecting global K")
    selected = select_k(rows, args.min_cluster_size)
    k_global = int(selected["selected_k"])
    json_dump(selected, out_dir / "selected_k_global.json")
    plot_k_scan(rows, k_global, out_dir / "global_elbow_silhouette.png", "Global K scan")
    print(f"Selected K_global={k_global}: {selected['reason']}")

    print("[4/7] fitting final global KMeans")
    global_km = fit_kmeans(X, k_global, args.batch_size, args.n_init)
    global_labels = global_km.labels_.astype(np.int32)
    np.save(out_dir / "global_labels.npy", global_labels)

    print("[5/7] plotting global scatter")
    plot_pca_scatter(X, global_labels, out_dir / "scatter_pca_global.png", "Global clusters (PCA)", args.plot_sample_size)
    maybe_plot_umap(X, global_labels, out_dir / "scatter_umap_global.png", "Global clusters (UMAP)", args.plot_sample_size, args.skip_umap)

    print("[6/7] subclustering each global cluster")
    hierarchy, macro_rows = build_hierarchy(
        X,
        paths,
        viz_paths,
        metadata,
        original_indices,
        global_labels,
        global_km.cluster_centers_,
        args,
        out_dir,
    )
    json_dump(hierarchy, out_dir / "hierarchy.json")
    write_macro_summary(macro_rows, out_dir / "macro_summary.csv")

    print("[7/7] done")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
