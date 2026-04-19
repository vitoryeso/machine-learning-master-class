"""Extract metadata from ALL images in D:/media and dump analysis."""
import json
import os
from collections import Counter
from pathlib import Path

from PIL import Image

import argparse as _ap
_parser = _ap.ArgumentParser()
_parser.add_argument("--media-root", default="media")
_args, _ = _parser.parse_known_args()
MEDIA_ROOT = _args.media_root
OUTPUT_DIR = "."
ALL_PATHS_FILE = "all_paths.txt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# Common display resolutions (w, h)
DISPLAY_RESOLUTIONS = {
    # Desktop
    (1920, 1080): "FHD",
    (2560, 1440): "QHD",
    (3840, 2160): "4K",
    (1366, 768): "HD",
    (1440, 900): "WXGA+",
    (1536, 864): "HD+",
    (1680, 1050): "WSXGA+",
    (2560, 1080): "UW-FHD",
    (3440, 1440): "UW-QHD",
    (1280, 720): "720p",
    (1280, 800): "WXGA",
    (1600, 900): "HD+",
    (2560, 1600): "WQXGA",
    # Mobile (portrait)
    (1170, 2532): "iPhone12",
    (1125, 2436): "iPhoneX",
    (1242, 2688): "iPhone11ProMax",
    (1080, 2340): "Android-FHD+",
    (1080, 2400): "Android-FHD+",
    (1080, 1920): "Mobile-FHD",
    (750, 1334): "iPhone8",
    (1284, 2778): "iPhone13ProMax",
    (1290, 2796): "iPhone14ProMax",
    (828, 1792): "iPhoneXR",
    (1242, 2208): "iPhone8Plus",
    # Tablet
    (2048, 2732): "iPadPro12",
    (1668, 2388): "iPadPro11",
    (2160, 1620): "iPad10",
}

# Also check landscape versions
DISPLAY_RESOLUTIONS_ALL = {}
for (w, h), name in DISPLAY_RESOLUTIONS.items():
    DISPLAY_RESOLUTIONS_ALL[(w, h)] = name
    DISPLAY_RESOLUTIONS_ALL[(h, w)] = name + "-L"


def aspect_bucket(w: int, h: int) -> str:
    if w == 0 or h == 0:
        return "zero"
    r = w / h
    if r > 2.0:
        return "ultrawide"
    elif r > 1.6:
        return "wide"       # ~16:9, 16:10
    elif r > 1.2:
        return "landscape"  # ~4:3, 3:2
    elif r > 0.83:
        return "square"     # ~1:1
    elif r > 0.625:
        return "portrait"   # ~3:4, 2:3
    elif r > 0.5:
        return "tall"       # ~9:16
    else:
        return "supertall"


def top_folder(path: str) -> str:
    """First folder under D:/media."""
    rel = os.path.relpath(path, MEDIA_ROOT)
    parts = Path(rel).parts
    return parts[0] if parts else "root"


def main():
    # Scan ALL images in MEDIA_ROOT
    print(f"Scanning {MEDIA_ROOT} ...")
    paths = []
    for root, dirs, files in os.walk(MEDIA_ROOT):
        # Skip the machine-learning-master-class folder to avoid recursion
        dirs[:] = [d for d in dirs if d != "machine-learning-master-class"]
        for fname in files:
            if Path(fname).suffix.lower() in IMAGE_EXTS:
                paths.append(os.path.join(root, fname).replace("\\", "/"))

    paths.sort()
    print(f"Total paths: {len(paths)}")

    records = []
    errors = 0
    for i, p in enumerate(paths):
        if i % 1000 == 0:
            print(f"  [{i}/{len(paths)}]")
        try:
            with Image.open(p) as img:
                w, h = img.size
        except Exception:
            errors += 1
            continue

        fsize = os.path.getsize(p)
        ext = Path(p).suffix.lower()
        folder = top_folder(p)
        bucket = aspect_bucket(w, h)
        is_display_res = (w, h) in DISPLAY_RESOLUTIONS_ALL

        records.append({
            "path": p,
            "w": w, "h": h,
            "aspect": round(w / h, 4) if h > 0 else 0,
            "bucket": bucket,
            "fsize_kb": round(fsize / 1024, 1),
            "ext": ext,
            "folder": folder,
            "is_display_res": is_display_res,
            "display_name": DISPLAY_RESOLUTIONS_ALL.get((w, h), ""),
            "megapixels": round(w * h / 1e6, 2),
        })

    print(f"Loaded {len(records)} images, {errors} errors")

    # --- Analysis ---
    print("\n=== ASPECT RATIO BUCKETS ===")
    bucket_counts = Counter(r["bucket"] for r in records)
    for b, c in bucket_counts.most_common():
        print(f"  {b:12s}: {c:5d} ({100*c/len(records):.1f}%)")

    print("\n=== TOP FOLDERS ===")
    folder_counts = Counter(r["folder"] for r in records)
    for f, c in folder_counts.most_common(15):
        print(f"  {f:30s}: {c:5d} ({100*c/len(records):.1f}%)")

    print("\n=== DISPLAY RESOLUTIONS ===")
    display_imgs = [r for r in records if r["is_display_res"]]
    print(f"  Match exact display res: {len(display_imgs)} ({100*len(display_imgs)/len(records):.1f}%)")
    disp_counts = Counter(r["display_name"] for r in display_imgs)
    for d, c in disp_counts.most_common(15):
        print(f"    {d:20s}: {c:4d}")

    print("\n=== FOLDER × BUCKET CROSSTAB ===")
    top_folders = [f for f, _ in folder_counts.most_common(8)]
    all_buckets = ["ultrawide", "wide", "landscape", "square", "portrait", "tall", "supertall"]
    header = f"{'folder':20s} " + " ".join(f"{b:>10s}" for b in all_buckets) + f" {'TOTAL':>8s}"
    print(f"  {header}")
    for fld in top_folders:
        fld_recs = [r for r in records if r["folder"] == fld]
        bc = Counter(r["bucket"] for r in fld_recs)
        row = f"  {fld:20s} " + " ".join(f"{bc.get(b,0):10d}" for b in all_buckets) + f" {len(fld_recs):8d}"
        print(row)

    print("\n=== RESOLUTION CLUSTERS (common WxH) ===")
    res_counts = Counter((r["w"], r["h"]) for r in records)
    for (w, h), c in res_counts.most_common(20):
        tag = DISPLAY_RESOLUTIONS_ALL.get((w, h), "")
        print(f"  {w:5d}x{h:<5d}: {c:4d}  {tag}")

    # Save full metadata for next step
    out_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"\nSaved metadata to {out_path}")

    # Save all valid paths (same order as records) for embedding extraction
    valid_paths = [r["path"] for r in records]
    with open(ALL_PATHS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(valid_paths))
    print(f"Saved {len(valid_paths)} paths to {ALL_PATHS_FILE}")


if __name__ == "__main__":
    main()
