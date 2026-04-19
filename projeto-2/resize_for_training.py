"""
Resize all training images to 256x256 JPEG for cloud storage.
Network input is 224x224 — 256px gives margin for RandomResizedCrop.

Usage (run from projeto-2/):
  python resize_for_training.py --media-root D:/media --out-dir media_256
  python resize_for_training.py --media-root /mnt/media  --out-dir media_256

Output:
  media_256/<original_relative_path>.jpg   (all converted to JPEG)
  all_paths_256.txt                         (updated paths pointing to media_256/)
"""
import argparse
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image, UnidentifiedImageError

TARGET_SIZE = 256   # resize shorter side to 256, then centercrop to 256x256
JPEG_QUALITY = 90
WORKERS = 8


def resize_image(src_path: str, dst_path: str) -> bool:
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            # Resize shortest side to TARGET_SIZE
            w, h = img.size
            if w < h:
                new_w, new_h = TARGET_SIZE, int(h * TARGET_SIZE / w)
            else:
                new_w, new_h = int(w * TARGET_SIZE / h), TARGET_SIZE
            img = img.resize((new_w, new_h), Image.LANCZOS)
            # Center crop to TARGET_SIZE x TARGET_SIZE
            left = (new_w - TARGET_SIZE) // 2
            top  = (new_h - TARGET_SIZE) // 2
            img = img.crop((left, top, left + TARGET_SIZE, top + TARGET_SIZE))
            Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
            img.save(dst_path, "JPEG", quality=JPEG_QUALITY, optimize=True)
        return True
    except (UnidentifiedImageError, OSError, Exception):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--media-root", required=True, help="Root of original images")
    parser.add_argument("--out-dir", default="media_256", help="Output directory")
    parser.add_argument("--paths-file", default="all_paths.txt")
    args = parser.parse_args()

    media_root = Path(args.media_root).resolve()
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.paths_file, encoding="utf-8") as f:
        src_paths = [l.strip() for l in f if l.strip()]

    print(f"Images to resize: {len(src_paths)}")
    print(f"Source root:      {media_root}")
    print(f"Output dir:       {out_dir}  ({TARGET_SIZE}x{TARGET_SIZE} JPEG q={JPEG_QUALITY})")

    dst_paths = []
    tasks = []
    for src in src_paths:
        rel = Path(src).relative_to(media_root)
        dst = str(out_dir / rel.with_suffix(".jpg"))
        dst_paths.append(dst)
        tasks.append((src, dst))

    t0 = time.time()
    done = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(resize_image, s, d): i for i, (s, d) in enumerate(tasks)}
        for fut in as_completed(futs):
            ok = fut.result()
            done += 1
            if not ok:
                errors += 1
            if done % 500 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(tasks) - done) / rate
                print(f"  [{done:>6}/{len(tasks)}]  {rate:.0f} img/s  ETA {eta/60:.1f}min", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min — {done - errors} ok, {errors} errors")

    # Estimate output size
    total_bytes = sum(os.path.getsize(d) for d in dst_paths if os.path.exists(d))
    print(f"Output size: {total_bytes/1e9:.2f} GB")

    # Save updated paths file
    with open("all_paths_256.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(dst_paths))
    print(f"Updated paths saved to all_paths_256.txt")


if __name__ == "__main__":
    main()
