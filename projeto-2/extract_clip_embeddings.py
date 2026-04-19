"""
Extract CLIP ViT-B/32 embeddings for all images in all_paths.txt.
Saves embeddings_all.bin in the same format as the Rust binary:
  [n: u64 LE][dim: u64 LE][f32 * n * dim, row-major]
"""
import struct
import sys
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image, UnidentifiedImageError

PATHS_FILE  = "all_paths.txt"
OUTPUT_BIN  = "embeddings_all.bin"
BATCH_SIZE  = 64
MODEL_NAME  = "ViT-B-32"
PRETRAINED  = "openai"   # same weights as CLIP paper


def load_paths(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def save_bin(embeddings: np.ndarray, out_path: str):
    n, dim = embeddings.shape
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", n))
        f.write(struct.pack("<Q", dim))
        f.write(embeddings.astype(np.float32).tobytes())
    print(f"Saved {n}x{dim} embeddings to {out_path}  ({Path(out_path).stat().st_size / 1e6:.1f} MB)")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading {MODEL_NAME} ({PRETRAINED})...")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device).eval()

    paths = load_paths(PATHS_FILE)
    print(f"Paths to embed: {len(paths)}")

    all_embeddings = []
    errors = []
    t0 = time.time()

    for batch_start in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[batch_start:batch_start + BATCH_SIZE]
        images = []
        valid_idx = []

        for i, p in enumerate(batch_paths):
            try:
                img = Image.open(p).convert("RGB")
                images.append(preprocess(img))
                valid_idx.append(i)
            except (UnidentifiedImageError, OSError, Exception):
                errors.append(p)

        if not images:
            # fill with zeros for missing images in this batch
            for _ in batch_paths:
                all_embeddings.append(np.zeros(512, dtype=np.float32))
            continue

        batch_tensor = torch.stack(images).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch_tensor)
            # L2 normalize (same as fastembed-rs CLIP)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats = feats.cpu().float().numpy()

        # interleave valid embeddings with zeros for error images
        emb_iter = iter(feats)
        valid_set = set(valid_idx)
        for i in range(len(batch_paths)):
            if i in valid_set:
                all_embeddings.append(next(emb_iter))
            else:
                all_embeddings.append(np.zeros(512, dtype=np.float32))

        processed = batch_start + len(batch_paths)
        elapsed = time.time() - t0
        rate = processed / elapsed
        eta = (len(paths) - processed) / rate if rate > 0 else 0
        print(f"  [{processed:>6}/{len(paths)}]  {rate:.1f} img/s  ETA {eta/60:.1f} min", end="\r", flush=True)

    print()
    print(f"Done in {(time.time()-t0)/60:.1f} min — {len(errors)} errors")

    X = np.stack(all_embeddings)
    print(f"Embeddings shape: {X.shape}  dtype: {X.dtype}")
    save_bin(X, OUTPUT_BIN)

    if errors:
        err_path = OUTPUT_BIN.replace(".bin", "_errors.txt")
        with open(err_path, "w") as f:
            f.write("\n".join(errors))
        print(f"Error paths saved to {err_path}")


if __name__ == "__main__":
    main()
