"""
Build labeled dataset from metadata rules + load CLIP embeddings.
Labels:
  image_type: camera_photo | desktop_wallpaper | mobile_screenshot | ai_generated | thumbnail | screenshot_desktop
  has_people: 0 | 1
"""
import json
import struct
import numpy as np
from pathlib import Path
from collections import Counter

METADATA_FILE = "metadata.json"
EMBEDDINGS_FILE = "embeddings_all.bin"
PATHS_FILE = "all_paths.txt"
OUTPUT_DIR = "dataset"

# === Resolution sets ===

CAMERA_RESOLUTIONS = {
    (3024, 4032), (4032, 3024),  # iPhone 12 main
    (2316, 3088), (3088, 2316),  # iPhone 12 ultrawide
    (1600, 1200), (1200, 1600),  # older cameras
    (3664, 2062), (2062, 3664),  # some DSLR
    (4000, 3000), (3000, 4000),  # common camera
    (4000, 2250), (2250, 4000),  # 16:9 camera
}

DESKTOP_WALLPAPER_RESOLUTIONS = {
    (3840, 2160),  # 4K
    (2560, 1440),  # QHD
    (1920, 1080),  # FHD
    (2560, 1080),  # UW-FHD
    (3440, 1440),  # UW-QHD
    (1680, 1050),  # WSXGA+
    (1440, 900),   # WXGA+
    (1366, 768),   # HD
    (1536, 864),   # HD+
    (1600, 900),   # HD+
    (2560, 1600),  # WQXGA
    (1280, 800),   # WXGA
}

MOBILE_SCREENSHOT_RESOLUTIONS = {
    (1170, 2532), (1125, 2436), (1242, 2688),  # iPhone
    (1284, 2778), (1290, 2796), (828, 1792),    # iPhone
    (750, 1334), (1242, 2208),                   # iPhone
    (1080, 2340), (1080, 2400),                  # Android
    (1080, 1920),                                 # Mobile FHD
}

AI_GENERATED_RESOLUTIONS = {
    (512, 512), (1024, 1024), (448, 448), (608, 608),
    (2048, 2048), (768, 768), (256, 256),
    (512, 768), (768, 512),    # SD portrait/landscape
    (512, 682), (682, 512),    # SD variant
    (896, 1152), (1152, 896),  # SDXL
    (832, 1216), (1216, 832),  # SDXL
}

THUMBNAIL_RESOLUTIONS = {
    (180, 320), (320, 180),
    (270, 480), (480, 270),
    (160, 120), (120, 160),
    (320, 240), (240, 320),
    (293, 520), (520, 293),
}

# Folders that strongly indicate people
PEOPLE_FOLDERS = {"people"}


def assign_label(rec: dict) -> tuple[str, int]:
    """Returns (image_type, has_people)."""
    w, h = rec["w"], rec["h"]
    folder = rec["folder"]
    bucket = rec["bucket"]
    res = (w, h)

    has_people = 1 if folder in PEOPLE_FOLDERS else 0

    # --- Tier 1: exact resolution match ---
    if res in THUMBNAIL_RESOLUTIONS:
        return "thumbnail", has_people

    if res in CAMERA_RESOLUTIONS:
        return "camera_photo", has_people

    if res in MOBILE_SCREENSHOT_RESOLUTIONS:
        return "mobile_screenshot", has_people

    if res in AI_GENERATED_RESOLUTIONS:
        return "ai_generated", has_people

    if res in DESKTOP_WALLPAPER_RESOLUTIONS:
        return "desktop_wallpaper", has_people

    # --- Tier 2: folder + aspect heuristics ---
    if folder == "Capturas de tela":
        return "screenshot_desktop", has_people

    if folder == "wallpapers" and bucket in ("wide", "landscape", "ultrawide"):
        return "desktop_wallpaper", has_people

    if folder == "ai_images" and bucket in ("square", "portrait", "landscape"):
        return "ai_generated", has_people

    if folder == "backup" and bucket in ("tall", "supertall"):
        return "thumbnail", has_people

    if folder == "iphone 12":
        # remaining iphone images not matching exact res
        if bucket in ("tall", "supertall"):
            return "mobile_screenshot", has_people
        else:
            return "camera_photo", has_people

    if folder == "people":
        return "camera_photo", has_people

    if folder == "backup":
        return "camera_photo", has_people

    # --- Tier 3: fallback by aspect ---
    if bucket in ("wide", "ultrawide"):
        return "desktop_wallpaper", has_people
    if bucket == "square" and rec["megapixels"] < 2:
        return "ai_generated", has_people

    return "camera_photo", has_people  # safe default


def load_embeddings() -> np.ndarray:
    """Load embeddings.bin: [n:u64][dim:u64][f32 * n * dim]"""
    with open(EMBEDDINGS_FILE, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        dim = struct.unpack("<Q", f.read(8))[0]
        data = np.frombuffer(f.read(n * dim * 4), dtype=np.float32)
    return data.reshape(n, dim)


def main():
    # Load metadata
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Load paths (same order as embeddings)
    with open(PATHS_FILE, "r", encoding="utf-8") as f:
        emb_paths = [line.strip() for line in f if line.strip()]

    # Build path -> metadata index
    meta_by_path = {rec["path"]: rec for rec in metadata}

    # Assign labels
    labels = []
    skipped = 0
    for p in emb_paths:
        rec = meta_by_path.get(p)
        if rec is None:
            skipped += 1
            labels.append(("unknown", 0))
            continue
        labels.append(assign_label(rec))

    image_types = [l[0] for l in labels]
    has_people = [l[1] for l in labels]

    # Stats
    print(f"Total: {len(labels)}, skipped: {skipped}")
    print(f"\n=== image_type distribution ===")
    type_counts = Counter(image_types)
    for t, c in type_counts.most_common():
        print(f"  {t:25s}: {c:5d} ({100*c/len(labels):.1f}%)")

    print(f"\n=== has_people distribution ===")
    people_counts = Counter(has_people)
    for p, c in people_counts.items():
        print(f"  {p}: {c:5d} ({100*c/len(labels):.1f}%)")

    print(f"\n=== image_type x has_people ===")
    cross = Counter(labels)
    for (t, p), c in sorted(cross.items(), key=lambda x: -x[1]):
        print(f"  {t:25s} people={p}: {c:5d}")

    # Save dataset
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Type encoding
    type_names = sorted(type_counts.keys())
    type_to_idx = {t: i for i, t in enumerate(type_names)}
    print(f"\n=== Class encoding ===")
    for t, i in type_to_idx.items():
        print(f"  {i}: {t}")

    y_type = np.array([type_to_idx[t] for t in image_types], dtype=np.int64)
    y_people = np.array(has_people, dtype=np.float32)

    np.save(f"{OUTPUT_DIR}/y_type.npy", y_type)
    np.save(f"{OUTPUT_DIR}/y_people.npy", y_people)

    with open(f"{OUTPUT_DIR}/class_names.json", "w") as f:
        json.dump(type_names, f)

    print(f"\nLabels saved to {OUTPUT_DIR}/")
    print(f"  y_type.npy        : {y_type.shape} ({len(type_names)} classes)")
    print(f"  y_people.npy      : {y_people.shape}")
    print(f"  class_names.json  : {type_names}")

    # Load and save CLIP embeddings if available
    if Path(EMBEDDINGS_FILE).exists():
        print("\nLoading CLIP embeddings...")
        X = load_embeddings()
        print(f"Embeddings shape: {X.shape}")
        np.save(f"{OUTPUT_DIR}/X_embeddings.npy", X)
        print(f"  X_embeddings.npy  : {X.shape}")
    else:
        print(f"\n[SKIP] {EMBEDDINGS_FILE} not found — run CLIP extraction first.")


if __name__ == "__main__":
    main()
