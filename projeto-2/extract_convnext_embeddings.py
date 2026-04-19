"""
Extract ConvNeXt-Tiny 768-dim features for ALL images in all_paths.txt.
Saves dataset/X_convnext.npy (22328, 768).
"""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

PATHS_FILE   = "all_paths.txt"
OUTPUT_FILE  = "dataset/X_convnext.npy"
BATCH_SIZE   = 32
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


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
            return torch.zeros(3, 224, 224), idx


def build_model():
    model = models.convnext_tiny(weights="DEFAULT")
    model.classifier = nn.Sequential(
        model.classifier[0],       # LayerNorm2d
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    model.eval()
    return model.to(DEVICE)


def main():
    print(f"Device: {DEVICE}")

    with open(PATHS_FILE, encoding="utf-8") as f:
        paths = [l.strip() for l in f if l.strip()]
    print(f"Paths to embed: {len(paths)}")

    transform = transforms.Compose([
        transforms.Resize(236, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImagePathDataset(paths, transform)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=(DEVICE == "cuda"))

    model = build_model()
    features = np.zeros((len(paths), 768), dtype=np.float32)

    t0 = time.time()
    with torch.no_grad():
        for batch_imgs, batch_idx in loader:
            batch_imgs = batch_imgs.to(DEVICE)
            feats = model(batch_imgs).cpu().float().numpy()
            for i, idx in enumerate(batch_idx.numpy()):
                features[idx] = feats[i]

            done = int(batch_idx[-1]) + 1
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (len(paths) - done) / rate if rate > 0 else 0
            print(f"  [{done:>6}/{len(paths)}]  {rate:.1f} img/s  ETA {eta/60:.1f} min", end="\r", flush=True)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
    np.save(OUTPUT_FILE, features)
    print(f"Saved {features.shape} → {OUTPUT_FILE}  ({features.nbytes/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
