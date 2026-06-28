#!/usr/bin/env python3
"""Projeto 10 — CNN para dados 1D e 2D (PyTorch).

Enunciado: construir CNNs simples p/ classificação binária demonstrando como as
arquiteturas se adaptam a dados 1D e 2D, com/sem múltiplos canais; analisar o
efeito de filtros/pooling; e visualizar os feature maps das camadas convolucionais.

Problema REAL (fecha o ciclo 1/2): classificação binária das imagens pessoais —
'screenshot vs foto de câmera' (tarefa visualmente saliente → feature maps
interpretáveis). Imagens em data/ (pasta = rótulo), baixadas do Google Drive; treino no RTX 3060 do xmain.

  - 2D: CNN sobre a imagem (RGB = 3 canais vs grayscale = 1 canal)
  - 1D: CNN sobre o perfil de intensidade por linha (sinal de comprimento H)
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = 64


# =============================================================================
# Dados: screenshot vs foto, imagens raw de D:/media
# =============================================================================
def load_dataset():
    """Imagens reais baixadas do Google Drive (pasta = rótulo).
    data/screenshot/* -> classe 1 ; data/foto/* (fotos de People) -> classe 0."""
    base = os.path.join(HERE, "data")
    X, y = [], []
    for name, lab in [("screenshot", 1), ("foto", 0)]:
        d = os.path.join(base, name)
        for root, _, files in os.walk(d):
            for f in files:
                if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                try:
                    im = Image.open(os.path.join(root, f)).convert("RGB").resize((SIZE, SIZE))
                    X.append(np.asarray(im, dtype=np.float32) / 255.0)
                    y.append(lab)
                except Exception:
                    continue
    X = np.array(X).transpose(0, 3, 1, 2)            # (N,3,H,W)
    y = np.array(y)
    # embaralha (estavam agrupados por classe)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    print("Carregadas {} imagens ({} screenshot / {} foto)".format(len(y), int(y.sum()), int((y == 0).sum())))
    return X, y


def split(X, y, frac=0.8):
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y))
    cut = int(frac * len(y))
    tr, te = perm[:cut], perm[cut:]
    return X[tr], y[tr], X[te], y[te]


# =============================================================================
# Modelos
# =============================================================================
class CNN2D(nn.Module):
    def __init__(self, in_ch=3, nf=16, k=3):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv2d(in_ch, nf, k, padding=p)
        self.conv2 = nn.Conv2d(nf, nf * 2, k, padding=p)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(nf * 2 * (SIZE // 4) * (SIZE // 4), 1)
        self.act = nn.ReLU()

    def features(self, x):
        return self.act(self.conv1(x))              # feature maps da 1ª camada

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        return self.fc(x.flatten(1)).squeeze(1)


class CNN1D(nn.Module):
    def __init__(self, nf=16, k=5):
        super().__init__()
        self.conv1 = nn.Conv1d(1, nf, k, padding=k // 2)
        self.conv2 = nn.Conv1d(nf, nf * 2, k, padding=k // 2)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(nf * 2 * (SIZE // 4), 1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        return self.fc(x.flatten(1)).squeeze(1)


def train_eval(model, Xtr, ytr, Xte, yte, epochs=15, lr=1e-3, bs=64):
    model = model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    n = len(Xtr_t)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, bs):
            bi = perm[s:s + bs]
            xb = Xtr_t[bi].to(DEV); yb = ytr_t[bi].to(DEV)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
    # eval
    model.eval()
    with torch.no_grad():
        Xte_t = torch.tensor(Xte, dtype=torch.float32).to(DEV)
        logits = model(Xte_t).cpu().numpy()
    pred = (logits >= 0).astype(int)
    tp = int(((pred == 1) & (yte == 1)).sum()); fp = int(((pred == 1) & (yte == 0)).sum())
    fn = int(((pred == 0) & (yte == 1)).sum())
    acc = float((pred == yte).mean())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"acc": acc, "f1": f1, "prec": prec, "rec": rec}


def main():
    print("=== Projeto 10 — CNN 1D/2D (device={}) ===\n".format(DEV))
    X, y = load_dataset()
    Xtr, ytr, Xte, yte = split(X, y)
    print("Train {}  Test {}\n".format(len(ytr), len(yte)))

    results = {}

    # -- 2D RGB (3 canais) --------------------------------------------------
    print("[2D RGB] CNN[nf=16,k=3] sobre imagem RGB")
    results["2D-RGB"] = train_eval(CNN2D(3, 16, 3), Xtr, ytr, Xte, yte)
    print("  ", results["2D-RGB"])

    # -- 2D grayscale (1 canal) --------------------------------------------
    Xtr_g = Xtr.mean(1, keepdims=True); Xte_g = Xte.mean(1, keepdims=True)
    print("[2D GRAY] mesma CNN, 1 canal (escala de cinza)")
    results["2D-gray"] = train_eval(CNN2D(1, 16, 3), Xtr_g, ytr, Xte_g, yte)
    print("  ", results["2D-gray"])

    # -- 1D: perfil de intensidade por linha (sinal length=H) --------------
    sig_tr = Xtr.mean(1).mean(2)[:, None, :]        # (N,1,H): média sobre canais e colunas
    sig_te = Xte.mean(1).mean(2)[:, None, :]
    print("[1D] CNN1D sobre o perfil de intensidade por linha (sinal length={})".format(SIZE))
    results["1D-perfil"] = train_eval(CNN1D(16, 5), sig_tr, ytr, sig_te, yte)
    print("  ", results["1D-perfil"])
    print()

    # -- ablação: nº de filtros e tamanho do kernel (2D RGB) ---------------
    print("[ablação] nº de filtros e kernel (2D RGB)")
    ablation = {}
    for nf in [4, 16, 32]:
        ablation["nf={}".format(nf)] = train_eval(CNN2D(3, nf, 3), Xtr, ytr, Xte, yte)["f1"]
    for k in [3, 5, 7]:
        ablation["k={}".format(k)] = train_eval(CNN2D(3, 16, k), Xtr, ytr, Xte, yte)["f1"]
    for key, v in ablation.items():
        print("  {:<8} f1={:.3f}".format(key, v))
    print()

    # -- feature maps do conv1 (modelo RGB re-treinado p/ extrair) ---------
    print("[feature maps] conv1 sobre 1 screenshot e 1 foto")
    fmodel = CNN2D(3, 16, 3).to(DEV)
    train_eval(fmodel, Xtr, ytr, Xte, yte, epochs=15)   # treina p/ filtros úteis
    ishot = int(np.where(yte == 1)[0][0]); iphoto = int(np.where(yte == 0)[0][0])
    fig, axes = plt.subplots(2, 7, figsize=(13, 4.2))
    for row, (ii, name) in enumerate([(ishot, "screenshot"), (iphoto, "foto")]):
        img = Xte[ii]
        axes[row][0].imshow(img.transpose(1, 2, 0)); axes[row][0].set_title(name, fontsize=9)
        axes[row][0].axis("off")
        with torch.no_grad():
            fm = fmodel.features(torch.tensor(img[None], dtype=torch.float32).to(DEV))[0].cpu().numpy()
        for j in range(6):
            axes[row][j + 1].imshow(fm[j], cmap="viridis")
            axes[row][j + 1].set_title("filtro {}".format(j), fontsize=8)
            axes[row][j + 1].axis("off")
    fig.suptitle("Feature maps do conv1 (screenshot vs foto)", fontsize=12)
    plt.tight_layout()
    p_fm = os.path.join(OUT, "feature_maps.png")
    plt.savefig(p_fm, dpi=120, bbox_inches="tight"); plt.close()
    print("  Saved:", p_fm)

    # -- plot resumo: modalidades + ablação --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    mods = ["2D-RGB", "2D-gray", "1D-perfil"]
    axes[0].bar(mods, [results[m]["f1"] for m in mods], color=["#4e79a7", "#76b7b2", "#f28e2b"])
    axes[0].set_ylabel("F1 (teste)"); axes[0].set_ylim(0, 1)
    axes[0].set_title("Modalidade: 2D RGB vs 2D cinza vs 1D")
    for i, m in enumerate(mods):
        axes[0].text(i, results[m]["f1"] + 0.02, "{:.3f}".format(results[m]["f1"]), ha="center", fontsize=9)
    axes[1].bar(list(ablation.keys()), list(ablation.values()), color="#4e79a7")
    axes[1].set_ylabel("F1 (teste)"); axes[1].set_ylim(0, 1)
    axes[1].set_title("Ablação: nº de filtros / kernel (2D RGB)")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    p_sum = os.path.join(OUT, "modalities_ablation.png")
    plt.savefig(p_sum, dpi=120, bbox_inches="tight"); plt.close()
    print("  Saved:", p_sum)

    # -- report -------------------------------------------------------------
    lines = ["=" * 60, "PROJETO 10 — CNN 1D/2D", "=" * 60, "",
             "Tarefa: screenshot vs foto (imagens em data/, {}x{}).".format(SIZE, SIZE),
             "Device: {} | imagens: {} (treino {} / teste {})".format(DEV, len(y), len(ytr), len(yte)), "",
             "Modalidades (F1 teste):"]
    for m in mods:
        r = results[m]
        lines.append("  {:<10} acc={:.4f} f1={:.4f} prec={:.4f} rec={:.4f}".format(
            m, r["acc"], r["f1"], r["prec"], r["rec"]))
    lines += ["", "Ablação (F1 teste):"]
    for key, v in ablation.items():
        lines.append("  {:<8} {:.4f}".format(key, v))
    with open(os.path.join(OUT, "report.txt"), "w") as f:
        f.write("\n".join(lines))
    print("\nDone.")


if __name__ == "__main__":
    main()
