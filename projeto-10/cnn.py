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
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
    plt.rcParams.update({           # fontes GRANDES p/ legibilidade no slide
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })
except ImportError:                     # host sem matplotlib (ex.: mcculloch)
    plt = None
    HAVE_MPL = False
try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:                     # host sem Pillow — usa cache .npz
    Image = None
    HAVE_PIL = False
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
N_SEEDS = 3  # rodadas para media +/- desvio; -n muda em runtime
SEEDS = [SEED + i for i in range(N_SEEDS)]


# =============================================================================
# Dados: screenshot vs foto, imagens raw de D:/media
# =============================================================================
def load_dataset(seed=SEED):
    """Imagens reais baixadas do Google Drive (pasta = rótulo).
    data/screenshot/* -> classe 1 ; data/foto/* (fotos de People) -> classe 0."""
    base = os.path.join(HERE, "data")
    cache = os.path.join(base, "cache_{}.npz".format(SIZE))
    if os.path.exists(cache):                        # rapido + portavel (sem PIL)
        d = np.load(cache)
        X, y = d["X"], d["y"]
    else:
        if not HAVE_PIL:
            raise SystemExit("Sem Pillow e sem cache {}: gere o cache numa maquina com PIL.".format(cache))
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
        X = np.array(X).transpose(0, 3, 1, 2)        # (N,3,H,W)
        y = np.array(y)
        np.savez(cache, X=X, y=y)                     # salva p/ reuso portavel
    # embaralha (estavam agrupados por classe)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    print("Carregadas {} imagens ({} screenshot / {} foto)".format(len(y), int(y.sum()), int((y == 0).sum())))
    return X, y


def split(X, y, frac=0.8, seed=SEED):
    rng = np.random.default_rng(seed)
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


# =============================================================================
# Multi-seed — robustez estatistica (media +/- desvio sobre N_SEEDS rodadas)
# =============================================================================
def run_multiseed():
    """Roda as 3 modalidades (2D-RGB, 2D-gray, 1D) sobre N_SEEDS realizacoes;
    cada seed = novo split + nova init dos pesos (torch). Resultado OFICIAL.
    Escreve report_multiseed.txt + metrics_multiseed.png. seed=42 fica so p/ os
    feature maps (nao-mediaveis) no restante do main()."""
    X, y = load_dataset(SEED)               # carrega as imagens 1x
    mods = ["2D-RGB", "2D-gray", "1D"]
    agg = {m: {"f1": [], "acc": []} for m in mods}
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        Xtr, ytr, Xte, yte = split(X, y, seed=seed)
        r = train_eval(CNN2D(3, 16, 3), Xtr, ytr, Xte, yte)
        agg["2D-RGB"]["f1"].append(r["f1"]); agg["2D-RGB"]["acc"].append(r["acc"])
        Xtr_g = Xtr.mean(1, keepdims=True); Xte_g = Xte.mean(1, keepdims=True)
        r = train_eval(CNN2D(1, 16, 3), Xtr_g, ytr, Xte_g, yte)
        agg["2D-gray"]["f1"].append(r["f1"]); agg["2D-gray"]["acc"].append(r["acc"])
        sig_tr = Xtr.mean(1).mean(2)[:, None, :]; sig_te = Xte.mean(1).mean(2)[:, None, :]
        r = train_eval(CNN1D(16, 5), sig_tr, ytr, sig_te, yte)
        agg["1D"]["f1"].append(r["f1"]); agg["1D"]["acc"].append(r["acc"])

    def ms(v):
        return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
    stats = {m: {k: ms(agg[m][k]) for k in ("f1", "acc")} for m in mods}

    lines = ["Multi-seed: media +/- desvio (ddof=1) sobre {} seeds {}".format(N_SEEDS, SEEDS),
             "(cada seed = novo split + nova init dos pesos; {} imagens {}x{})".format(len(y), SIZE, SIZE), ""]
    hdr = "{:<10} {:>18} {:>18}".format("Modalidade", "F1", "Acc")
    lines += [hdr, "-" * len(hdr)]
    for m in mods:
        f1m, f1s = stats[m]["f1"]; am, ast = stats[m]["acc"]
        lines.append("{:<10} {:>11.4f} +/-{:>4.4f} {:>11.4f} +/-{:>4.4f}".format(m, f1m, f1s, am, ast))
    lines.append("-" * len(hdr))

    # CSV sempre (p/ plotar off-host quando faltar matplotlib, ex.: mcculloch)
    csv_path = os.path.join(OUT, "metrics_multiseed.csv")
    with open(csv_path, "w") as f:
        f.write("modalidade,f1_mean,f1_std,acc_mean,acc_std\n")
        for m in mods:
            f.write("{},{:.6f},{:.6f},{:.6f},{:.6f}\n".format(
                m, stats[m]["f1"][0], stats[m]["f1"][1], stats[m]["acc"][0], stats[m]["acc"][1]))
    ms_path = os.path.join(OUT, "metrics_multiseed.png")
    if HAVE_MPL:
        colors = {"2D-RGB": "#4e79a7", "2D-gray": "#76b7b2", "1D": "#f28e2b"}
        fig, ax = plt.subplots(figsize=(7, 4.5))
        xs = np.arange(len(mods))
        means = [stats[m]["f1"][0] for m in mods]; stds = [stats[m]["f1"][1] for m in mods]
        ax.bar(xs, means, yerr=stds, capsize=8, color=[colors[m] for m in mods],
               edgecolor="k", linewidth=0.7, error_kw=dict(ecolor="#333", lw=1.3))
        lo = min(m - s for m, s in zip(means, stds)); hi = max(m + s for m, s in zip(means, stds))
        ax.set_ylim(max(0.0, lo - 0.05), min(1.0, hi + 0.05))     # piso adaptativo
        ax.set_xticks(xs); ax.set_xticklabels(mods)
        ax.set_ylabel("F1 (teste)")
        ax.set_title("Modalidades -- media +/- desvio ({} seeds)".format(N_SEEDS))
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(ms_path, dpi=120, bbox_inches="tight"); plt.close()
    else:
        ms_path = csv_path      # sem matplotlib: CSV vira o entregavel (plote off-host)
    with open(os.path.join(OUT, "report_multiseed.txt"), "w") as f:
        f.write("\n".join(lines))
    return stats, lines, ms_path


def main():
    print("=== Projeto 10 — CNN 1D/2D (device={}) ===\n".format(DEV))

    # -- 0. Multi-seed: resultado oficial (media +/- desvio) ------------------
    _, ms_report, ms_path = run_multiseed()
    print("\n".join(ms_report)); print("Saved:", ms_path); print()
    if not HAVE_MPL:                     # host de compute (mcculloch): so o oficial
        print("(sem matplotlib: multi-seed = report_multiseed.txt + CSV; ilustracao single-seed pulada)")
        return
    print("--- Ilustracao (seed=42, 1 realizacao): tabela single-seed + feature maps ---\n")

    X, y = load_dataset(SEED)
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

    # -- plot resumo: modalidades + ablação (empilhado, fonte grande) -------
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 9.5))
    mods = ["2D-RGB", "2D-gray", "1D-perfil"]
    axes[0].bar(mods, [results[m]["f1"] for m in mods], color=["#4e79a7", "#76b7b2", "#f28e2b"])
    axes[0].set_ylabel("F1 (teste)"); axes[0].set_ylim(0, 1)
    axes[0].set_title("Modalidade: 2D RGB vs 2D cinza vs 1D")
    for i, m in enumerate(mods):
        axes[0].text(i, results[m]["f1"] + 0.03, "{:.3f}".format(results[m]["f1"]),
                      ha="center", fontsize=16, fontweight="bold")
    axes[1].bar(list(ablation.keys()), list(ablation.values()), color="#4e79a7")
    axes[1].set_ylabel("F1 (teste)"); axes[1].set_ylim(0, 1)
    axes[1].set_title("Ablação: nº de filtros / kernel (2D RGB)")
    axes[1].tick_params(axis="x", rotation=0)
    plt.tight_layout()
    p_sum = os.path.join(OUT, "modalities_ablation.png")
    plt.savefig(p_sum, dpi=160, bbox_inches="tight"); plt.close()
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
    import argparse
    _ap = argparse.ArgumentParser(description="CNN 1D/2D — multi-seed")
    _ap.add_argument("-n", "--n-seeds", type=int, default=N_SEEDS,
                     help="numero de seeds para media +/- desvio (default: %(default)s)")
    _a = _ap.parse_args()
    N_SEEDS = _a.n_seeds
    SEEDS = [SEED + i for i in range(N_SEEDS)]
    main()
