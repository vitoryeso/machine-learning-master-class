#!/usr/bin/env python3
"""Projeto 9 — ANN/MLP construída manualmente (numpy), fronteiras não-lineares.

Enunciado: MLP do zero p/ classificação binária; comparar MSE vs cross-entropy;
ablar nº de camadas/neurônios/ativações; mostrar no plano 2D como os neurônios
da camada oculta particionam o espaço.

Problema REAL (compartilhado com os projetos 7/8): classificação binária sobre a
coleção pessoal de imagens (features CLIP), 3 tarefas. Desempenho é medido em
FULL-DIM (real); o plano 2D (PCA) é usado só para ILUSTRAR as partições.
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shared_problem as sp

SEED = 42
N_SEEDS = 3  # rodadas p/ media +/- desvio; suba p/ 10/30 p/ std mais firme
SEEDS = [SEED + i for i in range(N_SEEDS)]  # [42, 43, 44]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)


# =============================================================================
# MLP do zero (numpy) — forward + backprop manual, perda MSE ou BCE
# =============================================================================
class MLP:
    def __init__(self, sizes, activation="relu", loss="ce", seed=SEED):
        """sizes: [D, h1, ..., 1] (saída 1 = binário, sempre sigmoid).
        loss: 'ce' (binary cross-entropy) ou 'mse'."""
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            scale = np.sqrt(2.0 / sizes[i])          # init He
            self.W.append(rng.normal(0, scale, (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))
        self.activation, self.loss, self.seed = activation, loss, seed

    def _act(self, z):
        if self.activation == "relu":
            return np.maximum(0, z)
        if self.activation == "tanh":
            return np.tanh(z)
        return 1 / (1 + np.exp(-z))                  # sigmoid

    def _act_grad(self, z, a):
        if self.activation == "relu":
            return (z > 0).astype(z.dtype)
        if self.activation == "tanh":
            return 1 - a ** 2
        return a * (1 - a)

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

    def _forward(self, X):
        zs, acts = [], [X]
        a = X
        for l in range(len(self.W)):
            z = a @ self.W[l] + self.b[l]
            zs.append(z)
            a = self._sigmoid(z) if l == len(self.W) - 1 else self._act(z)
            acts.append(a)
        return zs, acts

    def predict_proba(self, X):
        return self._forward(X)[1][-1].ravel()

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def fit(self, X, y, epochs=60, lr=0.05, batch=256, verbose=False):
        y = y.reshape(-1, 1).astype(float)
        n = len(X)
        rng = np.random.default_rng(self.seed)  # shuffle segue a seed do modelo
        hist = []
        for ep in range(epochs):
            perm = rng.permutation(n)
            for s in range(0, n, batch):
                bi = perm[s:s + batch]
                xb, yb = X[bi], y[bi]
                zs, acts = self._forward(xb)
                p = acts[-1]
                m = len(xb)
                # delta na saída (sigmoid): BCE -> (p-y); MSE -> 2(p-y)p(1-p)
                if self.loss == "ce":
                    delta = (p - yb) / m
                else:
                    delta = 2 * (p - yb) * p * (1 - p) / m
                # backprop
                for l in range(len(self.W) - 1, -1, -1):
                    dW = acts[l].T @ delta
                    db = delta.sum(axis=0)
                    if l > 0:
                        delta = (delta @ self.W[l].T) * self._act_grad(zs[l - 1], acts[l])
                    self.W[l] -= lr * dW
                    self.b[l] -= lr * db
            # loss da época (no conjunto todo)
            p = self.predict_proba(X).reshape(-1, 1)
            eps = 1e-9
            if self.loss == "ce":
                L = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            else:
                L = np.mean((p - y) ** 2)
            hist.append(float(L))
            if verbose and ep % 10 == 0:
                print("  ep{:>3} loss={:.4f}".format(ep, L))
        return hist


# =============================================================================
# Multi-seed — robustez estatistica (media +/- desvio sobre N_SEEDS rodadas)
# =============================================================================
# Os dados reais (features CLIP) sao FIXOS. Por seed varia (a) o split
# estratificado treino/teste e (b) a inicializacao He dos pesos do MLP + a ordem
# dos minibatches. A inicializacao aleatoria e a principal fonte de variancia.
EPOCHS, LR = 60, 0.1
ARCH_SIZES = [0, 2, 8, 32, 128]   # 0 = sem camada oculta (= regressao logistica)
_FEAT_CACHE = {}


def _prepare_seed(task, seed, features="clip", test_size=0.2):
    """Split full-dim padronizado com random_state=seed (a seed varia a particao).
    Espelha sp.prepare mas com seed parametrizavel — sp.prepare fixa SEED=42 e
    esta fora de projeto-9/, entao refazemos o prep aqui com as funcoes publicas."""
    mask, y, _, _ = sp.get_task(task)
    key = (task, features)
    if key not in _FEAT_CACHE:                      # features nao mudam por seed
        _FEAT_CACHE[key] = sp.load_features(features)[mask]
    X = _FEAT_CACHE[key]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr)
    return {"X_train": sc.transform(Xtr), "X_test": sc.transform(Xte),
            "y_train": ytr, "y_test": yte}


def _ms(v):
    """(media, desvio amostral ddof=1). Com 1 seed o desvio cai p/ 0."""
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)


def _adaptive_ylim(ax, means, stds, pad_lo=0.05, pad_hi=0.03):
    """Piso do eixo y ADAPTATIVO: parte um pouco abaixo da menor barra-erro, em
    vez de fixar 0 — asssim as diferencas minusculas (e seus stds) ficam legiveis."""
    lo = min(m - s for m, s in zip(means, stds))
    hi = max(m + s for m, s in zip(means, stds))
    ax.set_ylim(max(0.0, lo - pad_lo), min(1.0, hi + pad_hi))


def run_multiseed(task_cmp="macro_vs_rest"):
    """Roda a ablacao sobre N_SEEDS realizacoes e agrega media +/- desvio.
    Coleta, por tarefa e por perda (CE, MSE): accuracy e F1 do MLP[D-64-1] relu.
    Coleta tambem a ablacao de arquitetura (0..128 ocultos, CE) em task_cmp.
    Escreve report_multiseed.txt + metrics_multiseed.png. Aditivo: nao mexe na
    analise single-seed (seed=42) do main(). Retorna (stats, lines, png_path)."""
    losses = ("ce", "mse")
    agg = {t: {ls: {"accuracy": [], "f1": []} for ls in losses} for t in sp.TASKS}
    agg_arch = {h: [] for h in ARCH_SIZES}

    for seed in SEEDS:
        for task in sp.TASKS:
            d = _prepare_seed(task, seed)
            D = d["X_train"].shape[1]
            for ls in losses:
                net = MLP([D, 64, 1], activation="relu", loss=ls, seed=seed)
                net.fit(d["X_train"], d["y_train"], epochs=EPOCHS, lr=LR)
                m = sp.evaluate(d["y_test"], net.predict(d["X_test"]))
                agg[task][ls]["accuracy"].append(m["accuracy"])
                agg[task][ls]["f1"].append(m["f1"])
        # ablacao de arquitetura (CE) na tarefa de comparacao
        d = _prepare_seed(task_cmp, seed)
        D = d["X_train"].shape[1]
        for h in ARCH_SIZES:
            sizes = [D, 1] if h == 0 else [D, h, 1]
            net = MLP(sizes, activation="relu", loss="ce", seed=seed)
            net.fit(d["X_train"], d["y_train"], epochs=EPOCHS, lr=LR)
            agg_arch[h].append(sp.evaluate(d["y_test"], net.predict(d["X_test"]))["f1"])

    stats = {t: {ls: {k: _ms(v) for k, v in agg[t][ls].items()} for ls in losses}
             for t in sp.TASKS}
    stats_arch = {h: _ms(agg_arch[h]) for h in ARCH_SIZES}

    # -- report textual ------------------------------------------------------
    lines = ["Multi-seed: media +/- desvio (ddof=1) sobre {} seeds {}".format(N_SEEDS, SEEDS),
             "(dados CLIP fixos; a seed varia split + init He dos pesos + ordem dos batches)",
             "MLP[D-64-1], relu, {} epocas, lr={}".format(EPOCHS, LR), ""]
    hdr = "{:<22} {:>20} {:>20}".format("Tarefa / metrica", "CE", "MSE")
    lines += [hdr, "-" * len(hdr)]
    for t in sp.TASKS:
        for k in ("accuracy", "f1"):
            cm, cs = stats[t]["ce"][k]
            mm, ms_ = stats[t]["mse"][k]
            lines.append("{:<22} {:>12.4f} +/-{:>5.4f} {:>12.4f} +/-{:>5.4f}".format(
                "{}/{}".format(t[:16], k[:3]), cm, cs, mm, ms_))
    lines += ["-" * len(hdr), "",
              "Ablacao de arquitetura em '{}' (CE, F1 teste):".format(task_cmp),
              "{:<14} {:>20}".format("ocultos", "F1")]
    for h in ARCH_SIZES:
        fm, fs = stats_arch[h]
        lines.append("{:<14} {:>12.4f} +/-{:>5.4f}".format(
            "0 (logistica)" if h == 0 else str(h), fm, fs))
    lines.append("")

    # -- veredito automatico: as diferencas minusculas caem dentro do std? ----
    def _overlap(a, b):  # intervalos [mean-std, mean+std] se sobrepoem?
        return not (a[0] + a[1] < b[0] - b[1] or b[0] + b[1] < a[0] - a[1])
    ce_vs_mse = all(_overlap(stats[t]["ce"]["f1"], stats[t]["mse"]["f1"]) for t in sp.TASKS)
    arch_ovl = _overlap(stats_arch[0], stats_arch[128])
    lines += ["Veredito (F1, faixas mean+/-std):",
              "  MSE ~= CE por tarefa (faixas se sobrepoem)?  {}".format("SIM" if ce_vs_mse else "NAO"),
              "  0 ocultas ~= 128 ocultas (faixas se sobrepoem)? {}".format("SIM" if arch_ovl else "NAO"),
              "  => diferencas dentro do ruido da inicializacao." if (ce_vs_mse and arch_ovl)
              else "  => ao menos uma diferenca excede o std.", ""]

    with open(os.path.join(OUT, "report_multiseed.txt"), "w") as f:
        f.write("\n".join(lines))

    # -- figura: barras com yerr=std, piso adaptativo ------------------------
    colors = {"ce": "#4e79a7", "mse": "#f28e2b"}
    short = {"has_people": "has_people", "screenshot_vs_photo": "shot_vs_photo",
             "macro_vs_rest": "macro_vs_rest"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    xpos = np.arange(len(sp.TASKS))
    width = 0.38
    for ax, metric, mlabel in ((axes[0], "f1", "F1"), (axes[1], "accuracy", "Acuracia")):
        allm, alls = [], []
        for i, ls in enumerate(("ce", "mse")):
            means = [stats[t][ls][metric][0] for t in sp.TASKS]
            stds = [stats[t][ls][metric][1] for t in sp.TASKS]
            allm += means; alls += stds
            ax.bar(xpos + (i - 0.5) * width, means, width, yerr=stds, capsize=6,
                   label=ls.upper(), color=colors[ls], edgecolor="k", linewidth=0.7,
                   error_kw=dict(ecolor="#333", lw=1.3))
        ax.set_xticks(xpos)
        ax.set_xticklabels([short[t] for t in sp.TASKS], rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("{} (teste)".format(mlabel))
        ax.set_title("{} por tarefa: CE vs MSE".format(mlabel))
        _adaptive_ylim(ax, allm, alls)
        ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    # painel 3: ablacao de arquitetura (F1 vs nº de ocultos), com barras de erro
    ax = axes[2]
    labels = ["0(log)" if h == 0 else str(h) for h in ARCH_SIZES]
    ameans = [stats_arch[h][0] for h in ARCH_SIZES]
    astds = [stats_arch[h][1] for h in ARCH_SIZES]
    ax.bar(np.arange(len(ARCH_SIZES)), ameans, 0.6, yerr=astds, capsize=6,
           color="#59a14f", edgecolor="k", linewidth=0.7, error_kw=dict(ecolor="#333", lw=1.3))
    ax.set_xticks(np.arange(len(ARCH_SIZES))); ax.set_xticklabels(labels)
    ax.set_xlabel("neuronios na camada oculta"); ax.set_ylabel("F1 (teste)")
    ax.set_title("Ablacao arquitetura ({}), CE".format(task_cmp))
    _adaptive_ylim(ax, ameans, astds)
    ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Projeto 9 — multi-seed: media +/- desvio ({} seeds)".format(N_SEEDS), fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    ms_path = os.path.join(OUT, "metrics_multiseed.png")
    plt.savefig(ms_path, dpi=120, bbox_inches="tight"); plt.close()
    return stats, lines, ms_path


def main():
    print("=== Projeto 9 — MLP manual (numpy) ===\n")

    # -- 0. Multi-seed: RESULTADO OFICIAL (media +/- desvio sobre N_SEEDS) -----
    _, ms_report, ms_path = run_multiseed()
    print("\n".join(ms_report))
    print("Saved:", ms_path)
    print()

    print("[ilustracao seed=42, 1 realizacao — numeros oficiais = multi-seed acima]\n")

    # -- 1. Ablação principal: MLP (1 hidden, 64, relu, CE) full-dim nas 3 tarefas
    print("[1] Desempenho FULL-DIM (CLIP) por tarefa — MLP[D-64-1], relu, CE (seed=42)")
    rows = []
    for task in sp.TASKS:
        d = sp.prepare(task, features="clip")
        D = d["X_train"].shape[1]
        net = MLP([D, 64, 1], activation="relu", loss="ce")
        net.fit(d["X_train"], d["y_train"], epochs=60, lr=0.1)
        m = sp.evaluate(d["y_test"], net.predict(d["X_test"]))
        base = max(np.mean(d["y_test"] == 0), np.mean(d["y_test"] == 1))
        rows.append((task, m, base))
        print("  {:<20} acc={:.3f} f1={:.3f} (baseline {:.3f})".format(
            task, m["accuracy"], m["f1"], base))
    print()

    # -- 2. MSE vs Cross-Entropy (mesma arquitetura) numa tarefa
    task_cmp = "macro_vs_rest"
    d = sp.prepare(task_cmp, features="clip")
    D = d["X_train"].shape[1]
    print("[2] MSE vs CE em '{}' (MLP[D-64-1], relu)".format(task_cmp))
    curves = {}
    cmp_metrics = {}
    for loss in ("ce", "mse"):
        net = MLP([D, 64, 1], activation="relu", loss=loss)
        curves[loss] = net.fit(d["X_train"], d["y_train"], epochs=60, lr=0.1)
        cmp_metrics[loss] = sp.evaluate(d["y_test"], net.predict(d["X_test"]))
        print("  {:<3} -> acc={:.3f} f1={:.3f} loss_final={:.4f}".format(
            loss.upper(), cmp_metrics[loss]["accuracy"], cmp_metrics[loss]["f1"], curves[loss][-1]))
    print()

    # -- 3. Ablação de arquitetura (nº de neurônios na camada oculta) — CE
    print("[3] Ablação de arquitetura em '{}' (hidden size)".format(task_cmp))
    arch_sizes = [0, 2, 8, 32, 128]   # 0 = sem camada oculta (= logística)
    arch_f1 = []
    for h in arch_sizes:
        sizes = [D, 1] if h == 0 else [D, h, 1]
        net = MLP(sizes, activation="relu", loss="ce")
        net.fit(d["X_train"], d["y_train"], epochs=60, lr=0.1)
        f1 = sp.evaluate(d["y_test"], net.predict(d["X_test"]))["f1"]
        arch_f1.append(f1)
        print("  hidden={:<4} f1={:.3f}".format(h if h else "0(log)", f1))
    print()

    # -- 4. Ilustração 2D: partições dos neurônios (treina em 2D PCA, só p/ viz)
    print("[4] Ilustração 2D das partições dos neurônios (treina em PCA-2D)")
    H = 4
    net2d = MLP([2, H, 1], activation="tanh", loss="ce")
    net2d.fit(d["Z_train"], d["y_train"], epochs=300, lr=0.2)

    def neuron_lines(ax):
        W1, b1 = net2d.W[0], net2d.b[0]           # (2,H), (H,)
        xs = np.array([d["Z_train"][:, 0].min() - 0.5, d["Z_train"][:, 0].max() + 0.5])
        for j in range(W1.shape[1]):
            w0, w1 = W1[0, j], W1[1, j]
            if abs(w1) > 1e-6:
                ys = -(w0 * xs + b1[j]) / w1
                ax.plot(xs, ys, "k--", lw=1, alpha=0.6)

    fig, ax = plt.subplots(figsize=(6, 5))
    sp.plot_boundary(ax, net2d.predict, d["Z_test"], d["y_test"],
                     "Partições de {} neurônios ocultos — {}\n(ilustração 2D PCA)".format(H, task_cmp),
                     extra=neuron_lines)
    p = os.path.join(OUT, "boundary_neurons.png")
    plt.tight_layout(); plt.savefig(p, dpi=120, bbox_inches="tight"); plt.close()
    print("  Saved:", p)

    # -- plots: MSE vs CE curves + arch ablation
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(curves["ce"], label="Cross-Entropy", color="#4e79a7")
    axes[0].plot(curves["mse"], label="MSE", color="#f28e2b")
    axes[0].set_xlabel("época"); axes[0].set_ylabel("loss (treino)")
    axes[0].set_title("Convergência: MSE vs CE ({})".format(task_cmp)); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot([str(h) if h else "0(log)" for h in arch_sizes], arch_f1, "o-", color="#4e79a7")
    axes[1].set_xlabel("neurônios na camada oculta"); axes[1].set_ylabel("F1 (teste)")
    axes[1].set_title("Ablação de arquitetura ({})".format(task_cmp)); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(OUT, "mse_vs_ce_and_arch.png")
    plt.savefig(p2, dpi=120, bbox_inches="tight"); plt.close()
    print("  Saved:", p2)

    # -- report
    lines = ["=" * 60, "PROJETO 9 — MLP MANUAL (numpy)", "=" * 60, "",
             "Problema real: classificação binária sobre imagens pessoais (CLIP full-dim).",
             "Desempenho reportado em full-dim; 2D PCA só ilustra partições.", "",
             "RESULTADO OFICIAL (multi-seed):"] + ms_report + [
             "", "-" * 60,
             "Ilustração (seed=42, 1 realização) — base das figuras de partição/curvas:",
             "",
             "[1] Desempenho por tarefa (MLP[D-64-1], relu, CE):"]
    for task, m, base in rows:
        lines.append("  {:<20} acc={:.4f} prec={:.4f} rec={:.4f} f1={:.4f} (baseline {:.4f})".format(
            task, m["accuracy"], m["precision"], m["recall"], m["f1"], base))
    lines += ["", "[2] MSE vs CE em {}:".format(task_cmp)]
    for loss in ("ce", "mse"):
        m = cmp_metrics[loss]
        lines.append("  {:<3} acc={:.4f} f1={:.4f} loss_final={:.4f}".format(
            loss.upper(), m["accuracy"], m["f1"], curves[loss][-1]))
    lines += ["", "[3] Ablação de arquitetura ({}):".format(task_cmp)]
    for h, f1 in zip(arch_sizes, arch_f1):
        lines.append("  hidden={:<8} f1={:.4f}".format(h if h else "0(log)", f1))
    with open(os.path.join(OUT, "report.txt"), "w") as f:
        f.write("\n".join(lines))
    print("\nDone.")


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description="MLP manual (numpy) — MSE vs CE, multi-seed")
    _ap.add_argument("-n", "--n-seeds", type=int, default=N_SEEDS,
                     help="numero de seeds para media +/- desvio (default: %(default)s)")
    _args = _ap.parse_args()
    N_SEEDS = _args.n_seeds
    SEEDS = [SEED + i for i in range(N_SEEDS)]
    main()
