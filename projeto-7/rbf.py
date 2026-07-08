#!/usr/bin/env python3
"""Projeto 7 — Rede RBF construída manualmente (numpy).

Enunciado: RBF Network p/ classificação binária; comparar estratégias de centros
(aleatório / K-means / subconjunto); mostrar no plano 2D como as funções radiais
particionam o espaço.

Problema REAL (compartilhado com 8/9): classificação binária sobre as imagens
pessoais (features CLIP), 3 tarefas. Desempenho em full-dim; 2D PCA só ilustra.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shared_problem as sp

SEED = 42
N_SEEDS = 3  # rodadas p/ media +/- desvio; suba p/ 10/30 p/ std mais firme
SEEDS = [SEED + i for i in range(N_SEEDS)]  # [42, 43, 44]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)


class RBFNet:
    """RBF Network: camada oculta de gaussianas + saída linear (least squares).
    center_strategy: 'kmeans' | 'random' | 'subset'."""
    def __init__(self, n_centers=30, center_strategy="kmeans", seed=SEED):
        self.k = n_centers
        self.strategy = center_strategy
        self.seed = seed

    def _pick_centers(self, X):
        rng = np.random.default_rng(self.seed)
        if self.strategy == "random":
            idx = rng.choice(len(X), self.k, replace=False)
            return X[idx]
        if self.strategy == "subset":
            return X[:self.k]                      # primeiros k (subconjunto fixo)
        km = KMeans(n_clusters=self.k, random_state=self.seed, n_init=4).fit(X)
        return km.cluster_centers_                 # default: K-means

    def _sqdist(self, X):
        # ||x - c||^2 = |x|^2 + |c|^2 - 2 x·c
        d2 = (np.sum(X ** 2, 1)[:, None] + np.sum(self.C ** 2, 1)[None, :]
              - 2 * X @ self.C.T)
        np.maximum(d2, 0, out=d2)
        return d2

    def _phi(self, X):
        return np.exp(-self._sqdist(X) / (2 * self.sigma ** 2))

    def fit(self, X, y):
        self.C = self._pick_centers(X)
        # sigma calibrado pela MEDIANA das distâncias ponto->centro: garante que
        # as gaussianas tenham espalhamento útil em qualquer dimensão (o heurístico
        # d_max/sqrt(2k) colapsa em alta-dim, zerando as ativações).
        d2 = self._sqdist(X)
        self.sigma = np.sqrt(max(np.median(d2), 1e-9))
        Phi = np.hstack([np.exp(-d2 / (2 * self.sigma ** 2)), np.ones((len(X), 1))])
        self.w, *_ = np.linalg.lstsq(Phi, y.astype(float), rcond=None)
        return self

    def predict(self, X):
        Phi = np.hstack([self._phi(X), np.ones((len(X), 1))])
        return (Phi @ self.w >= 0.5).astype(int)


# =============================================================================
# Multi-seed — robustez estatistica (media +/- desvio sobre N_SEEDS rodadas)
# =============================================================================
def prepare_seeded(task, features="clip", test_size=0.2, seed=SEED):
    """Como sp.prepare, mas com a SEED do split treino/teste variavel.

    Os dados reais (features CLIP) sao FIXOS; sp.prepare fixa random_state=SEED.
    Aqui variamos a seed p/ obter realizacoes diferentes do split estratificado
    (a seed dos centros do RBFNet tambem varia, no run_multiseed). Reusa
    sp.get_task/sp.load_features; devolve so o full-dim padronizado (metricas
    reais) — a projecao 2D PCA e ilustrativa e nao entra na media."""
    mask, y, pos, neg = sp.get_task(task)
    X = sp.load_features(features)[mask]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr)
    return {"X_train": sc.transform(Xtr), "X_test": sc.transform(Xte),
            "y_train": ytr, "y_test": yte, "task": task, "pos": pos, "neg": neg,
            "balance": [int((y == 0).sum()), int((y == 1).sum())]}


STRATEGIES = ["kmeans", "random", "subset"]
MS_KEYS = ["accuracy", "f1", "precision", "recall"]
MS_K = 30  # nº de centros fixo nas rodadas multi-seed


def run_multiseed(features="clip"):
    """Roda as 3 tarefas x 3 estrategias de centro (k=30) sobre N_SEEDS seeds e
    agrega media +/- desvio amostral (ddof=1). Por seed varia: (i) o split
    treino/teste e (ii) a init dos centros (kmeans/aleatorio). Escreve
    report_multiseed.txt + metrics_multiseed.png. Aditivo: nao mexe no main().
    Retorna (stats, report_lines, png_path)."""
    # agg[task][strategy][metric] = lista sobre seeds
    agg = {t: {s: {k: [] for k in MS_KEYS} for s in STRATEGIES} for t in sp.TASKS}
    for seed in SEEDS:
        for task in sp.TASKS:
            d = prepare_seeded(task, features=features, seed=seed)
            for s in STRATEGIES:
                net = RBFNet(MS_K, s, seed=seed).fit(d["X_train"], d["y_train"])
                m = sp.evaluate(d["y_test"], net.predict(d["X_test"]))
                for k in MS_KEYS:
                    agg[task][s][k].append(m[k])

    def ms(v):
        return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)

    stats = {t: {s: {k: ms(agg[t][s][k]) for k in MS_KEYS} for s in STRATEGIES}
             for t in sp.TASKS}

    # ---- report -------------------------------------------------------------
    lines = ["Multi-seed: media +/- desvio (ddof=1) sobre {} seeds {}".format(N_SEEDS, SEEDS),
             "(cada seed = novo split estratificado + nova init de centros; k={})".format(MS_K),
             "Dados CLIP reais sao fixos; a seed varia so o split e os centros.", ""]
    # [A] tabela oficial: metricas por tarefa (estrategia = kmeans)
    lines += ["[A] Desempenho por tarefa (centros K-means, k={}) — RESULTADO OFICIAL:".format(MS_K)]
    hdr = "{:<22} {:>16} {:>16} {:>16} {:>16}".format("Tarefa", "accuracy", "f1", "precision", "recall")
    lines += [hdr, "-" * len(hdr)]
    for t in sp.TASKS:
        cells = []
        for k in MS_KEYS:
            mn, sd = stats[t]["kmeans"][k]
            cells.append("{:.3f}+/-{:.3f}".format(mn, sd))
        lines.append("{:<22} {:>16} {:>16} {:>16} {:>16}".format(t, *cells))
    lines += ["-" * len(hdr), ""]
    # [B] ranking de estrategias de centro (F1 mean+/-std por tarefa)
    lines += ["[B] Ablacao de centros — F1 (media +/- desvio) por tarefa x estrategia:"]
    hdrb = "{:<22} {:>18} {:>18} {:>18}".format("Tarefa", "kmeans", "random", "subset")
    lines += [hdrb, "-" * len(hdrb)]
    for t in sp.TASKS:
        cells = ["{:.3f}+/-{:.3f}".format(*stats[t][s]["f1"]) for s in STRATEGIES]
        lines.append("{:<22} {:>18} {:>18} {:>18}".format(t, *cells))
    lines += ["-" * len(hdrb)]

    # ---- PNG: barras acc/F1 por tarefa (kmeans) com yerr=std ----------------
    metrics_to_plot = [("accuracy", "Acuracia"), ("f1", "F1")]
    colors = {"accuracy": "#4e79a7", "f1": "#59a14f"}
    fig, ax = plt.subplots(figsize=(8, 4.8))
    xpos = np.arange(len(sp.TASKS))
    width = 0.38
    lows = []
    for i, (k, lbl) in enumerate(metrics_to_plot):
        means = [stats[t]["kmeans"][k][0] for t in sp.TASKS]
        stds = [stats[t]["kmeans"][k][1] for t in sp.TASKS]
        lows += [m - s for m, s in zip(means, stds)]
        ax.bar(xpos + (i - 0.5) * width, means, width, yerr=stds, capsize=6,
               label=lbl, color=colors[k], edgecolor="k", linewidth=0.7,
               error_kw=dict(ecolor="#333", lw=1.3))
    # piso do eixo y ADAPTATIVO: ~min(media-std) - folga, arredondado p/ baixo (0.05)
    floor = max(0.0, np.floor((min(lows) - 0.03) * 20) / 20)
    ax.set_ylim(floor, 1.0)
    ax.set_xticks(xpos)
    ax.set_xticklabels(sp.TASKS, rotation=12, ha="right")
    ax.set_ylabel("Score (teste)")
    ax.set_title("RBF (centros K-means, k={}): media +/- desvio sobre {} seeds".format(MS_K, N_SEEDS))
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    ms_path = os.path.join(OUT, "metrics_multiseed.png")
    plt.savefig(ms_path, dpi=120, bbox_inches="tight")
    plt.close()

    with open(os.path.join(OUT, "report_multiseed.txt"), "w") as f:
        f.write("\n".join(lines))
    return stats, lines, ms_path


def main():
    print("=== Projeto 7 — Rede RBF manual ===\n")

    # [0] Multi-seed: RESULTADO OFICIAL (media +/- desvio) --------------------
    _, ms_report, ms_path = run_multiseed()
    print("\n".join(ms_report))
    print("Saved:", ms_path, "\n")

    # [1] full-dim, 3 tarefas (centros K-means, k=30) — ILUSTRACAO seed=42
    print("[1] Desempenho FULL-DIM (CLIP), RBF[k=30, K-means] "
          "[ilustracao seed=42, 1 realizacao — oficial = multi-seed acima]")
    rows = []
    for task in sp.TASKS:
        d = sp.prepare(task, features="clip")
        net = RBFNet(30, "kmeans").fit(d["X_train"], d["y_train"])
        m = sp.evaluate(d["y_test"], net.predict(d["X_test"]))
        base = max(np.mean(d["y_test"] == 0), np.mean(d["y_test"] == 1))
        rows.append((task, m, base))
        print("  {:<20} acc={:.3f} f1={:.3f} (base {:.3f})".format(task, m["accuracy"], m["f1"], base))
    print()

    # [2] ablação de estratégia de centros × k (tarefa macro)
    task_ab = "macro_vs_rest"
    d = sp.prepare(task_ab, features="clip")
    print("[2] Ablação de centros em '{}' (estratégia × k)".format(task_ab))
    ks = [10, 30, 60]
    strategies = ["random", "subset", "kmeans"]
    ab = {s: [] for s in strategies}
    for s in strategies:
        for k in ks:
            net = RBFNet(k, s).fit(d["X_train"], d["y_train"])
            f1 = sp.evaluate(d["y_test"], net.predict(d["X_test"]))["f1"]
            ab[s].append(f1)
        print("  {:<8} f1@k{}={}".format(s, ks, ["{:.3f}".format(v) for v in ab[s]]))
    print()

    # [3] ilustração 2D: fronteira + centros (treina em PCA-2D) — seed=42
    print("[3] Ilustração 2D (fronteira + centros), k=12 K-means [seed=42, 1 realização]")
    net2d = RBFNet(12, "kmeans").fit(d["Z_train"], d["y_train"])

    def show_centers(ax):
        ax.scatter(net2d.C[:, 0], net2d.C[:, 1], marker="*", s=220,
                   c="white", edgecolors="k", linewidths=1.2, label="centros", zorder=5)
        ax.legend(loc="upper right", fontsize=8)

    fig, ax = plt.subplots(figsize=(6, 5))
    sp.plot_boundary(ax, net2d.predict, d["Z_test"], d["y_test"],
                     "RBF: 12 centros (K-means) particionando o plano — {}\n(ilustração 2D PCA, seed=42, 1 realização)".format(task_ab),
                     extra=show_centers)
    p1 = os.path.join(OUT, "boundary_centers.png")
    plt.tight_layout(); plt.savefig(p1, dpi=120, bbox_inches="tight"); plt.close()
    print("  Saved:", p1)

    # plot ablação
    fig, ax = plt.subplots(figsize=(7, 4))
    for s in strategies:
        ax.plot([str(k) for k in ks], ab[s], "o-", label=s)
    ax.set_xlabel("nº de centros (k)"); ax.set_ylabel("F1 (teste)")
    ax.set_title("RBF: estratégia de centros vs k ({})".format(task_ab))
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(OUT, "centers_ablation.png")
    plt.savefig(p2, dpi=120, bbox_inches="tight"); plt.close()
    print("  Saved:", p2)

    lines = ["=" * 60, "PROJETO 7 — REDE RBF MANUAL", "=" * 60, "",
             "Problema real: classificação binária (imagens pessoais, CLIP full-dim).",
             "RBF: gaussianas + saída least-squares; sigma = mediana das distâncias ponto→centro.", "",
             "RESULTADO OFICIAL (multi-seed):"] + ms_report + [
             "",
             "-" * 60,
             "Ilustração (seed=42, 1 realização) — base das figuras fronteira/ablação:",
             "[1] Desempenho por tarefa (k=30, K-means):"]
    for task, m, base in rows:
        lines.append("  {:<20} acc={:.4f} prec={:.4f} rec={:.4f} f1={:.4f} (base {:.4f})".format(
            task, m["accuracy"], m["precision"], m["recall"], m["f1"], base))
    lines += ["", "[2] Ablação de centros ({}), F1 por k={}:".format(task_ab, ks)]
    for s in strategies:
        lines.append("  {:<8} {}".format(s, ["{:.4f}".format(v) for v in ab[s]]))
    with open(os.path.join(OUT, "report.txt"), "w") as f:
        f.write("\n".join(lines))
    print("\nDone.")


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description="Rede RBF manual — multi-seed")
    _ap.add_argument("-n", "--n-seeds", type=int, default=N_SEEDS,
                     help="numero de seeds para media +/- desvio (default: %(default)s)")
    _args = _ap.parse_args()
    N_SEEDS = _args.n_seeds
    SEEDS = [SEED + i for i in range(N_SEEDS)]
    main()
