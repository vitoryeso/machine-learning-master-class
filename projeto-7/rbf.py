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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shared_problem as sp

SEED = 42
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


def main():
    print("=== Projeto 7 — Rede RBF manual ===\n")

    # [1] full-dim, 3 tarefas (centros K-means, k=30)
    print("[1] Desempenho FULL-DIM (CLIP), RBF[k=30, K-means]")
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

    # [3] ilustração 2D: fronteira + centros (treina em PCA-2D)
    print("[3] Ilustração 2D (fronteira + centros), k=12 K-means")
    net2d = RBFNet(12, "kmeans").fit(d["Z_train"], d["y_train"])

    def show_centers(ax):
        ax.scatter(net2d.C[:, 0], net2d.C[:, 1], marker="*", s=220,
                   c="white", edgecolors="k", linewidths=1.2, label="centros", zorder=5)
        ax.legend(loc="upper right", fontsize=8)

    fig, ax = plt.subplots(figsize=(6, 5))
    sp.plot_boundary(ax, net2d.predict, d["Z_test"], d["y_test"],
                     "RBF: 12 centros (K-means) particionando o plano — {}\n(ilustração 2D PCA)".format(task_ab),
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
    main()
