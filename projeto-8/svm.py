#!/usr/bin/env python3
"""Projeto 8 — SVM com kernel RBF.

Enunciado: treinar SVM-RBF p/ classificação binária; interpretar os vetores de
suporte como "centros"; varrer C e gamma; mostrar a fronteira 2D destacando os
vetores de suporte. (NÃO pede "manual" → sklearn.SVC é legítimo.)

Problema REAL (compartilhado com 7/9): classificação binária sobre as imagens
pessoais (CLIP). Full-dim para métricas; 2D PCA para ilustrar. SVM escala mal com
N (O(N^2)), então o treino full-dim usa um subconjunto amostrado aleatoriamente.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shared_problem as sp

SEED = 42
SUB = 6000   # subconjunto de treino p/ tornar o SVM tratável
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)


def subsample(X, y, n, seed=SEED):
    if len(X) <= n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), n, replace=False)
    return X[idx], y[idx]


def main():
    print("=== Projeto 8 — SVM kernel RBF ===\n")

    # [1] full-dim, 3 tarefas (treino subamostrado)
    print("[1] Desempenho FULL-DIM (CLIP), SVC(rbf, C=10, gamma=scale), treino<= {}".format(SUB))
    rows = []
    for task in sp.TASKS:
        d = sp.prepare(task, features="clip")
        Xtr, ytr = subsample(d["X_train"], d["y_train"], SUB)
        clf = SVC(kernel="rbf", C=10, gamma="scale").fit(Xtr, ytr)
        m = sp.evaluate(d["y_test"], clf.predict(d["X_test"]))
        base = max(np.mean(d["y_test"] == 0), np.mean(d["y_test"] == 1))
        nsv = int(clf.n_support_.sum())
        rows.append((task, m, base, nsv, len(Xtr)))
        print("  {:<20} acc={:.3f} f1={:.3f} (base {:.3f}) SVs={}/{}".format(
            task, m["accuracy"], m["f1"], base, nsv, len(Xtr)))
    print()

    # [2] varredura C × gamma (tarefa macro, subamostrado)
    task_ab = "macro_vs_rest"
    d = sp.prepare(task_ab, features="clip")
    Xtr, ytr = subsample(d["X_train"], d["y_train"], SUB)
    Cs = [0.1, 1, 10, 100]
    gammas = ["scale", 0.001, 0.01, 0.1]
    print("[2] Varredura C × gamma em '{}' (F1 teste)".format(task_ab))
    grid = np.zeros((len(Cs), len(gammas)))
    for i, C in enumerate(Cs):
        for j, g in enumerate(gammas):
            clf = SVC(kernel="rbf", C=C, gamma=g).fit(Xtr, ytr)
            grid[i, j] = sp.evaluate(d["y_test"], clf.predict(d["X_test"]))["f1"]
        print("  C={:<5} {}".format(C, ["{:.3f}".format(v) for v in grid[i]]))
    print()

    # [3] ilustração 2D: fronteira + vetores de suporte (treina em PCA-2D)
    print("[3] Ilustração 2D (fronteira + vetores de suporte)")
    clf2d = SVC(kernel="rbf", C=10, gamma="scale").fit(d["Z_train"], d["y_train"])
    sv = clf2d.support_vectors_

    def show_svs(ax):
        ax.scatter(sv[:, 0], sv[:, 1], s=60, facecolors="none",
                   edgecolors="k", linewidths=0.7, label="vetores de suporte", zorder=5)
        ax.legend(loc="upper right", fontsize=8)

    fig, ax = plt.subplots(figsize=(6, 5))
    sp.plot_boundary(ax, clf2d.predict, d["Z_test"], d["y_test"],
                     "SVM-RBF: fronteira e vetores de suporte — {}\n(ilustração 2D PCA, {} SVs)".format(
                         task_ab, int(clf2d.n_support_.sum())),
                     extra=show_svs)
    p1 = os.path.join(OUT, "boundary_svs.png")
    plt.tight_layout(); plt.savefig(p1, dpi=120, bbox_inches="tight"); plt.close()
    print("  Saved:", p1)

    # heatmap C × gamma
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(grid, cmap="Blues", vmin=grid.min(), vmax=grid.max(), aspect="auto")
    ax.set_xticks(range(len(gammas))); ax.set_xticklabels([str(g) for g in gammas])
    ax.set_yticks(range(len(Cs))); ax.set_yticklabels([str(c) for c in Cs])
    ax.set_xlabel("gamma"); ax.set_ylabel("C")
    ax.set_title("SVM-RBF: F1 (teste) por C × gamma — {}".format(task_ab))
    for i in range(len(Cs)):
        for j in range(len(gammas)):
            ax.text(j, i, "{:.2f}".format(grid[i, j]), ha="center", va="center",
                    color="white" if grid[i, j] > (grid.min() + grid.max()) / 2 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    p2 = os.path.join(OUT, "c_gamma_sweep.png")
    plt.savefig(p2, dpi=120, bbox_inches="tight"); plt.close()
    print("  Saved:", p2)

    lines = ["=" * 60, "PROJETO 8 — SVM KERNEL RBF", "=" * 60, "",
             "Problema real: classificação binária (imagens pessoais, CLIP full-dim).",
             "SVC(kernel='rbf'); treino subamostrado a {} (SVM escala O(N^2)).".format(SUB), "",
             "[1] Desempenho por tarefa (C=10, gamma=scale):"]
    for task, m, base, nsv, ntr in rows:
        lines.append("  {:<20} acc={:.4f} prec={:.4f} rec={:.4f} f1={:.4f} (base {:.4f}) SVs={}/{}".format(
            task, m["accuracy"], m["precision"], m["recall"], m["f1"], base, nsv, ntr))
    lines += ["", "[2] Varredura C × gamma ({}), F1:".format(task_ab),
              "       gamma=" + "  ".join(str(g) for g in gammas)]
    for i, C in enumerate(Cs):
        lines.append("  C={:<5} {}".format(C, "  ".join("{:.4f}".format(v) for v in grid[i])))
    with open(os.path.join(OUT, "report.txt"), "w") as f:
        f.write("\n".join(lines))
    print("\nDone.")


if __name__ == "__main__":
    main()
