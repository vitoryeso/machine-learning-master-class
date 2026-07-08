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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shared_problem as sp

SEED = 42
SUB = 6000   # subconjunto de treino p/ tornar o SVM tratável
N_SEEDS = 3  # rodadas para media +/- desvio; suba p/ 5/10 (SVM é O(N^2), cuidado)
SEEDS = [SEED + i for i in range(N_SEEDS)]  # [42, 43, 44]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)


def subsample(X, y, n, seed=SEED):
    if len(X) <= n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), n, replace=False)
    return X[idx], y[idx]


# =============================================================================
# Multi-seed — robustez estatistica (media +/- desvio sobre N_SEEDS rodadas)
# =============================================================================
# Onde entra a aleatoriedade: a SVC(rbf) é DETERMINÍSTICA dado (dados, split).
# As duas fontes de variação por seed são (a) o split treino/teste estratificado
# e (b) o subsample de 6000 (SVM é O(N^2), não cabe o dataset inteiro). Variamos
# AMBOS por seed. sp.prepare() fixa SEED=42 internamente, então replicamos aqui o
# pipeline full-dim padronizado com random_state variável.
def _prepare_seed(task, seed, features="clip", test_size=0.2):
    """Split estratificado + padronização (fit no treino) com seed variável.
    Retorna (X_train_std, X_test_std, y_train, y_test) em full-dim (métricas)."""
    mask, y, pos, neg = sp.get_task(task)
    X = sp.load_features(features)[mask]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), ytr, yte


# grade do sweep C × gamma (mesma do main); agregada p/ testar o claim "γ crítico"
MS_CS = [0.1, 1, 10, 100]
MS_GAMMAS = ["scale", 0.001, 0.01, 0.1]


def run_multiseed(n_seeds=N_SEEDS):
    """Roda as 3 tarefas (C=10, gamma=scale) sobre n_seeds realizações e agrega
    media +/- desvio amostral (ddof=1) de acc, F1 e nº de vetores de suporte.
    Também agrega a grade C×gamma (F1) na macro p/ verificar se "γ crítico"
    sobrevive ao desvio. Escreve report_multiseed.txt + metrics_multiseed.png.
    Aditivo: não toca na análise single-seed (seed=42) do main().
    Retorna (stats, report_lines, png_path)."""
    seeds = [SEED + i for i in range(n_seeds)]
    agg = {t: {"accuracy": [], "f1": [], "nsv": [], "base": []} for t in sp.TASKS}
    task_ab = "macro_vs_rest"
    grids = []  # uma grade F1 (len(Cs) x len(gammas)) por seed

    for seed in seeds:
        # [A] métricas por tarefa no ponto de operação (C=10, gamma=scale)
        for task in sp.TASKS:
            Xtr, Xte, ytr, yte = _prepare_seed(task, seed)
            Xtr, ytr = subsample(Xtr, ytr, SUB, seed=seed)
            clf = SVC(kernel="rbf", C=10, gamma="scale").fit(Xtr, ytr)
            m = sp.evaluate(yte, clf.predict(Xte))
            agg[task]["accuracy"].append(m["accuracy"])
            agg[task]["f1"].append(m["f1"])
            agg[task]["nsv"].append(int(clf.n_support_.sum()))
            agg[task]["base"].append(float(max(np.mean(yte == 0), np.mean(yte == 1))))
        # [B] varredura C × gamma na macro (F1) p/ o claim do gamma
        Xtr, Xte, ytr, yte = _prepare_seed(task_ab, seed)
        Xtr, ytr = subsample(Xtr, ytr, SUB, seed=seed)
        g = np.zeros((len(MS_CS), len(MS_GAMMAS)))
        for i, C in enumerate(MS_CS):
            for j, gm in enumerate(MS_GAMMAS):
                clf = SVC(kernel="rbf", C=C, gamma=gm).fit(Xtr, ytr)
                g[i, j] = sp.evaluate(yte, clf.predict(Xte))["f1"]
        grids.append(g)

    def ms(v):
        return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)

    stats = {t: {k: ms(agg[t][k]) for k in ("accuracy", "f1", "nsv", "base")}
             for t in sp.TASKS}
    grids = np.array(grids)
    grid_mean = grids.mean(axis=0)
    grid_std = (grids.std(axis=0, ddof=1) if len(grids) > 1
                else np.zeros_like(grid_mean))

    # --- relatório texto ---
    lines = ["Multi-seed: media +/- desvio (ddof=1) sobre {} seeds {}".format(n_seeds, seeds),
             "(cada seed = novo split estratificado + novo subsample de {})".format(SUB),
             "SVC(kernel='rbf', C=10, gamma='scale'); features CLIP full-dim.", ""]
    hdr = "{:<20} {:>16} {:>16} {:>16} {:>10}".format(
        "Tarefa", "Accuracy", "F1", "SVs", "Base")
    lines += [hdr, "-" * len(hdr)]
    for t in sp.TASKS:
        am, asd = stats[t]["accuracy"]
        fm, fsd = stats[t]["f1"]
        nm, nsd = stats[t]["nsv"]
        bm, _ = stats[t]["base"]
        lines.append("{:<20} {:>9.4f}+/-{:<5.4f} {:>9.4f}+/-{:<5.4f} {:>8.0f}+/-{:<6.0f} {:>10.4f}".format(
            t, am, asd, fm, fsd, nm, nsd, bm))
    lines.append("-" * len(hdr))
    lines.append("")

    # --- claim 1: "γ crítico" na varredura C×gamma (macro), F1 mean+/-std ---
    lines.append("Varredura C x gamma em '{}' — F1 media +/- desvio ({} seeds):".format(
        task_ab, n_seeds))
    lines.append("       gamma=" + "  ".join("{:>13}".format(str(gm)) for gm in MS_GAMMAS))
    for i, C in enumerate(MS_CS):
        cells = ["{:.3f}+/-{:.3f}".format(grid_mean[i, j], grid_std[i, j])
                 for j in range(len(MS_GAMMAS))]
        lines.append("  C={:<5} {}".format(C, "  ".join("{:>13}".format(c) for c in cells)))
    lines.append("")
    # melhor vs pior célula e se o gap sobrevive ao desvio combinado
    flat = grid_mean.ravel()
    bi, wi = int(flat.argmax()), int(flat.argmin())
    br, bc = divmod(bi, len(MS_GAMMAS))
    wr, wc = divmod(wi, len(MS_GAMMAS))
    gap = grid_mean[br, bc] - grid_mean[wr, wc]
    comb = float(np.hypot(grid_std[br, bc], grid_std[wr, wc]))
    lines.append("Claim 'gamma critico': melhor cell C={}, gamma={} -> F1={:.3f}+/-{:.3f}".format(
        MS_CS[br], MS_GAMMAS[bc], grid_mean[br, bc], grid_std[br, bc]))
    lines.append("                       pior  cell C={}, gamma={} -> F1={:.3f}+/-{:.3f}".format(
        MS_CS[wr], MS_GAMMAS[wc], grid_mean[wr, wc], grid_std[wr, wc]))
    lines.append("  gap melhor-pior = {:.3f}  |  desvio combinado (hypot) = {:.3f}  -> {}".format(
        gap, comb, "SOBREVIVE ao std" if gap > comb else "NAO sobrevive ao std"))
    lines.append("")
    # --- claim 2: SVM bate o baseline (classe majoritaria)? ---
    lines.append("Claim 'SVM > baseline': acc - base por tarefa (sobrevive se margem > desvio):")
    for t in sp.TASKS:
        am, asd = stats[t]["accuracy"]
        bm, _ = stats[t]["base"]
        marg = am - bm
        lines.append("  {:<20} acc={:.4f}+/-{:.4f}  base={:.4f}  margem={:+.4f}  -> {}".format(
            t, am, asd, bm, marg, "SOBREVIVE" if marg > asd else "dentro do ruido"))

    with open(os.path.join(OUT, "report_multiseed.txt"), "w") as f:
        f.write("\n".join(lines))

    # --- figura: barras acc/F1 por tarefa com yerr=std, piso adaptativo ---
    fig, ax = plt.subplots(figsize=(8, 4.8))
    xpos = np.arange(len(sp.TASKS))
    width = 0.38
    metric_style = [("accuracy", "Accuracy", "#4e79a7"), ("f1", "F1", "#e15759")]
    lo_vals, hi_vals = [], []
    for i, (k, lbl, col) in enumerate(metric_style):
        means = [stats[t][k][0] for t in sp.TASKS]
        stds = [stats[t][k][1] for t in sp.TASKS]
        ax.bar(xpos + (i - 0.5) * width, means, width, yerr=stds, capsize=6,
               label=lbl, color=col, edgecolor="k", linewidth=0.7,
               error_kw=dict(ecolor="#333", lw=1.3))
        lo_vals += [m - s for m, s in zip(means, stds)]
        hi_vals += [m + s for m, s in zip(means, stds)]
    # piso ADAPTATIVO (F1 varia ~0.76..0.95): não ancora em 0 nem fixo
    ymin = max(0.0, min(lo_vals) - 0.04)
    ymax = min(1.02, max(hi_vals) + 0.04)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(xpos)
    ax.set_xticklabels(sp.TASKS, fontsize=9)
    ax.set_ylabel("Score (teste)")
    ax.set_title("SVM-RBF (C=10, gamma=scale): media +/- desvio ({} seeds)".format(n_seeds))
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    ms_path = os.path.join(OUT, "metrics_multiseed.png")
    plt.savefig(ms_path, dpi=120, bbox_inches="tight")
    plt.close()
    return stats, lines, ms_path


def main():
    print("=== Projeto 8 — SVM kernel RBF ===\n")

    # [0] Multi-seed: RESULTADO OFICIAL (media +/- desvio sobre N_SEEDS) --------
    _, ms_report, ms_path = run_multiseed(N_SEEDS)
    print("\n".join(ms_report))
    print("Saved:", ms_path, "\n")

    # [1] full-dim, 3 tarefas (treino subamostrado)
    # ILUSTRAÇÃO (seed=42, 1 realização). Números OFICIAIS = multi-seed acima.
    print("[1] ILUSTRAÇÃO seed=42 — FULL-DIM (CLIP), SVC(rbf, C=10, gamma=scale), treino<= {}".format(SUB))
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
    ax.set_title("SVM-RBF: F1 (teste) por C × gamma — {}\n(ilustração seed=42, 1 realização)".format(task_ab))
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
             "RESULTADO OFICIAL = multi-seed (ver report_multiseed.txt / metrics_multiseed.png).",
             "As linhas abaixo são ILUSTRAÇÃO (seed=42, 1 realização) — base das figuras.", "",
             "[1] Desempenho por tarefa (C=10, gamma=scale) [seed=42]:"]
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
    import argparse
    _ap = argparse.ArgumentParser(description="Projeto 8 — SVM-RBF multi-seed")
    _ap.add_argument("-n", "--n-seeds", type=int, default=N_SEEDS,
                     help="numero de seeds para media +/- desvio (default: %(default)s)")
    _args = _ap.parse_args()
    N_SEEDS = _args.n_seeds
    SEEDS = [SEED + i for i in range(N_SEEDS)]
    main()
