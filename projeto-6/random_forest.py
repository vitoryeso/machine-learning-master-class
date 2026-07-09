#!/usr/bin/env python3
"""Projeto 6 — Random Forest: Ensemble e Variabilidade (construido MANUALMENTE).

Enunciado: construir um Random Forest p/ classificacao binaria 2D, mostrando como
MULTIPLAS arvores (bootstrap/bagging + subamostragem de features) melhoram a
ROBUSTEZ vs uma arvore unica; visualizar em 2D as particoes individuais e a
combinada.

Fecha o arco do estudo de variancia dos miniprojetos: o Projeto 5 diagnosticou
(via teste pareado) que a arvore nao-podada e um estimador de ALTA VARIANCIA — o
desvio entre seeds NAO caia com mais dados, porque a variancia e do MODELO. O
Random Forest e a cura classica: o bagging faz a media de muitas arvores diversas
e REDUZ a variancia sem inflar o vies. Medimos isso com o MESMO ferramental
multi-seed/pareado do artigo (arvore-unica vs floresta, na mesma realizacao por
seed): esperamos media >= e, sobretudo, DESVIO ENTRE SEEDS menor na floresta.
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

SEED = 42
N_SEEDS = 10               # rodadas para media +/- desvio (-n muda em runtime)
SEEDS = [SEED + i for i in range(N_SEEDS)]
N_TREES = 51               # arvores na floresta (impar -> voto de maioria sem empate)
MAX_FEATURES = 1           # features candidatas por split (de 2) — o "random" do RF
N_ILUSTRA = 500            # dataset das figuras 2D (seed=42, legivel)
N_EVAL = 2000              # dataset do multi-seed (teste ~400, estimativa precisa)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Impureza de Gini
# =============================================================================
def _probs(y):
    if len(y) == 0:
        return np.array([])
    _, c = np.unique(y, return_counts=True)
    return c / c.sum()


def gini(y):
    p = _probs(y)
    return float(1.0 - np.sum(p ** 2))


# =============================================================================
# Arvore CART manual — com subamostragem de features (o coracao do RF)
# =============================================================================
class Node:
    __slots__ = ("feature", "threshold", "left", "right", "value", "is_leaf")

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.is_leaf = False


class DecisionTree:
    """CART binario. max_features=None usa todas (arvore normal, tipo Projeto 5);
    max_features=k considera k features aleatorias por split (arvore de RF)."""

    def __init__(self, max_depth=None, min_samples_split=2, max_features=None, rng=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        self.n_features_ = X.shape[1]
        self.root = self._build(X, y, 0)
        return self

    def _build(self, X, y, depth):
        node = Node()
        classes, counts = np.unique(y, return_counts=True)
        node.value = int(classes[np.argmax(counts)])
        if (gini(y) == 0.0
                or (self.max_depth is not None and depth >= self.max_depth)
                or len(y) < self.min_samples_split):
            node.is_leaf = True
            return node
        best = self._best_split(X, y)
        if best is None:
            node.is_leaf = True
            return node
        feat, thr, mask = best
        node.feature, node.threshold = feat, thr
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def _best_split(self, X, y):
        n = len(y)
        parent = gini(y)
        best_gain, best = 0.0, None
        feats = range(self.n_features_)
        if self.max_features is not None and self.max_features < self.n_features_:
            feats = self.rng.choice(self.n_features_, size=self.max_features, replace=False)
        for feat in feats:
            vals = np.unique(X[:, feat])
            if len(vals) < 2:
                continue
            thrs = (vals[:-1] + vals[1:]) / 2.0
            for thr in thrs:
                mask = X[:, feat] <= thr
                nl = int(mask.sum())
                nr = n - nl
                if nl == 0 or nr == 0:
                    continue
                child = (nl / n) * gini(y[mask]) + (nr / n) * gini(y[~mask])
                gain = parent - child
                if gain > best_gain:
                    best_gain, best = gain, (feat, float(thr), mask)
        return best

    def _predict_one(self, x):
        node = self.root
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        X = np.asarray(X, float)
        return np.array([self._predict_one(x) for x in X])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


# =============================================================================
# Random Forest manual — bootstrap (bagging) + features aleatorias + voto
# =============================================================================
class RandomForest:
    def __init__(self, n_trees=N_TREES, max_features=MAX_FEATURES, max_depth=None, seed=SEED):
        self.n_trees = n_trees
        self.max_features = max_features
        self.max_depth = max_depth
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        n = len(y)
        rng = np.random.default_rng(self.seed)
        self.trees = []
        for _ in range(self.n_trees):
            idx = rng.integers(0, n, size=n)                 # bootstrap (com reposicao)
            tree = DecisionTree(max_depth=self.max_depth, max_features=self.max_features,
                                rng=np.random.default_rng(int(rng.integers(1 << 31))))
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        votes = np.array([t.predict(X) for t in self.trees])  # (n_trees, n)
        return (votes.mean(0) >= 0.5).astype(int)             # voto de maioria (binario)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


def f1_bin(yt, yp):
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def _make(seed, n):
    X, y = make_classification(n_samples=n, n_features=2, n_informative=2, n_redundant=0,
                               n_clusters_per_class=1, class_sep=0.9, random_state=seed)
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


# =============================================================================
# Multi-seed: arvore unica vs floresta (media+/-desvio, PAREADO, reducao de var.)
# =============================================================================
def run_multiseed():
    ta, fa, tf, ff = [], [], [], []          # tree/forest acc, tree/forest f1
    d_acc, d_f1 = [], []                     # diferencas pareadas por seed (floresta - arvore)
    for s in SEEDS:
        Xtr, Xte, ytr, yte = _make(s, N_EVAL)
        tree = DecisionTree(rng=np.random.default_rng(s)).fit(Xtr, ytr)   # arvore unica, todas features
        forest = RandomForest(seed=s).fit(Xtr, ytr)
        a_t, a_f = tree.score(Xte, yte), forest.score(Xte, yte)
        f_t = f1_bin(yte, tree.predict(Xte))
        f_f = f1_bin(yte, forest.predict(Xte))
        ta.append(a_t); fa.append(a_f); tf.append(f_t); ff.append(f_f)
        d_acc.append(a_f - a_t); d_f1.append(f_f - f_t)

    def ms(v):
        a = np.array(v)
        return float(a.mean()), float(a.std(ddof=1))

    tam, tas = ms(ta); fam, fas = ms(fa)
    tfm, tfs = ms(tf); ffm, ffs = ms(ff)
    dm, ds = ms(d_acc)
    se = ds / np.sqrt(len(d_acc)) if len(d_acc) > 1 else 0.0
    t_stat = dm / se if se > 0 else float("inf")
    var_reduction = (tas / fas) if fas > 0 else float("inf")

    lines = [
        "Multi-seed: arvore unica vs Random Forest (N=%d arvores) sobre %d seeds %s" % (N_TREES, N_SEEDS, SEEDS),
        "(cada seed = nova realizacao make_classification de %d amostras + split; teste ~%d)" % (N_EVAL, N_EVAL // 5),
        "",
        "%-14s %-22s %-22s" % ("Metrica", "Arvore unica", "Random Forest"),
        "-" * 60,
        "%-14s %10.4f +/-%6.4f %10.4f +/-%6.4f" % ("accuracy", tam, tas, fam, fas),
        "%-14s %10.4f +/-%6.4f %10.4f +/-%6.4f" % ("f1", tfm, tfs, ffm, ffs),
        "-" * 60,
        "",
        ">> REDUCAO DE VARIANCIA (o ponto do ensemble):",
        "   desvio da acuracia entre seeds: arvore %.4f  ->  floresta %.4f  (%.1fx menor)" % (tas, fas, var_reduction),
        "",
        ">> TESTE PAREADO (floresta - arvore, mesma realizacao por seed):",
        "   diferenca de acuracia = %+.4f +/- %.4f | SE %.4f | t = %+.2f" % (dm, ds, se, t_stat),
        "   (|t| > ~2 => a floresta e consistentemente melhor; nao e acaso de uma seed)",
    ]
    with open(os.path.join(OUTPUT_DIR, "report_multiseed.txt"), "w") as f:
        f.write("\n".join(lines))

    # -- plot: barras arvore vs floresta (acc, f1) com barra de erro = desvio --
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    groups = ["Acuracia", "F1"]
    tree_m = [tam, tfm]; tree_s = [tas, tfs]
    for_m = [fam, ffm]; for_s = [fas, ffs]
    x = np.arange(len(groups)); w = 0.36
    ax.bar(x - w / 2, tree_m, w, yerr=tree_s, capsize=7, label="Arvore unica",
           color="#f28e2b", edgecolor="k", linewidth=0.7, error_kw=dict(ecolor="#333", lw=1.3))
    ax.bar(x + w / 2, for_m, w, yerr=for_s, capsize=7, label="Random Forest (%d)" % N_TREES,
           color="#4e79a7", edgecolor="k", linewidth=0.7, error_kw=dict(ecolor="#333", lw=1.3))
    lo = min(min(m - s for m, s in zip(tree_m, tree_s)), min(m - s for m, s in zip(for_m, for_s)))
    ax.set_ylim(max(0.0, lo - 0.04), 1.0)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("Score (teste)")
    ax.set_title("Arvore unica vs Random Forest -- media +/- desvio (%d seeds)\n"
                 "floresta: barra de erro MENOR = variancia reduzida" % N_SEEDS)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    ms_path = os.path.join(OUTPUT_DIR, "metrics_multiseed.png")
    plt.savefig(ms_path, dpi=120, bbox_inches="tight")
    plt.close()
    return lines, ms_path, (tas, fas, dm, ds, t_stat)


# =============================================================================
# Figura 2D: particoes individuais vs combinada (requisito do enunciado)
# =============================================================================
def plot_partitions_2d():
    X, y = make_classification(n_samples=N_ILUSTRA, n_features=2, n_informative=2, n_redundant=0,
                               n_clusters_per_class=1, class_sep=0.9, random_state=SEED)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    forest = RandomForest(seed=SEED).fit(Xtr, ytr)
    single = DecisionTree(rng=np.random.default_rng(SEED)).fit(Xtr, ytr)

    cmap = mcolors.ListedColormap(["#a0c4e4", "#f7c59f"])
    pts = mcolors.ListedColormap(["#4e79a7", "#f28e2b"])
    x0, x1 = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y0, y1 = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x0, x1, 220), np.linspace(y0, y1, 220))
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6))
    # linha 1: 4 arvores individuais da floresta (particoes diversas por bagging)
    for j in range(4):
        ax = axes[0][j]
        zz = forest.trees[j * 3].predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, zz, alpha=0.5, cmap=cmap, levels=[-0.5, 0.5, 1.5])
        ax.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, cmap=pts, s=8, alpha=0.5, edgecolors="none")
        ax.set_title("Arvore #%d da floresta" % (j * 3 + 1), fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    # linha 2: arvore unica, floresta combinada, e uma nota
    labels = ["Arvore unica (P5)\n(instavel, recortada)", "Random Forest (%d arvores)\n(voto -> fronteira suave)" % N_TREES]
    models = [single, forest]
    for j in range(2):
        ax = axes[1][j]
        zz = models[j].predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, zz, alpha=0.5, cmap=cmap, levels=[-0.5, 0.5, 1.5])
        ax.scatter(Xte[:, 0], Xte[:, 1], c=yte, cmap=pts, s=14, edgecolors="k", linewidths=0.3)
        ax.set_title("%s  |  acc=%.3f" % (labels[j], models[j].score(Xte, yte)), fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    # probabilidade de voto da floresta (mapa continuo)
    ax = axes[1][2]
    votes = np.array([t.predict(grid) for t in forest.trees]).mean(0).reshape(xx.shape)
    cf = ax.contourf(xx, yy, votes, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.85)
    ax.scatter(Xte[:, 0], Xte[:, 1], c=yte, cmap=pts, s=12, edgecolors="k", linewidths=0.3)
    ax.set_title("Fracao de votos da floresta\n(incerteza na fronteira)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    axes[1][3].axis("off")
    axes[1][3].text(0.02, 0.5,
                    "Bagging (bootstrap) + features\naleatorias por split geram arvores\nDIVERSAS (linha de cima).\n\n"
                    "O voto de maioria (linha de baixo)\ncombina as particoes recortadas\nnuma fronteira mais suave e\nESTAVEL — reduz a variancia da\narvore unica sem aumentar o vies.",
                    fontsize=10, va="center")
    fig.suptitle("Projeto 6 — Random Forest: particoes individuais vs combinada (2D)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(OUTPUT_DIR, "forest_partitions.png")
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.close()
    return p


# =============================================================================
# Extra: variancia vs numero de arvores (a "variabilidade" do enunciado)
# =============================================================================
def plot_variance_vs_ntrees():
    ns = [1, 3, 5, 11, 21, 51, 101]
    stds = []
    for nt in ns:
        accs = []
        for s in SEEDS:
            Xtr, Xte, ytr, yte = _make(s, N_ILUSTRA)
            accs.append(RandomForest(n_trees=nt, seed=s).fit(Xtr, ytr).score(Xte, yte))
        stds.append(np.std(accs, ddof=1))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(ns, stds, "o-", color="#4e79a7")
    ax.set_xscale("log")
    ax.set_xlabel("nº de arvores na floresta")
    ax.set_ylabel("desvio da acuracia entre seeds")
    ax.set_title("Variabilidade cai com mais arvores (bagging reduz variancia)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "variance_vs_ntrees.png")
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.close()
    return p, list(zip(ns, stds))


def main():
    print("=== Projeto 6 — Random Forest: Ensemble e Variabilidade ===\n")
    lines, ms_path, (tas, fas, dm, ds, t_stat) = run_multiseed()
    print("\n".join(lines))
    print("Saved:", ms_path)
    p2d = plot_partitions_2d()
    print("Saved:", p2d)
    pv, sweep = plot_variance_vs_ntrees()
    print("Saved:", pv)
    print("\nVariancia da acuracia por nº de arvores:", [(n, round(s, 4)) for n, s in sweep])
    print("\nDone.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Random Forest manual — ensemble e variabilidade")
    ap.add_argument("-n", "--n-seeds", type=int, default=N_SEEDS,
                    help="numero de seeds para media +/- desvio (default: %(default)s)")
    ap.add_argument("-t", "--n-trees", type=int, default=N_TREES, help="arvores na floresta")
    a = ap.parse_args()
    N_SEEDS = a.n_seeds
    SEEDS = [SEED + i for i in range(N_SEEDS)]
    N_TREES = a.n_trees
    main()
