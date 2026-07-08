#!/usr/bin/env python3
"""Projeto 5 — Arvore de Decisao construida MANUALMENTE (sem sklearn.tree).

O enunciado pede: "Construa manualmente uma arvore de decisao ... calculando e
comparando explicitamente os criterios de divisao Gini e Entropia."

Por isso a arvore (criterio de impureza, busca de split, recursao, predicao,
importancia de feature) e implementada do zero abaixo. O sklearn e usado apenas
para gerar o dataset sintetico (make_classification) e o split treino/teste —
utilitarios de dados, nao o modelo.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

SEED = 42
N_SEEDS = 3  # rodadas para media +/- desvio; suba p/ 10/30 p/ std mais firme
SEEDS = [SEED + i for i in range(N_SEEDS)]  # [42, 43, 44]
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Criterios de impureza  —  o coracao do projeto (Gini vs Entropia)
# =============================================================================
def _class_probs(y):
    """Proporcao p_k de cada classe presente no vetor de rotulos y.

    Retorna um np.array com as proporcoes (somam 1). Helper para as funcoes de
    impureza abaixo. Ex.: y=[0,0,1] -> array([0.667, 0.333]).
    """
    if len(y) == 0:
        return np.array([])
    _, counts = np.unique(y, return_counts=True)
    return counts / counts.sum()


def gini(y):
    """Impureza de Gini do no:  G = 1 - sum(p_k^2).

    Use _class_probs(y). No puro -> 0.0 ; binario uniforme (p=[0.5,0.5]) -> 0.5.
    """
    p = _class_probs(y)
    return float(1.0 - np.sum(p ** 2))


def entropy(y):
    """Entropia de Shannon do no:  H = -sum(p_k * log2(p_k)).

    Use _class_probs(y). Cuidado com p_k=0 (convencao 0*log2(0)=0). _class_probs
    so retorna classes presentes, entao nao havera p_k=0 — mas e bom saber.
    No puro -> 0.0 ; binario uniforme -> 1.0.
    """
    p = _class_probs(y)
    return float(-np.sum(p * np.log2(p)))


def impurity(y, criterion):
    return gini(y) if criterion == "gini" else entropy(y)


# =============================================================================
# Arvore de decisao (CART binario) — construida manualmente
# =============================================================================
class Node:
    __slots__ = ("feature", "threshold", "left", "right", "value",
                 "n_samples", "impurity", "is_leaf")

    def __init__(self):
        self.feature = None       # indice da feature usada no split
        self.threshold = None     # limiar do split (x[feature] <= threshold -> esquerda)
        self.left = None
        self.right = None
        self.value = None         # classe majoritaria (predicao da folha)
        self.n_samples = 0
        self.impurity = 0.0
        self.is_leaf = False


class DecisionTree:
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.n_features_ = X.shape[1]
        self.n_total_ = len(y)
        self.importances_ = np.zeros(self.n_features_)
        self.root = self._build(X, y, depth=0)
        total = self.importances_.sum()
        if total > 0:
            self.importances_ = self.importances_ / total
        return self

    def _build(self, X, y, depth):
        node = Node()
        node.n_samples = len(y)
        node.impurity = impurity(y, self.criterion)
        # classe majoritaria (desempate pela menor label)
        classes, counts = np.unique(y, return_counts=True)
        node.value = int(classes[np.argmax(counts)])

        # condicoes de parada: no puro, profundidade maxima, poucas amostras
        if (node.impurity == 0.0
                or (self.max_depth is not None and depth >= self.max_depth)
                or len(y) < self.min_samples_split):
            node.is_leaf = True
            return node

        best = self._best_split(X, y, node.impurity)
        if best is None:
            node.is_leaf = True
            return node

        feat, thr, gain, left_mask = best
        # importancia = fracao de amostras no no * reducao de impureza
        self.importances_[feat] += (len(y) / self.n_total_) * gain
        node.feature, node.threshold = feat, thr
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _best_split(self, X, y, parent_impurity):
        """Busca gulosa: a cada no, escolhe (feature, threshold) que maximiza a
        reducao de impureza. Thresholds candidatos = pontos medios entre valores
        consecutivos distintos da feature."""
        n = len(y)
        best_gain = 0.0
        best = None
        for feat in range(self.n_features_):
            values = np.unique(X[:, feat])
            if len(values) < 2:
                continue
            thresholds = (values[:-1] + values[1:]) / 2.0
            for thr in thresholds:
                left_mask = X[:, feat] <= thr
                n_left = int(left_mask.sum())
                n_right = n - n_left
                if n_left == 0 or n_right == 0:
                    continue
                imp_left = impurity(y[left_mask], self.criterion)
                imp_right = impurity(y[~left_mask], self.criterion)
                child_impurity = (n_left / n) * imp_left + (n_right / n) * imp_right
                gain = parent_impurity - child_impurity
                if gain > best_gain:
                    best_gain = gain
                    best = (feat, float(thr), gain, left_mask)
        return best

    # -- predicao -------------------------------------------------------------
    def _predict_one(self, x):
        node = self.root
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x) for x in X])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    # -- introspeccao da estrutura -------------------------------------------
    def get_depth(self):
        def d(node):
            return 0 if node.is_leaf else 1 + max(d(node.left), d(node.right))
        return d(self.root)

    def get_n_leaves(self):
        def c(node):
            return 1 if node.is_leaf else c(node.left) + c(node.right)
        return c(self.root)

    def get_n_nodes(self):
        def c(node):
            return 1 if node.is_leaf else 1 + c(node.left) + c(node.right)
        return c(self.root)

    def export_text(self, feature_names=None):
        names = feature_names or ["f{}".format(i) for i in range(self.n_features_)]
        lines = []

        def walk(node, depth):
            pad = "|   " * depth + "|--- "
            if node.is_leaf:
                lines.append("{}class: {}".format(pad, node.value))
                return
            lines.append("{}{} <= {:.5f}".format(pad, names[node.feature], node.threshold))
            walk(node.left, depth + 1)
            lines.append("{}{} >  {:.5f}".format(pad, names[node.feature], node.threshold))
            walk(node.right, depth + 1)

        walk(self.root, 0)
        return "\n".join(lines)


# =============================================================================
# Analises de estrutura da arvore (adaptadas a estrutura de Node manual)
# =============================================================================
def find_redundant_splits(tree, tol=1e-9):
    """Pares pai->filho que dividem na MESMA feature com threshold ~igual.

    Um filho que re-divide na mesma feature/threshold do pai e logicamente
    inalcancavel (a restricao do pai ja determina o lado). Retorna lista de
    (feature, threshold, lado). So checa pares diretos pai-filho.
    """
    redundant = []

    def walk(node):
        if node.is_leaf:
            return
        for side, child in (("left", node.left), ("right", node.right)):
            if (not child.is_leaf
                    and child.feature == node.feature
                    and abs(child.threshold - node.threshold) < tol):
                redundant.append((node.feature, node.threshold, side))
            walk(child)

    walk(tree.root)
    return redundant


def leaf_sample_counts(tree):
    """Array ordenado de n_samples por folha (verifica memorizacao/overfitting)."""
    counts = []

    def walk(node):
        if node.is_leaf:
            counts.append(node.n_samples)
        else:
            walk(node.left)
            walk(node.right)

    walk(tree.root)
    return np.sort(counts)


# =============================================================================
# Metricas (calculadas a mao a partir da matriz de confusao)
# =============================================================================
def confusion(y_true, y_pred):
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])


def prf1(cm):
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


# =============================================================================
# Multi-seed — robustez estatistica (media +/- desvio sobre N_SEEDS rodadas)
# =============================================================================
def _make_data(seed):
    """Gera o dataset e o split para uma seed. Cada seed = uma realizacao nova
    do problema (make_classification) + um split estratificado novo."""
    X, y = make_classification(
        n_samples=500, n_features=2, n_informative=2, n_redundant=0,
        n_clusters_per_class=1, class_sep=0.9, random_state=seed,
    )
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


def _train_eval(crit, X_tr, y_tr, X_te, y_te):
    """Treina uma arvore com o criterio dado e devolve as metricas de teste."""
    clf = DecisionTree(criterion=crit).fit(X_tr, y_tr)
    cm = confusion(y_te, clf.predict(X_te))
    prec, rec, f1 = prf1(cm)
    return {"accuracy": clf.score(X_te, y_te), "train_accuracy": clf.score(X_tr, y_tr),
            "precision": prec, "recall": rec, "f1": f1,
            "depth": clf.get_depth(), "n_leaves": clf.get_n_leaves()}


MS_KEYS = ["accuracy", "f1", "precision", "recall", "depth", "n_leaves", "train_accuracy"]


def run_multiseed():
    """Roda gini e entropy sobre N_SEEDS realizacoes e agrega media +/- desvio.
    Escreve report_multiseed.txt + metrics_multiseed.png. Aditivo: nao mexe na
    analise single-seed (seed=42) do main(). Retorna (stats, report_lines)."""
    agg = {c: {k: [] for k in MS_KEYS} for c in ("gini", "entropy")}
    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te = _make_data(seed)
        for crit in ("gini", "entropy"):
            m = _train_eval(crit, X_tr, y_tr, X_te, y_te)
            for k in MS_KEYS:
                agg[crit][k].append(m[k])
    # media e desvio amostral (ddof=1); com 1 seed cai p/ 0
    stats = {c: {k: (float(np.mean(v)),
                     float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
                 for k, v in agg[c].items()} for c in agg}

    lines = ["Multi-seed: media +/- desvio (ddof=1) sobre {} seeds {}".format(N_SEEDS, SEEDS),
             "(cada seed = novo make_classification + novo split estratificado)", ""]
    hdr = "{:<16} {:>22} {:>22}".format("Metric", "Gini", "Entropy")
    lines += [hdr, "-" * len(hdr)]
    for k in MS_KEYS:
        gm, gs = stats["gini"][k]
        em, es = stats["entropy"][k]
        if k in ("depth", "n_leaves"):
            lines.append("{:<16} {:>15.1f} +/-{:>5.1f} {:>15.1f} +/-{:>5.1f}".format(k, gm, gs, em, es))
        else:
            lines.append("{:<16} {:>14.4f} +/-{:>5.4f} {:>14.4f} +/-{:>5.4f}".format(k, gm, gs, em, es))
    lines.append("-" * len(hdr))

    # plot com barras de erro (std) — acuracia e F1
    metrics_to_plot = [("accuracy", "Acuracia"), ("f1", "F1")]
    colors = {"gini": "#4e79a7", "entropy": "#f28e2b"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xpos = np.arange(len(metrics_to_plot))
    width = 0.35
    for i, crit in enumerate(("gini", "entropy")):
        means = [stats[crit][k][0] for k, _ in metrics_to_plot]
        stds = [stats[crit][k][1] for k, _ in metrics_to_plot]
        ax.bar(xpos + (i - 0.5) * width, means, width, yerr=stds, capsize=6,
               label=crit.capitalize(), color=colors[crit], edgecolor="k",
               linewidth=0.7, error_kw=dict(ecolor="#333", lw=1.3))
    ax.set_xticks(xpos)
    ax.set_xticklabels([lbl for _, lbl in metrics_to_plot])
    ax.set_ylabel("Score (teste)")
    ax.set_ylim(0.90, 1.02)
    ax.set_title("Gini vs Entropia -- media +/- desvio ({} seeds)".format(N_SEEDS))
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    ms_path = os.path.join(OUTPUT_DIR, "metrics_multiseed.png")
    plt.savefig(ms_path, dpi=120, bbox_inches="tight")
    plt.close()

    with open(os.path.join(OUTPUT_DIR, "report_multiseed.txt"), "w") as f:
        f.write("\n".join(lines))
    return stats, lines, ms_path


def main():
    # -- 0. Multi-seed: robustez estatistica (media +/- desvio) ---------------
    _, ms_report, ms_path = run_multiseed()
    print("\n".join(ms_report))
    print("Saved:", ms_path)
    print()

    # -- 1. Dataset (seed=42, referencia para as analises qualitativas) -------
    X, y = make_classification(
        n_samples=500, n_features=2, n_informative=2, n_redundant=0,
        n_clusters_per_class=1, class_sep=0.9, random_state=SEED,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print("Dataset: {} samples, {} features, 2 classes".format(X.shape[0], X.shape[1]))
    print("Train: {}  Test: {}".format(X_train.shape[0], X_test.shape[0]))
    print()

    # -- 2. Treina as duas arvores (manuais) ----------------------------------
    criteria = ["gini", "entropy"]
    trees, metrics = {}, {}
    for crit in criteria:
        clf = DecisionTree(criterion=crit).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        cm = confusion(y_test, y_pred)
        prec, rec, f1 = prf1(cm)
        trees[crit] = clf
        metrics[crit] = {
            "train_accuracy": clf.score(X_train, y_train),
            "accuracy": clf.score(X_test, y_test),
            "precision": prec, "recall": rec, "f1": f1,
            "depth": clf.get_depth(), "n_leaves": clf.get_n_leaves(),
            "n_nodes": clf.get_n_nodes(),
            "feature_importances": clf.importances_.tolist(),
            "confusion_matrix": cm.tolist(),
        }

    # -- 3. Tabela de metricas (seed=42) --------------------------------------
    # ILUSTRACAO de 1 realizacao (a que gera as figuras abaixo). O resultado
    # OFICIAL e o multi-seed impresso no topo; esta tabela e so o caso da seed=42.
    header = "{:<22} {:>10} {:>10}".format("Metric (seed=42)", "Gini", "Entropy")
    sep = "-" * len(header)
    print("[ilustracao seed=42 — numeros oficiais = multi-seed acima]")
    print(sep); print(header); print(sep)
    for key in ["train_accuracy", "accuracy", "precision", "recall", "f1",
                "depth", "n_leaves", "n_nodes"]:
        g, e = metrics["gini"][key], metrics["entropy"][key]
        if isinstance(g, float):
            print("{:<22} {:>10.4f} {:>10.4f}".format(key, g, e))
        else:
            print("{:<22} {:>10} {:>10}".format(key, g, e))
    print(sep); print()

    for crit in criteria:
        fi = metrics[crit]["feature_importances"]
        print("Feature importance ({}): f0={:.4f}, f1={:.4f}".format(crit, fi[0], fi[1]))
    print()
    for crit in criteria:
        cm = metrics[crit]["confusion_matrix"]
        print("Confusion matrix ({}): TN={} FP={} FN={} TP={}".format(
            crit, cm[0][0], cm[0][1], cm[1][0], cm[1][1]))
    print()

    y_pred_gini = trees["gini"].predict(X_test)
    y_pred_entropy = trees["entropy"].predict(X_test)
    n_preds_differ = int(np.sum(y_pred_gini != y_pred_entropy))
    print("Predictions identical (gini vs entropy):", n_preds_differ == 0)
    print("Samples where predictions differ:", n_preds_differ)
    print()

    # -- 3b. Analises de estrutura (splits redundantes + folhas) --------------
    redundant_lines = []
    for crit in criteria:
        redundant = find_redundant_splits(trees[crit])
        if redundant:
            redundant_lines.append("Redundant splits ({}):".format(crit))
            for feat, thr, side in redundant:
                redundant_lines.append(
                    "  no e seu filho-{} ambos dividem em f{} <= {:.5f} (inalcancavel)".format(
                        side, feat, thr))
        else:
            redundant_lines.append(
                "{}: nenhum split redundante (pares pai-filho diretos)".format(crit))
    for line in redundant_lines:
        print(line)
    print()

    leaf_lines = []
    for crit in criteria:
        lc = leaf_sample_counts(trees[crit])
        leaf_lines.append(
            "Leaf sample counts ({}): n_leaves={}, min={}, max={}, mean={:.1f}, "
            "leaves_with_1_sample={}".format(
                crit, len(lc), int(lc.min()), int(lc.max()), float(lc.mean()),
                int(np.sum(lc == 1))))
    for line in leaf_lines:
        print(line)
    print()

    # -- 4. Fronteiras de decisao (predicao manual sobre um meshgrid) ---------
    colors = ["#4e79a7", "#f28e2b"]
    cmap = mcolors.ListedColormap(colors)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, crit in zip(axes, criteria):
        zz = trees[crit].predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, zz, alpha=0.3, cmap=cmap, levels=[-0.5, 0.5, 1.5])
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap,
                   marker="x", s=40, alpha=0.6, linewidths=1.0)
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap,
                   edgecolors="k", s=40, linewidths=0.6)
        m = metrics[crit]
        ax.set_title("criterion='{}'\nAcc={:.4f}  depth={}  leaves={}".format(
            crit, m["accuracy"], m["depth"], m["n_leaves"]), fontsize=11)
        ax.set_xlabel("Feature 0"); ax.set_ylabel("Feature 1")
    class_patches = [mpatches.Patch(color=c, label="Class {}".format(i))
                     for i, c in enumerate(colors)]
    train_handle = mlines.Line2D([], [], marker="x", color="gray", linestyle="None",
                                 markersize=7, label="train")
    test_handle = mlines.Line2D([], [], marker="o", color="gray", linestyle="None",
                                markersize=7, markeredgecolor="k", label="test")
    fig.legend(handles=class_patches + [train_handle, test_handle],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Arvore de Decisao MANUAL: Gini vs Entropy -- Fronteiras 2D", fontsize=13)
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    boundary_path = os.path.join(OUTPUT_DIR, "boundary_comparison.png")
    plt.savefig(boundary_path, dpi=120, bbox_inches="tight"); plt.close()
    print("Saved:", boundary_path)

    # -- 5. Importancia das features ------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    xpos = np.arange(2); width = 0.35
    bars_g = ax.bar(xpos - width / 2, metrics["gini"]["feature_importances"], width,
                    label="Gini", color="#4e79a7", edgecolor="k", linewidth=0.7)
    bars_e = ax.bar(xpos + width / 2, metrics["entropy"]["feature_importances"], width,
                    label="Entropy", color="#f28e2b", edgecolor="k", linewidth=0.7)
    ax.set_xticks(xpos); ax.set_xticklabels(["Feature 0", "Feature 1"])
    ax.set_xlabel("Feature"); ax.set_ylabel("Importance")
    ax.set_title("Importancia das Features: Gini vs Entropy")
    ax.legend()
    for bar in list(bars_g) + list(bars_e):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, "{:.3f}".format(h),
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    importance_path = os.path.join(OUTPUT_DIR, "importance.png")
    plt.savefig(importance_path, dpi=120, bbox_inches="tight"); plt.close()
    print("Saved:", importance_path)

    # -- 6. Sweep profundidade vs acuracia ------------------------------------
    max_natural = max(metrics["gini"]["depth"], metrics["entropy"]["depth"])
    depth_range = list(range(1, max_natural + 4))
    sweep_test = {"gini": [], "entropy": []}
    sweep_train = {"gini": [], "entropy": []}
    for d in depth_range:
        for crit in criteria:
            clf_d = DecisionTree(criterion=crit, max_depth=d).fit(X_train, y_train)
            sweep_test[crit].append(clf_d.score(X_test, y_test))
            sweep_train[crit].append(clf_d.score(X_train, y_train))

    sweep_header = "{:>10} {:>15} {:>16} {:>16} {:>18}".format(
        "max_depth", "gini_test_acc", "gini_train_acc", "entropy_test_acc", "entropy_train_acc")
    sweep_sep = "-" * len(sweep_header)
    print(sweep_sep); print("Depth-vs-accuracy sweep:"); print(sweep_header); print(sweep_sep)
    for i, d in enumerate(depth_range):
        print("{:>10} {:>15.4f} {:>16.4f} {:>16.4f} {:>18.4f}".format(
            d, sweep_test["gini"][i], sweep_train["gini"][i],
            sweep_test["entropy"][i], sweep_train["entropy"][i]))
    print(sweep_sep); print()

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.plot(depth_range, [v * 100 for v in sweep_test["gini"]], "o-", color="#4e79a7", label="Gini test")
    ax2.plot(depth_range, [v * 100 for v in sweep_train["gini"]], "o:", color="#4e79a7", alpha=0.5, label="Gini train")
    ax2.plot(depth_range, [v * 100 for v in sweep_test["entropy"]], "s--", color="#f28e2b", label="Entropy test")
    ax2.plot(depth_range, [v * 100 for v in sweep_train["entropy"]], "s:", color="#f28e2b", alpha=0.5, label="Entropy train")
    ax2.axvline(x=metrics["gini"]["depth"], color="#4e79a7", linestyle=":", alpha=0.7,
                label="Gini depth ({})".format(metrics["gini"]["depth"]))
    ax2.axvline(x=metrics["entropy"]["depth"], color="#f28e2b", linestyle=":", alpha=0.7,
                label="Entropy depth ({})".format(metrics["entropy"]["depth"]))
    ax2.set_xlabel("max_depth"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Profundidade vs Acuracia (Gini vs Entropy)")
    ax2.legend(fontsize=8); ax2.set_xticks(depth_range); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    depth_path = os.path.join(OUTPUT_DIR, "depth_vs_accuracy.png")
    plt.savefig(depth_path, dpi=120, bbox_inches="tight"); plt.close()
    print("Saved:", depth_path)

    # -- 7. Matriz de confusao (heatmap manual) -------------------------------
    fig_cm, axes_cm = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax_cm, crit in zip(axes_cm, criteria):
        cm = np.array(metrics[crit]["confusion_matrix"])
        ax_cm.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax_cm.text(j, i, str(v), ha="center", va="center",
                       color="white" if v > cm.max() / 2 else "black", fontsize=12)
        ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["Pred 0", "Pred 1"]); ax_cm.set_yticklabels(["Real 0", "Real 1"])
        ax_cm.set_title("criterion='{}'".format(crit), fontsize=11)
    fig_cm.suptitle("Matriz de Confusao: Gini vs Entropy (teste, n=100)", fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=120, bbox_inches="tight"); plt.close()
    print("Saved:", cm_path)

    # -- 8. Estrutura das arvores em texto ------------------------------------
    tree_texts = {c: trees[c].export_text(["f0", "f1"]) for c in criteria}

    # -- 9. report.txt --------------------------------------------------------
    report = [
        "=" * 60,
        "PROJETO 5 -- ARVORE DE DECISAO MANUAL: Gini vs Entropia",
        "=" * 60, "",
        "Dataset: make_classification, n_samples=500, n_features=2",
        "Implementacao: arvore CART construida do zero (ver main.py).", "",
        "RESULTADO OFICIAL (multi-seed):",
    ] + ms_report + [
        "",
        "-" * 60,
        "Ilustracao (seed=42, 1 realizacao) — base das figuras fronteira/arvore/sweep:",
        "Train: {}  |  Test: {}".format(X_train.shape[0], X_test.shape[0]),
        "Metrics (seed=42):", header, sep,
    ]
    for key in ["train_accuracy", "accuracy", "precision", "recall", "f1",
                "depth", "n_leaves", "n_nodes"]:
        g, e = metrics["gini"][key], metrics["entropy"][key]
        if isinstance(g, float):
            report.append("{:<22} {:>10.4f} {:>10.4f}".format(key, g, e))
        else:
            report.append("{:<22} {:>10} {:>10}".format(key, g, e))
    report += [sep, ""]
    for crit in criteria:
        fi = metrics[crit]["feature_importances"]
        report.append("Feature importance ({}): f0={:.4f}, f1={:.4f}".format(crit, fi[0], fi[1]))
    report.append("")
    report.append("Confusion Matrices:")
    for crit in criteria:
        cm = metrics[crit]["confusion_matrix"]
        report.append("  {} : TN={} FP={} FN={} TP={}".format(
            crit, cm[0][0], cm[0][1], cm[1][0], cm[1][1]))
    report += ["", "Samples where predictions differ (gini vs entropy): {}".format(n_preds_differ), ""]
    report.append("Redundant split analysis (pares pai-filho diretos):")
    report += ["  " + l for l in redundant_lines]
    report.append("")
    report.append("Leaf sample counts (verifica overfitting/memorizacao):")
    report += ["  " + l for l in leaf_lines]
    report.append("  Nota: export_text usa 5 casas decimais — sem o artefato de arredondamento")
    report.append("  de threshold que afeta o export_text do sklearn (que arredonda p/ 2 casas).")
    report.append("")
    report.append("Depth-vs-accuracy sweep:"); report.append(sweep_header); report.append(sweep_sep)
    for i, d in enumerate(depth_range):
        report.append("{:>10} {:>15.4f} {:>16.4f} {:>16.4f} {:>18.4f}".format(
            d, sweep_test["gini"][i], sweep_train["gini"][i],
            sweep_test["entropy"][i], sweep_train["entropy"][i]))
    report += [sweep_sep, "", "Tree structure (Gini):", tree_texts["gini"],
               "", "Tree structure (Entropy):", tree_texts["entropy"]]
    report_path = os.path.join(OUTPUT_DIR, "report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print("Saved:", report_path)
    print("\nDone.")


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description="Arvore Gini vs Entropia — multi-seed")
    _ap.add_argument("-n", "--n-seeds", type=int, default=N_SEEDS,
                     help="numero de seeds para media +/- desvio (default: %(default)s)")
    _args = _ap.parse_args()
    N_SEEDS = _args.n_seeds
    SEEDS = [SEED + i for i in range(N_SEEDS)]
    main()
