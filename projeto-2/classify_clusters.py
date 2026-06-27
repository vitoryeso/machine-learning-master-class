#!/usr/bin/env python3
"""Projeto 2 — Classificacao da taxonomia descoberta por clustering hierarquico.

Pipeline conceitual (fecha o ciclo projeto-1 -> projeto-2):
  1. K-means hierarquico (CLIP, ja rodado) descobriu uma taxonomia de 2 niveis:
     5 macro-grupos (G00..G04) e 25 subclasses-folha.  -> hierarchy.json
  2. Usamos esses rotulos como ALVO de classificacao e treinamos um linear probe
     para atribuir imagens novas a taxonomia (split treino/teste).

Teste metodologico (anti-circularidade):
  - Classificar sobre os MESMOS embeddings CLIP que geraram os clusters da
    acuracia alta, porem nao perfeita: os clusters sao aproximadamente celulas
    de Voronoi nesse espaco, mas o StandardScaler aqui recentraliza/reescala
    por dimensao e nao preserva exatamente a geometria (esfera unitaria ~
    cosseno) em que o k-means rodou — por isso o probe nao recupera 100%.
  - Ainda assim e pouco informativo; o teste honesto classifica sobre features
    ConvNeXt (modelo independente). Rodamos AMBOS para evidenciar a diferenca.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
)

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dataset")
OUT = os.path.join(HERE, "output", "classification")
os.makedirs(OUT, exist_ok=True)


def build_labels(hierarchy):
    """Constroi y_macro (0..4) e y_leaf (0..24) a partir do hierarchy.json.
    Retorna tambem leaf_names e leaf_to_macro (mapa folha->macro)."""
    n = hierarchy["dataset_size"]
    y_macro = np.full(n, -1, dtype=int)
    y_leaf = np.full(n, -1, dtype=int)
    leaf_names, leaf_to_macro = [], []
    leaf_id = 0
    for c in hierarchy["clusters"]:
        gl = c["global_label"]
        for idx in c["image_indices"]:
            y_macro[idx] = gl
        for s in c["subclusters"]:
            for idx in s["image_indices"]:
                y_leaf[idx] = leaf_id
            leaf_names.append(s.get("id", "{}_S{:02d}".format(c["id"], s.get("sub_label", 0))))
            leaf_to_macro.append(gl)
            leaf_id += 1
    assert (y_macro >= 0).all(), "imagem sem macro-label"
    assert (y_leaf >= 0).all(), "imagem sem leaf-label"
    return y_macro, y_leaf, leaf_names, np.array(leaf_to_macro)


def evaluate(X, y_leaf, leaf_to_macro, itr, ite, name):
    """Treina linear probe (logistic regression) sobre X para prever a folha;
    deriva o macro via mapa folha->macro. Retorna dict de metricas + predicoes."""
    scaler = StandardScaler().fit(X[itr])
    Xtr, Xte = scaler.transform(X[itr]), scaler.transform(X[ite])
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    clf.fit(Xtr, y_leaf[itr])
    pred_leaf = clf.predict(Xte)

    true_leaf = y_leaf[ite]
    true_macro = leaf_to_macro[true_leaf]
    pred_macro = leaf_to_macro[pred_leaf]

    def m(yt, yp):
        p, r, f, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
        return {"accuracy": accuracy_score(yt, yp), "precision": p, "recall": r, "f1": f}

    res = {"feature": name, "leaf": m(true_leaf, pred_leaf), "macro": m(true_macro, pred_macro)}
    print("[{}] leaf acc={:.4f} f1={:.4f}  |  macro acc={:.4f} f1={:.4f}".format(
        name, res["leaf"]["accuracy"], res["leaf"]["f1"],
        res["macro"]["accuracy"], res["macro"]["f1"]))
    return res, pred_leaf, true_leaf, pred_macro, true_macro


def plot_confusion(cm, labels, title, path, normalize=True):
    if normalize:
        cm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.5), max(4, len(labels) * 0.45)))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predito"); ax.set_ylabel("Real"); ax.set_title(title)
    if len(labels) <= 8:
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, "{:.2f}".format(v), ha="center", va="center",
                    color="white" if v > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    print("Saved:", path)


def main():
    hierarchy = json.load(open(os.path.join(HERE, "output", "hierarchical", "hierarchy.json")))
    y_macro, y_leaf, leaf_names, leaf_to_macro = build_labels(hierarchy)
    # macro_names indexado por global_label (nao por posicao na lista), para que
    # a matriz de confusao macro e a tabela de tamanhos casem com y_macro mesmo
    # se os clusters vierem fora de ordem no hierarchy.json.
    macro_names = [None] * len(hierarchy["clusters"])
    for c in hierarchy["clusters"]:
        macro_names[c["global_label"]] = c["id"]
    assert all(nm is not None for nm in macro_names), "global_label nao cobre 0..k-1"
    n = len(y_leaf)
    print("N={}, macro classes={}, leaf classes={}".format(n, len(macro_names), len(leaf_names)))

    Xc = np.load(os.path.join(DATA, "X_convnext.npy"))
    Xe = np.load(os.path.join(DATA, "X_embeddings.npy"))
    print("ConvNeXt:", Xc.shape, "| CLIP:", Xe.shape)
    # As linhas de X_*.npy precisam casar 1:1 com os image_indices do hierarchy.json.
    # ATENCAO: estes mesmos nomes de arquivo sao usados pela Parte I (8000 imgs);
    # se forem regenerados por ela, os indices (0..n-1) nao batem mais com as features.
    assert Xc.shape[0] == n and Xe.shape[0] == n, (
        "features desalinhadas com hierarchy.json: esperado {} linhas, "
        "ConvNeXt={} CLIP={}".format(n, Xc.shape[0], Xe.shape[0]))

    # mesmo split para comparacao justa entre os dois espacos de features
    idx = np.arange(n)
    itr, ite = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y_leaf)
    print("Train: {}  Test: {}".format(len(itr), len(ite)))
    print()

    # CLIP = controle (circular); ConvNeXt = teste honesto
    res_clip, *_ = evaluate(Xe, y_leaf, leaf_to_macro, itr, ite, "CLIP (circular)")
    res_cnx, pred_leaf, true_leaf, pred_macro, true_macro = evaluate(
        Xc, y_leaf, leaf_to_macro, itr, ite, "ConvNeXt (honesto)")
    print()

    # -- plots (ConvNeXt = o resultado principal, honesto) --------------------
    cm_macro = confusion_matrix(true_macro, pred_macro, labels=range(len(macro_names)))
    plot_confusion(cm_macro, macro_names,
                   "Confusao MACRO (ConvNeXt) — 5 classes",
                   os.path.join(OUT, "confusion_macro_convnext.png"))
    cm_leaf = confusion_matrix(true_leaf, pred_leaf, labels=range(len(leaf_names)))
    plot_confusion(cm_leaf, leaf_names,
                   "Confusao FOLHA (ConvNeXt) — 25 classes",
                   os.path.join(OUT, "confusion_leaf_convnext.png"))

    # barra comparativa CLIP vs ConvNeXt (evidencia a circularidade)
    fig, ax = plt.subplots(figsize=(7, 4))
    groups = ["macro\n(5 classes)", "folha\n(25 classes)"]
    xp = np.arange(2); w = 0.35
    clip_acc = [res_clip["macro"]["accuracy"], res_clip["leaf"]["accuracy"]]
    cnx_acc = [res_cnx["macro"]["accuracy"], res_cnx["leaf"]["accuracy"]]
    b1 = ax.bar(xp - w / 2, [a * 100 for a in clip_acc], w, label="CLIP (circular)", color="#bbb")
    b2 = ax.bar(xp + w / 2, [a * 100 for a in cnx_acc], w, label="ConvNeXt (honesto)", color="#4e79a7")
    ax.set_xticks(xp); ax.set_xticklabels(groups)
    ax.set_ylabel("Acuracia (%)"); ax.set_ylim(0, 105)
    ax.set_title("Classificacao da taxonomia descoberta: CLIP vs ConvNeXt")
    ax.legend()
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                    "{:.1f}".format(b.get_height()), ha="center", fontsize=9)
    plt.tight_layout()
    cmp_path = os.path.join(OUT, "accuracy_comparison.png")
    plt.savefig(cmp_path, dpi=120, bbox_inches="tight"); plt.close()
    print("Saved:", cmp_path)

    # -- report ---------------------------------------------------------------
    leaf_sizes = np.bincount(y_leaf, minlength=len(leaf_names))
    macro_sizes = np.bincount(y_macro, minlength=len(macro_names))
    lines = [
        "=" * 64,
        "PROJETO 2 -- CLASSIFICACAO DA TAXONOMIA DESCOBERTA (2 niveis)",
        "=" * 64, "",
        "Dataset: {} imagens | macro classes: {} | leaf classes: {}".format(
            n, len(macro_names), len(leaf_names)),
        "Split: treino {} / teste {} (estratificado por folha, seed=42)".format(len(itr), len(ite)),
        "Alvo: rotulos do K-means hierarquico sobre CLIP (hierarchy.json).",
        "Classificador: linear probe (LogisticRegression, features padronizadas).", "",
        "RESULTADOS (acuracia / precisao / recall / F1 macro-avg):",
        "{:<22} {:>10} {:>10} {:>10} {:>10}".format("", "acc", "prec", "rec", "f1"),
        "-" * 64,
    ]
    for res in (res_clip, res_cnx):
        for lvl in ("macro", "leaf"):
            m = res[lvl]
            lines.append("{:<22} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
                "{} / {}".format(res["feature"], lvl),
                m["accuracy"], m["precision"], m["recall"], m["f1"]))
    lines += ["-" * 64, "",
              "Interpretacao: CLIP classifica seus proprios clusters com acuracia",
              "alta, porem nao perfeita (circular; o gap ate 100% vem do mismatch",
              "entre o StandardScaler e a geometria normalizada do k-means).",
              "ConvNeXt, um espaco independente, mede a real separabilidade da",
              "taxonomia descoberta.", "",
              "Tamanho das macro-classes:"]
    for nm, sz in zip(macro_names, macro_sizes):
        lines.append("  {}: {}".format(nm, sz))
    lines.append("")
    lines.append("Tamanho das leaf-classes:")
    for nm, sz in zip(leaf_names, leaf_sizes):
        lines.append("  {}: {}".format(nm, sz))

    report_path = os.path.join(OUT, "report_classification.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print("Saved:", report_path)

    with open(os.path.join(OUT, "classification_metrics.json"), "w") as f:
        json.dump({"clip": res_clip, "convnext": res_cnx,
                   "n": n, "n_macro": len(macro_names), "n_leaf": len(leaf_names),
                   "leaf_names": leaf_names}, f, indent=2)
    print("\nDone.")


if __name__ == "__main__":
    main()
