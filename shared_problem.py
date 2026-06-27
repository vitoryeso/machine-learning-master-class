#!/usr/bin/env python3
"""Harness compartilhado dos projetos 7/8/9 (e contexto p/ 10).

Problema REAL (não sintético): classificação binária sobre a coleção pessoal de
imagens, usando as features já extraídas no projeto-2 (CLIP 512-d / ConvNeXt 768-d,
22.328 imagens, alinhadas índice-a-índice). Três tarefas binárias compõem o eixo
"problema" da ablação:

  - has_people          : pessoa vs sem pessoa            (y_people)
  - screenshot_vs_photo : screenshot vs foto de câmera    (subconjunto de y_type)
  - macro_vs_rest       : maior macro-grupo vs resto       (taxonomia do projeto-2)

Para atender a exigência de visualização no plano 2D (enunciados 7/8/9), as
features são projetadas para 2D via PCA (mesma técnica do projeto-1). O PCA é
ajustado SÓ no treino para não vazar o teste.
"""
import os
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(REPO, "projeto-2", "dataset")
HIER = os.path.join(REPO, "projeto-2", "output", "hierarchical", "hierarchy.json")
SEED = 42

TASKS = ["has_people", "screenshot_vs_photo", "macro_vs_rest"]


def load_features(which="clip"):
    """which: 'clip' (512-d) ou 'convnext' (768-d). Retorna (N, D)."""
    fname = "X_embeddings.npy" if which == "clip" else "X_convnext.npy"
    return np.load(os.path.join(DATA, fname))


def get_task(name):
    """Retorna (mask, y, pos_name, neg_name).
    mask: índices booleanos das imagens usadas na tarefa (subset p/ screenshot_vs_photo).
    y: rótulo binário 0/1 já restrito ao mask."""
    if name == "has_people":
        yp = np.load(os.path.join(DATA, "y_people.npy"))
        y = (yp > 0.5).astype(int)
        mask = np.ones(len(y), dtype=bool)
        return mask, y[mask], "pessoa", "sem pessoa"

    if name == "screenshot_vs_photo":
        yt = np.load(os.path.join(DATA, "y_type.npy"))
        names = json.load(open(os.path.join(DATA, "class_names.json")))
        idx = {n: i for i, n in enumerate(names)}
        shot = [idx[n] for n in ("mobile_screenshot", "screenshot_desktop") if n in idx]
        photo = idx["camera_photo"]
        mask = np.isin(yt, shot + [photo])
        y = np.isin(yt[mask], shot).astype(int)  # 1 = screenshot, 0 = foto
        return mask, y, "screenshot", "foto"

    if name == "macro_vs_rest":
        h = json.load(open(HIER))
        n = h["dataset_size"]
        sizes = [(c["global_label"], len(c["image_indices"])) for c in h["clusters"]]
        target = max(sizes, key=lambda t: t[1])[0]  # maior macro-grupo
        y = np.zeros(n, dtype=int)
        for c in h["clusters"]:
            if c["global_label"] == target:
                for i in c["image_indices"]:
                    y[i] = 1
        mask = np.ones(n, dtype=bool)
        return mask, y[mask], "G{:02d}".format(target), "resto"

    raise ValueError("tarefa desconhecida: " + name)


def make_2d(X_train, X_test):
    """Padroniza (fit no treino) e projeta para 2D via PCA (fit no treino).
    Retorna (Z_train, Z_test, var_explicada)."""
    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
    pca = PCA(n_components=2, random_state=SEED).fit(Xtr)
    return pca.transform(Xtr), pca.transform(Xte), float(pca.explained_variance_ratio_.sum())


def prepare(task, features="clip", test_size=0.2):
    """Pipeline completo: carrega features+labels da tarefa, split estratificado.
    Retorna full-dim padronizado (X_train/X_test, para MÉTRICAS reais) e a
    projeção 2D PCA (Z_train/Z_test, apenas ILUSTRAÇÃO da fronteira)."""
    mask, y, pos, neg = get_task(task)
    X = load_features(features)[mask]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y)
    # full-dim padronizado -> desempenho real (todas as tarefas são aprendíveis aqui)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    # 2D PCA -> só para ilustrar a fronteira no plano (retém pouca variância)
    Ztr, Zte, var = make_2d(Xtr, Xte)
    return {
        "X_train": Xtr_s, "X_test": Xte_s,   # full-dim padronizado (reportar)
        "Z_train": Ztr, "Z_test": Zte,        # 2D PCA (ilustrar)
        "y_train": ytr, "y_test": yte,
        "task": task, "features": features, "pos": pos, "neg": neg,
        "pca_var": var, "n": int(mask.sum()),
        "balance": [int((y == 0).sum()), int((y == 1).sum())],
    }


def evaluate(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p,
            "recall": r, "f1": f, "confusion": confusion_matrix(y_true, y_pred).tolist()}


def plot_boundary(ax, predict_fn, Z, y, title, extra=None):
    """Desenha a fronteira de decisão 2D de um modelo (predict_fn: (M,2)->{0,1}).
    extra: callable(ax) opcional para sobrepor centros/SVs/etc."""
    import matplotlib.colors as mcolors
    cmap = mcolors.ListedColormap(["#4e79a7", "#f28e2b"])
    pad = 0.5
    x0, x1 = Z[:, 0].min() - pad, Z[:, 0].max() + pad
    y0, y1 = Z[:, 1].min() - pad, Z[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300))
    zz = predict_fn(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.3, cmap=cmap, levels=[-0.5, 0.5, 1.5])
    ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap=cmap, s=8, alpha=0.5, linewidths=0)
    if extra:
        extra(ax)
    # fixa os limites na região dos dados (linhas de neurônios/SVs não devem
    # estourar o autoscale e espremer a nuvem de pontos)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")


if __name__ == "__main__":
    # smoke test: imprime balanço e variância PCA de cada tarefa
    for feat in ("clip", "convnext"):
        print("=== features:", feat, "===")
        for t in TASKS:
            d = prepare(t, features=feat)
            print("  {:<20} n={:<6} balance(neg,pos)={} pca_var2d={:.3f}  ({} vs {})".format(
                t, d["n"], d["balance"], d["pca_var"], d["pos"], d["neg"]))
