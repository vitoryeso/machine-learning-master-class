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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shared_problem as sp

SEED = 42
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
        self.activation, self.loss = activation, loss

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
        rng = np.random.default_rng(SEED)
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


def main():
    print("=== Projeto 9 — MLP manual (numpy) ===\n")

    # -- 1. Ablação principal: MLP (1 hidden, 64, relu, CE) full-dim nas 3 tarefas
    print("[1] Desempenho FULL-DIM (CLIP) por tarefa — MLP[D-64-1], relu, CE")
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
    main()
