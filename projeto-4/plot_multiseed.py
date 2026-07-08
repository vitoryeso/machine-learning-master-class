#!/usr/bin/env python3
"""Helper portatil de plot para o multi-seed do Projeto 4 (regressao logistica).

Le output/metrics_multiseed.csv (metric,mean,std — gerado pelo binario Rust) e
renderiza output/metrics_multiseed.png: barras das metricas de teste com barras
de erro = desvio amostral, PISO do eixo Y ADAPTATIVO (nao fixo em 0).

Por que existe: no Windows (xmain) o proprio Rust (crate `plotters`, backend
font-kit/DirectWrite) ja gera o PNG. Em maquinas headless sem `fontconfig`
(ex.: yalien) o backend de texto do plotters nao carrega, entao usamos este
helper matplotlib (numpy+matplotlib, sem pandas/torch) sobre o mesmo CSV.

Uso: python3 plot_multiseed.py   (rode a partir da pasta projeto-4/)
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
CSV_PATH = os.path.join(OUTPUT_DIR, "metrics_multiseed.csv")
PNG_PATH = os.path.join(OUTPUT_DIR, "metrics_multiseed.png")

# Metricas em escala [0,1] plotadas (BCE fica de fora — escala diferente).
PLOT_KEYS = [("accuracy", "Acuracia"), ("precision", "Precisao"),
             ("recall", "Recall"), ("f1", "F1")]


def load_csv(path):
    stats = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            stats[row["metric"]] = (float(row["mean"]), float(row["std"]))
    return stats


def main():
    if not os.path.exists(CSV_PATH):
        raise SystemExit("CSV nao encontrado: {} (rode o binario Rust primeiro)".format(CSV_PATH))
    stats = load_csv(CSV_PATH)
    labels = [lbl for _, lbl in PLOT_KEYS]
    means = [stats[k][0] for k, _ in PLOT_KEYS]
    stds = [stats[k][1] for k, _ in PLOT_KEYS]

    # Piso do eixo Y adaptativo: 3% abaixo do menor (media-desvio), sem descer de 0;
    # teto 3% acima do maior (media+desvio), sem passar de 1.
    lo = min(m - s for m, s in zip(means, stds))
    hi = max(m + s for m, s in zip(means, stds))
    y_min = max(0.0, lo - 0.03)
    y_max = min(1.0, hi + 0.03)
    if y_max <= y_min:
        y_max = min(1.0, y_min + 0.02)

    fig, ax = plt.subplots(figsize=(8, 5))
    xpos = range(len(labels))
    ax.bar(xpos, means, width=0.6, yerr=stds, capsize=8,
           color="#4e79a7", edgecolor="k", linewidth=0.8,
           error_kw=dict(ecolor="#222", lw=1.6))
    for x, m, s in zip(xpos, means, stds):
        ax.text(x, min(y_max, m + s + (y_max - y_min) * 0.02),
                "{:.3f}\n+/-{:.3f}".format(m, s),
                ha="center", va="bottom", fontsize=10)
    ax.set_xticks(list(xpos))
    ax.set_xticklabels(labels)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Score (teste)")
    n_seeds = "?"
    rep = os.path.join(OUTPUT_DIR, "report_multiseed.txt")
    if os.path.exists(rep):
        with open(rep) as f:
            first = f.readline()
            # "...sobre N seeds [..]"
            import re
            mtc = re.search(r"sobre (\d+) seeds", first)
            if mtc:
                n_seeds = mtc.group(1)
    ax.set_title("Metricas de Teste - media +/- desvio amostral ({} seeds)".format(n_seeds))
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved:", PNG_PATH)


if __name__ == "__main__":
    main()
