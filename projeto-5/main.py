#!/usr/bin/env python3
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.inspection import DecisionBoundaryDisplay

SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_redundant_splits(clf):
    """Return list of (parent_node_id, child_node_id, feature_idx, threshold, side)
    where a parent and one of its children (left OR right) split on the same
    feature with the same threshold. side is 'left' or 'right'.
    A 1e-10 tolerance catches true floating-point representation ties (e.g., the same
    threshold stored with rounding noise). It does NOT catch the visually ambiguous
    f0<=0.13 pair in this tree, where the two thresholds genuinely differ by 0.006
    (0.13318 vs 0.12690) and only appear identical because export_text rounds to 2 decimal places.
    Note: only checks direct parent->child pairs (depth+1). Grandchild or deeper
    redundant splits are not detected; absence of matches here does not rule out
    logically unreachable splits at deeper levels.
    """
    tree = clf.tree_
    redundant = []
    for node_id in range(tree.node_count):
        if tree.children_left[node_id] == -1:
            continue  # leaf
        # Check left child
        left = tree.children_left[node_id]
        if tree.children_left[left] != -1:  # left child is not a leaf
            if (tree.feature[node_id] == tree.feature[left] and
                    abs(tree.threshold[node_id] - tree.threshold[left]) < 1e-10):
                redundant.append((node_id, left, tree.feature[node_id], tree.threshold[node_id], 'left'))
        # Check right child
        right = tree.children_right[node_id]
        if tree.children_left[right] != -1:  # right child is not a leaf
            if (tree.feature[node_id] == tree.feature[right] and
                    abs(tree.threshold[node_id] - tree.threshold[right]) < 1e-10):
                redundant.append((node_id, right, tree.feature[node_id], tree.threshold[node_id], 'right'))
    return redundant


def get_leaf_sample_counts(clf):
    """Return sorted array of n_node_samples values for leaf nodes only."""
    tree = clf.tree_
    leaf_mask = tree.children_left == -1
    return np.sort(tree.n_node_samples[leaf_mask])


def main():
    # -- 1. Dataset ----------------------------------------------------------------
    X, y = make_classification(
        n_samples=500,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=0.9,
        random_state=SEED,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    print("Dataset: {} samples, {} features, 2 classes".format(X.shape[0], X.shape[1]))
    print("Train: {}  Test: {}".format(X_train.shape[0], X_test.shape[0]))
    print()

    # -- 2. Train trees ------------------------------------------------------------
    criteria = ["gini", "entropy"]
    trees = {}
    metrics = {}

    for crit in criteria:
        clf = DecisionTreeClassifier(criterion=crit, random_state=SEED)
        clf.fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred)
        trees[crit] = clf
        metrics[crit] = {
            "train_accuracy": train_acc,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "depth": clf.get_depth(),
            "n_leaves": clf.get_n_leaves(),
            "n_nodes": clf.tree_.node_count,
            "feature_importances": clf.feature_importances_.tolist(),
            "confusion_matrix": cm.tolist(),
        }

    # -- 3. Print metrics table ----------------------------------------------------
    header = "{:<22} {:>10} {:>10}".format("Metric", "Gini", "Entropy")
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for key in ["train_accuracy", "accuracy", "precision", "recall", "f1", "depth", "n_leaves", "n_nodes"]:
        g = metrics["gini"][key]
        e = metrics["entropy"][key]
        if isinstance(g, float):
            print("{:<22} {:>10.4f} {:>10.4f}".format(key, g, e))
        else:
            print("{:<22} {:>10} {:>10}".format(key, g, e))
    print(sep)
    print()

    for crit in criteria:
        fi = metrics[crit]["feature_importances"]
        print("Feature importance ({}): f0={:.4f}, f1={:.4f}".format(crit, fi[0], fi[1]))
    print()

    # -- 3b. Confusion matrices ----------------------------------------------------
    for crit in criteria:
        cm = metrics[crit]["confusion_matrix"]
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        print("Confusion matrix ({}): TN={} FP={} FN={} TP={}".format(crit, tn, fp, fn, tp))
    print()

    # -- 3c. Predictions comparison ------------------------------------------------
    y_pred_gini = trees["gini"].predict(X_test)
    y_pred_entropy = trees["entropy"].predict(X_test)
    preds_identical = bool(np.array_equal(y_pred_gini, y_pred_entropy))
    n_preds_differ = int(np.sum(y_pred_gini != y_pred_entropy))
    print("Predictions identical (gini vs entropy):", preds_identical)
    print("Samples where predictions differ:", n_preds_differ)
    print()

    # -- 3d. Redundant split detection ---------------------------------------------
    for crit in criteria:
        redundant = find_redundant_splits(trees[crit])
        if redundant:
            print("WARNING: Redundant splits in {} tree:".format(crit))
            for parent_id, child_id, feat_idx, thr, side in redundant:
                print("  Parent node {} and {}-child node {} both split on f{} <= {:.6f}".format(
                    parent_id, side, child_id, feat_idx, thr))
            print("  (Parent constraint makes child split on same feature/threshold unreachable.)")
        else:
            print("No redundant splits found in {} tree (direct parent-child pairs only; deeper levels not checked).".format(crit))
    # Diagnostic (SEED=42 only): print actual thresholds for display-ambiguous pairs
    # in both trees across all features — pairs where two distinct thresholds round to
    # the same 2-decimal string in export_text, making them look identical in the text output.
    # Generalized over all features and both criteria so it works for any seed/dataset.
    if SEED == 42:
        ambiguous_pairs = {}  # {(crit, feat_idx): [(a, b), ...]}
        feature_names = ["f{}".format(i) for i in range(X.shape[1])]
        for crit in criteria:
            clf_tree = trees[crit].tree_
            for feat_idx in range(X.shape[1]):
                internal_nodes = [
                    n for n in range(clf_tree.node_count)
                    if clf_tree.children_left[n] != -1 and clf_tree.feature[n] == feat_idx
                ]
                thresholds = sorted([clf_tree.threshold[n] for n in internal_nodes])
                pairs = []
                for i in range(len(thresholds) - 1):
                    a, b = thresholds[i], thresholds[i + 1]
                    if "{:.2f}".format(a) == "{:.2f}".format(b):
                        pairs.append((a, b))
                if pairs:
                    ambiguous_pairs[(crit, feat_idx)] = pairs
        if ambiguous_pairs:
            for (crit, feat_idx), pairs in sorted(ambiguous_pairs.items()):
                fname = feature_names[feat_idx]
                print("{} tree {} display-ambiguous threshold pairs (both show same 2-decimal value in export_text):".format(crit, fname))
                for a, b in pairs:
                    print("  {} thresholds: {:.5f} and {:.5f}  (both display as {} <= {:.2f})  diff={:.5f}".format(
                        fname, a, b, fname, round(a, 2), b - a))
        # collect ambiguous_pairs for gini/f0 for legacy use in report.txt
        ambiguous_pairs_gini_f0 = ambiguous_pairs.get(("gini", 0), [])
    else:
        ambiguous_pairs_gini_f0 = []
    print()

    # -- 3e. Leaf sample counts (verifies overfitting / memorization claims) -------
    leaf_sample_lines = []
    for crit in criteria:
        leaf_counts = get_leaf_sample_counts(trees[crit])
        n_single = int(np.sum(leaf_counts == 1))
        leaf_min = int(leaf_counts.min())
        leaf_max = int(leaf_counts.max())
        leaf_mean = float(leaf_counts.mean())
        line = (
            "Leaf sample counts ({}): n_leaves={}, min={}, max={}, mean={:.1f}, "
            "leaves_with_1_sample={}".format(
                crit, len(leaf_counts), leaf_min, leaf_max, leaf_mean, n_single
            )
        )
        print(line)
        leaf_sample_lines.append(line)
    print()

    # -- 4. Decision boundary plot -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#4e79a7", "#f28e2b"]
    cmap = mcolors.ListedColormap(colors)

    for ax, crit in zip(axes, criteria):
        clf = trees[crit]
        DecisionBoundaryDisplay.from_estimator(
            clf, X, response_method="predict",
            alpha=0.3, ax=ax, cmap=cmap,
        )
        ax.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train,
            cmap=cmap, marker="x", s=40, alpha=0.6, linewidths=1.0,
        )
        ax.scatter(
            X_test[:, 0], X_test[:, 1], c=y_test,
            cmap=cmap, edgecolors="k", s=40, linewidths=0.6,
        )
        acc = metrics[crit]["accuracy"]
        depth = metrics[crit]["depth"]
        leaves = metrics[crit]["n_leaves"]
        ax.set_title(
            "criterion={}\nAcc={:.4f}  depth={}  leaves={}".format(
                repr(crit), acc, depth, leaves
            ),
            fontsize=11,
        )
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")

    class_patches = [
        mpatches.Patch(color=c, label="Class {}".format(i))
        for i, c in enumerate(colors)
    ]
    train_handle = mlines.Line2D(
        [], [], marker="x", color="gray", linestyle="None",
        markersize=7, label="train",
    )
    test_handle = mlines.Line2D(
        [], [], marker="o", color="gray", linestyle="None",
        markersize=7, markeredgecolor="k", label="test",
    )
    fig.legend(
        handles=class_patches + [train_handle, test_handle],
        loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Decision Tree: Gini vs Entropy -- 2D Decision Boundaries",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    boundary_path = os.path.join(OUTPUT_DIR, "boundary_comparison.png")
    plt.savefig(boundary_path, dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: {}".format(boundary_path))

    # -- 5. Feature importance comparison -----------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    width = 0.35
    bars_g = ax.bar(
        x - width / 2, metrics["gini"]["feature_importances"], width,
        label="Gini", color="#4e79a7", edgecolor="k", linewidth=0.7,
    )
    bars_e = ax.bar(
        x + width / 2, metrics["entropy"]["feature_importances"], width,
        label="Entropy", color="#f28e2b", edgecolor="k", linewidth=0.7,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["Feature 0", "Feature 1"])
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance")
    g_acc = metrics["gini"]["accuracy"]
    g_depth = metrics["gini"]["depth"]
    e_depth = metrics["entropy"]["depth"]
    e_acc = metrics["entropy"]["accuracy"]
    ax.set_title(
        "Feature Importance: Gini vs Entropy\n(Acc Gini={:.1f}% / Entropy={:.1f}%, depth Gini={} / Entropy={})".format(
            g_acc * 100, e_acc * 100, g_depth, e_depth
        )
    )
    ax.legend()
    max_importance = max(
        max(metrics["gini"]["feature_importances"]),
        max(metrics["entropy"]["feature_importances"]),
    )
    ax.set_ylim(0, max_importance * 1.15)
    for bar in list(bars_g) + list(bars_e):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.01,
            "{:.3f}".format(h),
            ha="center", va="bottom", fontsize=9,
        )
    plt.tight_layout()
    importance_path = os.path.join(OUTPUT_DIR, "importance.png")
    plt.savefig(importance_path, dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: {}".format(importance_path))

    # -- 5b. Depth-vs-accuracy sweep ------------------------------------------
    # Extend range 3 levels beyond the deeper tree so the plateau is visible;
    # this also shows well beyond the shallower tree (e.g., 8 levels beyond Gini=9
    # when Entropy=14 is the deeper one).
    max_natural_depth = max(metrics["gini"]["depth"], metrics["entropy"]["depth"])
    depth_range = list(range(1, max_natural_depth + 4))
    sweep_acc = {"gini": [], "entropy": []}
    sweep_train_acc = {"gini": [], "entropy": []}
    for max_d in depth_range:
        for crit in criteria:
            clf_d = DecisionTreeClassifier(criterion=crit, max_depth=max_d, random_state=SEED)
            clf_d.fit(X_train, y_train)
            sweep_acc[crit].append(clf_d.score(X_test, y_test))
            sweep_train_acc[crit].append(clf_d.score(X_train, y_train))

    # Print sweep table so per-depth accuracy claims are verifiable from text output
    sweep_header = "{:>10} {:>15} {:>16} {:>16} {:>18}".format("max_depth", "gini_test_acc", "gini_train_acc", "entropy_test_acc", "entropy_train_acc")
    sweep_sep = "-" * len(sweep_header)
    print(sweep_sep)
    print("Depth-vs-accuracy sweep:")
    print(sweep_header)
    print(sweep_sep)
    for i, max_d in enumerate(depth_range):
        print("{:>10} {:>15.4f} {:>16.4f} {:>16.4f} {:>18.4f}".format(
            max_d, sweep_acc["gini"][i], sweep_train_acc["gini"][i],
            sweep_acc["entropy"][i], sweep_train_acc["entropy"][i]
        ))
    print(sweep_sep)
    print()

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.plot(depth_range, [v * 100 for v in sweep_acc["gini"]], "o-",
             color="#4e79a7", label="Gini test")
    ax2.plot(depth_range, [v * 100 for v in sweep_train_acc["gini"]], "o:",
             color="#4e79a7", alpha=0.5, label="Gini train")
    ax2.plot(depth_range, [v * 100 for v in sweep_acc["entropy"]], "s--",
             color="#f28e2b", label="Entropy test")
    ax2.plot(depth_range, [v * 100 for v in sweep_train_acc["entropy"]], "s:",
             color="#f28e2b", alpha=0.5, label="Entropy train")
    ax2.axvline(x=metrics["gini"]["depth"], color="#4e79a7", linestyle=":",
                alpha=0.7, label="Gini unrestricted depth ({})".format(metrics["gini"]["depth"]))
    ax2.axvline(x=metrics["entropy"]["depth"], color="#f28e2b", linestyle=":",
                alpha=0.7, label="Entropy unrestricted depth ({})".format(metrics["entropy"]["depth"]))
    ax2.set_xlabel("max_depth")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Depth vs Accuracy: Train vs Test (Gini vs Entropy)")
    ax2.legend(fontsize=8)
    ax2.set_xticks(depth_range)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    depth_sweep_path = os.path.join(OUTPUT_DIR, "depth_vs_accuracy.png")
    plt.savefig(depth_sweep_path, dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: {}".format(depth_sweep_path))

    # -- 5c. Confusion matrix heatmap -----------------------------------------
    fig_cm, axes_cm = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax_cm, crit in zip(axes_cm, criteria):
        cm_arr = np.array(metrics[crit]["confusion_matrix"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_arr, display_labels=["Class 0", "Class 1"])
        disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
        disp.ax_.set_title("criterion={}".format(repr(crit)), fontsize=11)
    fig_cm.suptitle("Confusion Matrix: Gini vs Entropy (test set, n=100)", fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: {}".format(cm_path))

    # -- 6. Tree structure text ----------------------------------------------------
    tree_texts = {}
    for crit in criteria:
        # max_depth=clf.get_depth() ensures the full tree is printed;
        # the default max_depth=10 would silently truncate trees deeper than 10 levels.
        tree_texts[crit] = export_text(trees[crit], feature_names=["f0", "f1"], max_depth=trees[crit].get_depth())
        print("Tree structure ({}): ".format(crit))
        print(tree_texts[crit])

    # -- 7. Save report.txt -------------------------------------------------------
    report_lines = [
        "=" * 60,
        "PROJETO 5 -- ARVORE DE DECISAO: Gini vs Entropia",
        "=" * 60,
        "",
        "Dataset: make_classification, n_samples=500, n_features=2, n_classes=2, random_state=42",
        "Train: {}  |  Test: {}".format(X_train.shape[0], X_test.shape[0]),
        "",
        "Metrics Comparison:",
        header,
        sep,
    ]
    for key in ["train_accuracy", "accuracy", "precision", "recall", "f1", "depth", "n_leaves", "n_nodes"]:
        g = metrics["gini"][key]
        e = metrics["entropy"][key]
        if isinstance(g, float):
            report_lines.append("{:<22} {:>10.4f} {:>10.4f}".format(key, g, e))
        else:
            report_lines.append("{:<22} {:>10} {:>10}".format(key, g, e))
    report_lines.append(sep)
    report_lines.append("")
    for crit in criteria:
        fi = metrics[crit]["feature_importances"]
        report_lines.append(
            "Feature importance ({}): f0={:.4f}, f1={:.4f}".format(crit, fi[0], fi[1])
        )
    report_lines.append("")
    report_lines.append("Confusion Matrices:")
    for crit in criteria:
        cm = metrics[crit]["confusion_matrix"]
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        report_lines.append("  {} : TN={} FP={} FN={} TP={}".format(crit, tn, fp, fn, tp))
    report_lines.append("")
    report_lines.append("Predictions identical (gini vs entropy): {}".format(preds_identical))
    report_lines.append("Samples where predictions differ: {}".format(n_preds_differ))
    report_lines.append("")
    report_lines.append("Redundant split analysis:")
    for crit in criteria:
        redundant = find_redundant_splits(trees[crit])
        if redundant:
            for parent_id, child_id, feat_idx, thr, side in redundant:
                report_lines.append(
                    "  WARNING ({}): node {} and {}-child {} both split on f{} <= {:.6f}".format(
                        crit, parent_id, side, child_id, feat_idx, thr
                    )
                )
        else:
            report_lines.append("  {} : no redundant splits (direct parent-child pairs only; deeper levels not checked)".format(crit))
    if ambiguous_pairs_gini_f0:
        report_lines.append("  Gini f0 display-ambiguous pairs (distinct thresholds shown as same 2-decimal value by export_text):")
        for a, b in ambiguous_pairs_gini_f0:
            report_lines.append("    {:.5f} and {:.5f} (both display as f0 <= {:.2f})  diff={:.5f}".format(
                a, b, round(a, 2), b - a))
    report_lines.append("")
    report_lines.append("Leaf sample counts (verifies overfitting claims):")
    for line in leaf_sample_lines:
        report_lines.append("  " + line)
    report_lines.append("")
    report_lines.append("Depth-vs-accuracy sweep:")
    report_lines.append(sweep_header)
    report_lines.append(sweep_sep)
    for i, max_d in enumerate(depth_range):
        report_lines.append("{:>10} {:>15.4f} {:>16.4f} {:>16.4f} {:>18.4f}".format(
            max_d, sweep_acc["gini"][i], sweep_train_acc["gini"][i],
            sweep_acc["entropy"][i], sweep_train_acc["entropy"][i]))
    report_lines.append(sweep_sep)
    report_lines.append("")
    report_lines.append("Tree structure (Gini):")
    report_lines.append(tree_texts["gini"])
    report_lines.append("Tree structure (Entropy):")
    report_lines.append(tree_texts["entropy"])

    report_path = os.path.join(OUTPUT_DIR, "report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print("Saved: {}".format(report_path))
    print()
    print("Done.")


if __name__ == "__main__":
    main()