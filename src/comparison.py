"""
Day 6 helpers: load every per-model metrics JSON + predictions npz and produce
the head-to-head comparison table, the side-by-side confusion-matrix figure,
and the per-class F1 chart.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CLASS_NAMES, FIGURES_DIR, PREDICTIONS_DIR
from src.io_utils import write_report
from src.plot_utils import save_fig


# Canonical model order in every comparison output (best-known-first by plan).
DEFAULT_MODELS = ["baseline_cnn", "mobilenetv2", "vgg16"]


def load_all_metrics(models: list[str] = None,
                     prefer_clean: bool = False) -> dict[str, dict]:
    """Load each model's metrics JSON.

    If ``prefer_clean=True`` (used by the head-to-head comparison figures),
    the loader prefers ``<model>_clean_metrics.json`` over
    ``<model>_metrics.json`` whenever both exist. The clean variant is what
    the three pre-EfficientNet baselines have after Phase B
    (``src/reeval_clean.py`` writes them); EfficientNet models were trained
    and evaluated on the clean split directly, so they only have the
    plain metrics file.
    """
    models = models or DEFAULT_MODELS
    out = {}
    for name in models:
        clean = PREDICTIONS_DIR / f"{name}_clean_metrics.json"
        plain = PREDICTIONS_DIR / f"{name}_metrics.json"
        if prefer_clean and clean.exists():
            path = clean
        elif plain.exists():
            path = plain
        elif clean.exists():
            path = clean
        else:
            continue
        with open(path, "r", encoding="utf-8") as f:
            out[name] = json.load(f)
    return out


def comparison_table(metrics: dict[str, dict]) -> pd.DataFrame:
    nan = float("nan")
    rows = []
    for name, m in metrics.items():
        rows.append({
            "model": name,
            "accuracy": round(m.get("accuracy", nan), 4),
            "macro_precision": round(m.get("macro_precision", nan), 4),
            "macro_recall": round(m.get("macro_recall", nan), 4),
            "macro_f1": round(m.get("macro_f1", nan), 4),
            "weighted_f1": round(m.get("weighted_f1", nan), 4),
            "epochs_run": m.get("training", {}).get("epochs_run"),
            "elapsed_seconds": m.get("training", {}).get("elapsed_seconds"),
            "n_test": m.get("n_test"),
        })
    return pd.DataFrame(rows)


def per_class_f1_table(metrics: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, m in metrics.items():
        for cls, f1 in zip(m["per_class"]["labels"], m["per_class"]["f1"]):
            rows.append({"model": name, "class": cls, "f1": round(f1, 4)})
    df = pd.DataFrame(rows).pivot(index="class", columns="model", values="f1")
    # Preserve the order in which models were loaded — alphabetical pivot order
    # is fragile if a future model is added.
    return df.reindex(columns=[m for m in metrics.keys() if m in df.columns])


def plot_confusion_matrices_side_by_side(metrics: dict[str, dict],
                                         normalize: bool = True,
                                         save_name: str = "06_confusion_matrices_side_by_side") -> Path:
    """Grid of one confusion matrix per model.

    Lays out the panels in a wrap-around grid (max 4 columns) instead of a
    single row, so each panel keeps a readable size when the model list grows.
    Class labels use a short code (NonD / VMild / Mild / Mod) along the axes
    so they fit at any panel size.
    """
    n = len(metrics)
    n_cols = min(n, 4)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.0 * n_cols, 4.4 * n_rows + 0.3),
                             squeeze=False,
                             gridspec_kw={"hspace": 0.55, "wspace": 0.30})
    short_labels = {
        "NonDemented": "NonD",
        "VeryMildDemented": "VMild",
        "MildDemented": "Mild",
        "ModerateDemented": "Mod",
    }
    short = [short_labels.get(c, c) for c in CLASS_NAMES]

    items = list(metrics.items())
    for idx in range(n_rows * n_cols):
        ax = axes[idx // n_cols, idx % n_cols]
        if idx >= n:
            ax.axis("off")
            continue
        name, m = items[idx]
        cm = np.asarray(m["confusion_matrix"], dtype=float)
        if normalize:
            cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else cm.max())
        ax.set_title(f"{name}\nacc={m['accuracy']:.3f}  macroF1={m['macro_f1']:.3f}",
                     fontsize=11)
        ax.set_xticks(range(len(short)))
        ax.set_yticks(range(len(short)))
        ax.set_xticklabels(short, rotation=0, fontsize=10)
        ax.set_yticklabels(short, fontsize=10)
        ax.set_xlabel("predicted", fontsize=10)
        if idx % n_cols == 0:
            ax.set_ylabel("true", fontsize=10)
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                v = cm[i, j]
                txt = f"{v:.2f}" if normalize else f"{int(v)}"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if v > thresh else "black", fontsize=10)
    # Use constrained-style padding rather than tight_layout — gridspec_kw
    # already pins the inter-row spacing, and tight_layout can override it.
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.06, right=0.98,
                        hspace=0.55, wspace=0.30)
    out = save_fig(fig, save_name)
    plt.close(fig)
    return out


def _bar_palette(n: int) -> list[str]:
    """Distinct, colorblind-friendlier palette that scales to ~10 models.

    Hand-picked from Tableau 10 + Set2 pastels so each adjacent pair has
    visibly different hue *and* lightness. Matters when we have 6-7 models
    in one chart and the default Seaborn deep palette starts to blend.
    """
    palette = [
        "#4C72B0",  # blue
        "#DD8452",  # orange
        "#55A868",  # green
        "#C44E52",  # red
        "#8172B2",  # purple
        "#937860",  # brown
        "#DA8BC3",  # pink
        "#8C8C8C",  # gray
    ]
    return palette[:n]


def _annotate_bars(ax, bars, vals, *, fmt: str = "{:.3f}",
                   rotate_when: float = 0.06):
    """Place numeric labels above bars, rotating them when the bars get
    so narrow that horizontal labels would collide."""
    bar_w = bars[0].get_width() if len(bars) else 0.1
    rotate = bar_w < rotate_when
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003,
                fmt.format(v),
                ha="center", va="bottom",
                fontsize=8 if not rotate else 7,
                rotation=90 if rotate else 0,
                rotation_mode="anchor")


def _grouped_bars(ax, models: list[str], group_x, vals_by_model: dict[str, list[float]],
                   *, width_total: float = 0.85, ymin: float = 0.5, ymax: float = 1.05):
    """Render a grouped bar chart: one cluster per group_x, one bar per model.

    Returns the per-model `BarContainer`s for caller annotation."""
    n = len(models)
    width = width_total / max(n, 1)
    palette = _bar_palette(n)
    bars_per_model = {}
    for i, model in enumerate(models):
        offset = (i - (n - 1) / 2) * width
        bars_per_model[model] = ax.bar(group_x + offset, vals_by_model[model],
                                        width=width, label=model,
                                        color=palette[i],
                                        edgecolor="white", linewidth=0.5)
    ax.set_ylim(ymin, ymax)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    return bars_per_model, width


def plot_per_class_f1_bars(metrics: dict[str, dict],
                            save_name: str = "06_per_class_f1") -> Path:
    """Grouped bar chart: one bar per (class, model) of per-class F1."""
    df = per_class_f1_table(metrics)
    classes = list(df.index)
    models = list(df.columns)
    x = np.arange(len(classes))

    fig, ax = plt.subplots(figsize=(13, 5.5))
    vals_by_model = {m: df[m].values.tolist() for m in models}
    bars_per_model, _ = _grouped_bars(ax, models, x, vals_by_model,
                                       ymin=0.0, ymax=1.10)
    for m in models:
        _annotate_bars(ax, bars_per_model[m], vals_by_model[m])
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1 — head-to-head", fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=min(len(models), 4), frameon=False, fontsize=9)
    fig.tight_layout()
    out = save_fig(fig, save_name)
    plt.close(fig)
    return out


def plot_headline_metrics_bars(metrics: dict[str, dict],
                                save_name: str = "06_metrics_grouped_bars") -> Path:
    """Grouped bar chart over the four headline metrics.

    Y-axis starts at 0.5 (not 0) so the small differences near 1.0 stay
    visible even when all models score above 0.7.
    """
    metric_names = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    pretty = {"accuracy": "Accuracy", "macro_precision": "Precision (macro)",
              "macro_recall": "Recall (macro)", "macro_f1": "F1 (macro)"}
    models = list(metrics.keys())
    x = np.arange(len(metric_names))

    fig, ax = plt.subplots(figsize=(13, 5.5))
    vals_by_model = {m: [metrics[m][k] for k in metric_names] for m in models}
    bars_per_model, _ = _grouped_bars(ax, models, x, vals_by_model,
                                       ymin=0.5, ymax=1.06)
    for m in models:
        _annotate_bars(ax, bars_per_model[m], vals_by_model[m])
    ax.set_xticks(x)
    ax.set_xticklabels([pretty[k] for k in metric_names], fontsize=10)
    ax.set_title("Headline metrics — head-to-head (y-axis starts at 0.5)",
                 fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=min(len(models), 4), frameon=False, fontsize=9)
    fig.tight_layout()
    out = save_fig(fig, save_name)
    plt.close(fig)
    return out


def biggest_confusion(metrics_one: dict) -> str:
    """Return a sentence describing the largest off-diagonal in a single model's CM."""
    cm = np.asarray(metrics_one["confusion_matrix"], dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    np.fill_diagonal(norm, 0.0)
    if norm.max() == 0:
        return "no off-diagonal confusions in this run."
    i, j = np.unravel_index(int(norm.argmax()), norm.shape)
    return (f"confuses **{CLASS_NAMES[i]} -> {CLASS_NAMES[j]}** "
            f"{norm[i, j] * 100:.1f}% of {CLASS_NAMES[i]} samples.")


def write_comparison_report(metrics: dict[str, dict]) -> Path:
    """Build a Markdown comparison report and save to outputs/reports/."""
    table = comparison_table(metrics).to_markdown(index=False)
    per_class = per_class_f1_table(metrics).round(4).to_markdown()

    best_acc = max(metrics, key=lambda k: metrics[k]["accuracy"])
    best_f1 = max(metrics, key=lambda k: metrics[k]["macro_f1"])

    confusion_lines = []
    for name, m in metrics.items():
        confusion_lines.append(f"* **{name}** — {biggest_confusion(m)}")
    confusion_block = "\n".join(confusion_lines)

    body = (
        f"# Day 6 — Model Comparison\n\n"
        f"## Headline metrics\n\n{table}\n\n"
        f"Best test accuracy: **{best_acc}** ({metrics[best_acc]['accuracy']:.4f}).\n\n"
        f"Best macro F1:     **{best_f1}** ({metrics[best_f1]['macro_f1']:.4f}).\n\n"
        f"## Per-class F1\n\n{per_class}\n\n"
        f"## Where each model is most confused\n\nLargest off-diagonal in each normalized confusion matrix:\n\n{confusion_block}\n\n"
        f"## Notes\n\n"
        f"* Accuracy alone is misleading on imbalanced data — macro F1 is reported alongside.\n"
        f"* The plan flagged VeryMildDemented vs MildDemented as the hardest pair — verify in the per-class F1 table above.\n"
        f"* All numbers are on an *augmented & upsampled* dataset with image-level split — they are upper bounds on a clinical evaluation.\n"
    )
    return write_report("06_model_comparison", body)
