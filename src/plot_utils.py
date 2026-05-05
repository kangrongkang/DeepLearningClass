"""
Plotting helpers shared across notebooks for consistent figure styling.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import CLASS_NAMES, FIGURES_DIR


def save_fig(fig, name: str) -> Path:
    out = FIGURES_DIR / f"{name}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    return out


def plot_class_counts(counts: dict, title: str = "Images per class") -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    classes = list(counts.keys())
    values = [counts[c] for c in classes]
    bars = ax.bar(classes, values,
                  color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"],
                  edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(values) * 0.012,
                f"{v:,}", ha="center", va="bottom", fontsize=11)
    total = sum(values)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v / 2,
                f"{100 * v / total:.1f}%", ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, max(values) * 1.10)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()
    out = save_fig(fig, "class_counts")
    plt.close(fig)
    return out


def plot_history(history: dict, title_prefix: str, save_name: str) -> Path:
    """Plot accuracy / loss curves for a single training run.

    The panels share an x-axis (epoch). Title is on the figure (suptitle)
    so the per-axis space is used for the curves; legends are outside the
    plot area so they never cover the data.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    epochs = list(range(1, len(history.get("loss", [])) + 1))
    if "accuracy" in history:
        axes[0].plot(epochs, history["accuracy"], label="train", lw=2)
    if "val_accuracy" in history:
        axes[0].plot(epochs, history["val_accuracy"], label="val", lw=2)
    axes[0].set_xlabel("epoch", fontsize=11)
    axes[0].set_ylabel("accuracy", fontsize=11)
    axes[0].set_title("Accuracy", fontsize=12)

    if "loss" in history:
        axes[1].plot(epochs, history["loss"], label="train", lw=2)
    if "val_loss" in history:
        axes[1].plot(epochs, history["val_loss"], label="val", lw=2)
    axes[1].set_xlabel("epoch", fontsize=11)
    axes[1].set_ylabel("loss", fontsize=11)
    axes[1].set_title("Loss", fontsize=12)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(linestyle=":", alpha=0.35)
        ax.tick_params(axis="both", labelsize=10)
        ax.legend(loc="best", frameon=False, fontsize=10)

    fig.suptitle(title_prefix, fontsize=13, y=1.00)
    fig.tight_layout()
    out = save_fig(fig, save_name)
    plt.close(fig)
    return out


def plot_confusion_matrix(cm: np.ndarray, title: str, save_name: str,
                          class_names=None, normalize: bool = False) -> Path:
    if class_names is None:
        class_names = CLASS_NAMES
    if normalize:
        cm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = save_fig(fig, save_name)
    plt.close(fig)
    return out


def plot_sample_grid(image_arrays, labels, class_names=None, n_per_class: int = 4,
                     save_name: str = "sample_grid", title: str = "Random samples per class") -> Path:
    """Render a (n_classes x n_per_class) grid of images.

    The caller is responsible for choosing which images to show — this function
    just lays out whatever it receives, taking the first ``n_per_class`` images
    of each class label in the order they appear in ``image_arrays`` /
    ``labels``. The previous version computed ``idxs[c % len(idxs)]`` which
    silently always picked the first 4 entries; that hid bugs in the caller.
    """
    if class_names is None:
        class_names = CLASS_NAMES
    n_classes = len(class_names)
    labels_arr = np.asarray(labels)
    fig, axes = plt.subplots(n_classes, n_per_class, figsize=(n_per_class * 2.2, n_classes * 2.2))
    if n_classes == 1:
        axes = np.array([axes])
    for r, cname in enumerate(class_names):
        cls_idxs = np.where(labels_arr == r)[0]
        for c in range(n_per_class):
            ax = axes[r, c]
            if c >= len(cls_idxs):
                ax.axis("off"); continue
            img = image_arrays[cls_idxs[c]]
            ax.imshow(img.squeeze(), cmap="gray" if img.ndim == 2 or img.shape[-1] == 1 else None)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(cname, fontsize=10)
    fig.suptitle(title)
    fig.tight_layout()
    out = save_fig(fig, save_name)
    plt.close(fig)
    return out
