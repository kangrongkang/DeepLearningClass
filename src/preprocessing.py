"""
Day 2 preprocessing: stratified 70/15/15 split (file-list level), saved as CSVs.

Run as: ``python -m src.preprocessing`` from the project root.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CLASS_NAMES,
    DATA_ROOT,
    REPORTS_DIR,
    SEED,
    SPLITS_DIR,
    TEST_FRAC,
    TRAIN_FRAC,
    VAL_FRAC,
)
from src.io_utils import get_logger, list_images, save_split


def stratified_split(df: pd.DataFrame, train_frac: float = TRAIN_FRAC,
                     val_frac: float = VAL_FRAC, test_frac: float = TEST_FRAC,
                     seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Two-step stratified split: first train vs (val+test), then val vs test.

    Both calls pass ``stratify=labels`` so each class keeps its proportion in
    every split. Inputs come from ``list_images``, which sorts within each class
    folder, so the result is fully deterministic given ``seed``.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError(
            f"Fractions must sum to 1; got {train_frac} + {val_frac} + {test_frac}"
        )

    train_df, rest_df = train_test_split(
        df,
        train_size=train_frac,
        random_state=seed,
        stratify=df["label"],
        shuffle=True,
    )

    # Within the (val + test) chunk, val and test are already symmetric in the
    # default plan (0.15 each), so split 50/50 of that chunk.
    rel_test = test_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        rest_df,
        test_size=rel_test,
        random_state=seed,
        stratify=rest_df["label"],
        shuffle=True,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def split_summary(train_df, val_df, test_df) -> dict:
    """Per-class counts and totals for sanity-checking the split."""
    def per_class(df):
        return {c: int((df["class"] == c).sum()) for c in CLASS_NAMES}
    return {
        "train": {"total": len(train_df), "by_class": per_class(train_df)},
        "val":   {"total": len(val_df),   "by_class": per_class(val_df)},
        "test":  {"total": len(test_df),  "by_class": per_class(test_df)},
    }


def visualize_augmentation(train_df: pd.DataFrame, n_examples: int = 4,
                            n_aug_per_example: int = 4, seed: int = SEED):
    """Render a (n_examples x (1 + n_aug_per_example)) grid showing the original
    image alongside augmented copies. Saved to outputs/figures/02_augmentation_examples.png.

    This is the cheapest way to verify visually that the augmentation policy is
    'light' as the project plan requires (small rotation, shift, zoom).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import keras
    from PIL import Image
    from src.config import IMG_SIZE
    from src.plot_utils import save_fig

    rng = np.random.default_rng(seed)
    sample_idxs = rng.choice(len(train_df), size=n_examples, replace=False)
    sample_paths = train_df.iloc[sample_idxs]["filepath"].tolist()
    sample_labels = train_df.iloc[sample_idxs]["class"].tolist()

    aug = keras.Sequential([
        keras.layers.RandomRotation(factor=0.03, seed=seed),
        keras.layers.RandomTranslation(0.05, 0.05, seed=seed),
        keras.layers.RandomZoom(0.05, seed=seed),
    ])

    fig, axes = plt.subplots(n_examples, 1 + n_aug_per_example,
                             figsize=((1 + n_aug_per_example) * 2.0, n_examples * 2.0))
    if n_examples == 1:
        axes = np.array([axes])

    for r, (path, label) in enumerate(zip(sample_paths, sample_labels)):
        img = np.asarray(Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE)),
                         dtype=np.float32) / 255.0
        axes[r, 0].imshow(img)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(label, fontsize=9)
        if r == 0:
            axes[r, 0].set_title("original", fontsize=9)
        for c in range(n_aug_per_example):
            aug_out = aug(img[None, ...], training=True)
            # Backend-agnostic conversion to numpy (handles torch cuda tensors)
            if hasattr(aug_out, "detach"):
                aug_img = aug_out.detach().cpu().numpy()[0]
            else:
                aug_img = np.asarray(aug_out)[0]
            aug_img = np.clip(aug_img, 0.0, 1.0)
            axes[r, 1 + c].imshow(aug_img)
            axes[r, 1 + c].set_xticks([]); axes[r, 1 + c].set_yticks([])
            if r == 0:
                axes[r, 1 + c].set_title(f"aug {c+1}", fontsize=9)
    fig.suptitle("Light augmentation policy (rotation ±10°, ±5% shift, ±5% zoom; no h-flip)")
    fig.tight_layout()
    out = save_fig(fig, "02_augmentation_examples")
    plt.close(fig)
    return out


def run_preprocessing() -> dict:
    log = get_logger("preprocessing")
    t0 = time.time()
    log.info("Listing images under %s", DATA_ROOT)
    df = list_images(DATA_ROOT)
    log.info("Total images: %d", len(df))

    train_df, val_df, test_df = stratified_split(df)
    p_train = save_split(train_df, "train")
    p_val = save_split(val_df, "val")
    p_test = save_split(test_df, "test")
    log.info("Saved splits:\n  train -> %s\n  val   -> %s\n  test  -> %s",
             p_train, p_val, p_test)

    summary = split_summary(train_df, val_df, test_df)
    summary["fractions"] = {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC}
    summary["seed"] = SEED
    summary["elapsed_seconds"] = round(time.time() - t0, 2)
    summary["paths"] = {"train": str(p_train), "val": str(p_val), "test": str(p_test)}

    out = REPORTS_DIR / "02_split_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Wrote split summary: %s", out)
    return summary


if __name__ == "__main__":
    s = run_preprocessing()
    print(json.dumps(s, indent=2))
