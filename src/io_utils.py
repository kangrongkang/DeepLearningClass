"""
Reusable helpers for image I/O, dataset building, plotting, and result persistence.
Imported by every notebook so data loading and figure styling stay consistent.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import (
    CLASS_NAMES,
    DATA_ROOT,
    IMG_SIZE,
    IMG_CHANNELS,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    SPLITS_DIR,
)


def get_logger(name: str = "alz_mri", logfile: Path | str | None = None) -> logging.Logger:
    """Configured logger that prints to stdout and (optionally) a file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logfile is not None:
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def list_images(data_root: Path | str = DATA_ROOT) -> pd.DataFrame:
    """Walk DATA_ROOT and return a DataFrame with columns [relpath, class, label, filepath].

    * ``relpath`` is the path relative to ``data_root`` (forward-slashed) — this is
      what gets persisted in the split CSVs, so the same CSV works on Windows
      and on Colab without rewriting.
    * ``filepath`` is the absolute resolved path on the *current* machine — convenient
      for any code that wants to ``open()`` the file immediately. Do not save it.
    * Files are returned in **sorted order by filename within each class** so the
      stratified split in Day 2 is deterministic across OSes.
    """
    data_root = Path(data_root)
    rows = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = data_root / cls_name
        if not cls_dir.exists():
            continue
        files = sorted(
            (p for p in cls_dir.iterdir()
             if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}),
            key=lambda p: p.name,
        )
        for p in files:
            rel = f"{cls_name}/{p.name}"  # forward-slash, portable
            rows.append({
                "relpath": rel,
                "class": cls_name,
                "label": cls_idx,
                "filepath": str(p),
            })
    df = pd.DataFrame(rows)
    return df


def _resolve_paths(df: pd.DataFrame, data_root: Path | str = DATA_ROOT) -> pd.DataFrame:
    """Make sure ``df`` has an absolute ``filepath`` column for the current host.

    A split CSV stores ``relpath`` only (so it is portable). When we load it for
    training we recompute ``filepath = data_root / relpath`` — this is the only
    place that knows about the local layout.
    """
    if "filepath" in df.columns and df["filepath"].notna().all() and \
            Path(df["filepath"].iloc[0]).exists():
        return df
    if "relpath" not in df.columns:
        raise ValueError("DataFrame is missing both 'filepath' and 'relpath'; cannot resolve.")
    data_root = Path(data_root)
    df = df.copy()
    df["filepath"] = df["relpath"].astype(str).map(lambda r: str(data_root / r))
    return df


def save_split(df: pd.DataFrame, split_name: str) -> Path:
    """Save a split DataFrame to outputs/splits/<split_name>.csv.

    Only the *portable* columns are written: ``relpath, class, label``. The
    absolute ``filepath`` column is dropped on save so the same CSV works on
    Windows and on Colab.
    """
    out = SPLITS_DIR / f"{split_name}.csv"
    cols = [c for c in ["relpath", "class", "label"] if c in df.columns]
    df[cols].to_csv(out, index=False)
    return out


def load_split(split_name: str, data_root: Path | str = DATA_ROOT) -> pd.DataFrame:
    """Load a split CSV and re-attach the absolute ``filepath`` for the current host.

    Also asserts that the (class, label) pairs in the CSV match the canonical
    ``CLASS_NAMES`` order in ``src/config.py`` — if Day 2 is ever rerun with a
    different config, this catches the mismatch immediately.
    """
    df = pd.read_csv(SPLITS_DIR / f"{split_name}.csv")
    if "class" in df.columns and "label" in df.columns:
        seen = df[["class", "label"]].drop_duplicates().sort_values("label")
        for _, row in seen.iterrows():
            expected = CLASS_NAMES[int(row["label"])]
            if row["class"] != expected:
                raise ValueError(
                    f"Split '{split_name}' has mismatched class/label: row says "
                    f"label={row['label']} -> '{row['class']}', config expects "
                    f"'{expected}'. Re-run src.preprocessing to regenerate splits."
                )
    return _resolve_paths(df, data_root)


def save_predictions(model_name: str, y_true: np.ndarray, y_pred: np.ndarray,
                     y_proba: np.ndarray | None = None,
                     relpaths: list | np.ndarray | None = None) -> Path:
    """Persist test predictions for cross-model comparison and Grad-CAM later.

    ``relpaths`` is a per-prediction lookup back to the source image — required
    by Grad-CAM (Day 7) so it can show the actual images that were predicted on.
    Storing it alongside avoids re-deriving from the split CSV, which doesn't
    work when the model was trained on a subsampled (SMOKE-mode) test set.
    """
    out = PREDICTIONS_DIR / f"{model_name}_test_predictions.npz"
    payload = {"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred)}
    if y_proba is not None:
        payload["y_proba"] = np.asarray(y_proba)
    if relpaths is not None:
        payload["relpaths"] = np.asarray(relpaths, dtype=object)
    np.savez_compressed(out, **payload)
    return out


def save_metrics(model_name: str, metrics: dict) -> Path:
    """Persist a JSON of headline metrics so the comparison notebook can pick them up."""
    out = PREDICTIONS_DIR / f"{model_name}_metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return out


def write_report(name: str, body: str) -> Path:
    """Write a markdown report to outputs/reports/<name>.md."""
    out = REPORTS_DIR / f"{name}.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(body)
    return out


def make_keras_dataset(df: pd.DataFrame, batch_size: int, shuffle: bool,
                       augment: bool = False, seed: int = 42,
                       num_workers: int = 0,
                       image_size: int | None = None,
                       one_hot: bool = False, num_classes: int | None = None,
                       mixup_alpha: float = 0.0):
    """Build a backend-agnostic Keras 3 ``PyDataset`` from a (filepath, label) DataFrame.

    Images are decoded with PIL, forced to 3 channels, resized to ``image_size``
    (default ``IMG_SIZE`` from config), and normalized to ``[0, 1]``.

    Parameters
    ----------
    augment : if True, run RandomRotation/Translation/Zoom inside the dataset.
        **Convention for this project: leave ``augment=False`` here and let
        the model factory's ``_augmentation_layers`` block handle augmentation
        in-graph.** Setting ``augment=True`` here AND ``augment=True`` in the
        model factory effectively *doubles* the augmentation strength, which
        was a real bug caught in code review.
    image_size : optional override of the global IMG_SIZE for the refined
        Phase C/D pipeline (e.g. 224 for EfficientNet-B0).
    one_hot : if True, labels come back as ``(B, num_classes)`` one-hot floats —
        required for Mixup and ``CategoricalCrossentropy``.
    mixup_alpha : if > 0 and ``one_hot=True``, apply Mixup (Zhang et al. 2018,
        arXiv:1710.09412) within each batch with mixing weight drawn from
        ``Beta(alpha, alpha)``. Default 0.0 (off).
    """
    cls = _get_pydataset_class()
    return cls(df=df, batch_size=batch_size, shuffle=shuffle,
               augment=augment, seed=seed, num_workers=num_workers,
               image_size=image_size, one_hot=one_hot,
               num_classes=num_classes, mixup_alpha=mixup_alpha)


# Backwards-compatible alias for any code paths still importing the old name
make_tf_dataset = make_keras_dataset


_DATASET_CLASS_CACHE = None


def _get_pydataset_class():
    """Build (and cache) a class that subclasses ``keras.utils.PyDataset``.

    Keras is imported lazily so ``import src.io_utils`` doesn't require Keras
    to be installed (e.g. for the Day 1 inspection script).
    """
    global _DATASET_CLASS_CACHE
    if _DATASET_CLASS_CACHE is not None:
        return _DATASET_CLASS_CACHE

    import keras

    class _ImageDirectoryDataset(keras.utils.PyDataset):
        def __init__(self, df, batch_size, shuffle, augment, seed, num_workers,
                     image_size=None, one_hot=False, num_classes=None,
                     mixup_alpha=0.0):
            super().__init__(workers=num_workers, use_multiprocessing=False)
            df = _resolve_paths(df)
            self._paths = df["filepath"].astype(str).tolist()
            self._labels = np.asarray(df["label"].astype(int).tolist(), dtype=np.int64)
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.augment = augment
            self.seed = seed
            self.image_size = int(image_size if image_size is not None else IMG_SIZE)
            self.one_hot = bool(one_hot)
            self.num_classes = int(num_classes) if num_classes is not None else None
            self.mixup_alpha = float(mixup_alpha)

            if self.one_hot and self.num_classes is None:
                raise ValueError("one_hot=True requires num_classes")
            if self.mixup_alpha > 0 and not self.one_hot:
                raise ValueError("mixup_alpha>0 requires one_hot=True (mixed targets are soft)")

            self._aug_layers = None
            if augment:
                self._aug_layers = keras.Sequential([
                    keras.layers.RandomRotation(factor=0.03, seed=seed),
                    keras.layers.RandomTranslation(0.05, 0.05, seed=seed),
                    keras.layers.RandomZoom(0.05, seed=seed),
                ], name="light_augment")

            self._index = np.arange(len(self._paths))
            self._rng = np.random.default_rng(seed)
            if self.shuffle:
                self._rng.shuffle(self._index)

        def __len__(self) -> int:
            return (len(self._paths) + self.batch_size - 1) // self.batch_size

        def _make_y(self, raw_labels: np.ndarray) -> np.ndarray:
            if not self.one_hot:
                return raw_labels.astype(np.int64)
            y = np.zeros((len(raw_labels), self.num_classes), dtype=np.float32)
            y[np.arange(len(raw_labels)), raw_labels] = 1.0
            return y

        def __getitem__(self, batch_idx: int):
            from PIL import Image
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(self._paths))
            idxs = self._index[start:end]

            sz = self.image_size
            X = np.zeros((len(idxs), sz, sz, IMG_CHANNELS), dtype=np.float32)
            for i, src_i in enumerate(idxs):
                img = Image.open(self._paths[src_i]).convert("RGB").resize(
                    (sz, sz), resample=Image.BILINEAR)
                X[i] = np.asarray(img, dtype=np.float32) / 255.0
            raw_labels = self._labels[idxs]
            y = self._make_y(raw_labels)

            if self._aug_layers is not None:
                aug_out = self._aug_layers(X, training=True)
                if hasattr(aug_out, "detach"):
                    X = aug_out.detach().cpu().numpy().astype(np.float32, copy=False)
                else:
                    X = np.asarray(aug_out, dtype=np.float32)

            # Mixup (Zhang et al., ICLR 2018, arXiv:1710.09412): one lambda
            # per batch, drawn from Beta(alpha, alpha), reflected to [0.5, 1]
            # so we don't catastrophically wipe out the primary sample. We
            # use a random *non-zero* roll as the permutation so no sample
            # gets mixed with itself.
            if self.mixup_alpha > 0 and len(X) > 1:
                lam = float(self._rng.beta(self.mixup_alpha, self.mixup_alpha))
                lam = max(lam, 1.0 - lam)
                shift = int(self._rng.integers(1, len(X)))  # 1..len(X)-1
                X = lam * X + (1.0 - lam) * np.roll(X, shift, axis=0)
                y = lam * y + (1.0 - lam) * np.roll(y, shift, axis=0)
                X = X.astype(np.float32, copy=False)
                y = y.astype(np.float32, copy=False)
            return X, y

        def on_epoch_end(self):
            if self.shuffle:
                self._rng.shuffle(self._index)

    _DATASET_CLASS_CACHE = _ImageDirectoryDataset
    return _DATASET_CLASS_CACHE


def to_numpy_dataset(df: pd.DataFrame, max_items: int | None = None,
                     normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Eagerly load (X, y) numpy arrays — convenient for small subsets like Grad-CAM."""
    from PIL import Image
    df = _resolve_paths(df)
    if max_items is not None:
        df = df.iloc[:max_items]
    X = np.zeros((len(df), IMG_SIZE, IMG_SIZE, IMG_CHANNELS), dtype=np.float32)
    y = np.zeros(len(df), dtype=np.int64)
    for i, row in enumerate(df.itertuples()):
        img = Image.open(row.filepath).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.asarray(img, dtype=np.float32)
        if normalize:
            arr /= 255.0
        X[i] = arr
        y[i] = row.label
    return X, y
