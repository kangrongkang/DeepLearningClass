"""
Phase B — re-evaluate the saved baseline_cnn / mobilenetv2 / vgg16 checkpoints
on the leakage-controlled (clean) test set.

Each model was trained at 128 x 128, so we load the test images at 128 x 128
to avoid attributing a resolution mismatch to leakage.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.config import CLASS_NAMES, DATA_ROOT, MODELS_DIR, PREDICTIONS_DIR, REPORTS_DIR, SPLITS_DIR
from src.io_utils import get_logger


def _resolve_paths(df: pd.DataFrame) -> pd.DataFrame:
    """Re-attach the absolute filepath column for the current host."""
    df = df.copy()
    df["filepath"] = df["relpath"].astype(str).map(lambda r: str(DATA_ROOT / r))
    return df


def _load_batch(paths: list[str], size: int, channels: int = 3) -> np.ndarray:
    out = np.zeros((len(paths), size, size, channels), dtype=np.float32)
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB").resize((size, size), resample=Image.BILINEAR)
        out[i] = np.asarray(img, dtype=np.float32) / 255.0
    return out


def reeval_model(model_name: str, input_size: int = 128,
                 batch_size: int = 64,
                 split_csv: str = "test_clean.csv") -> dict:
    """Load `outputs/models/<model_name>.keras` and predict on the clean test set.

    Saves:
      * outputs/predictions/<model_name>_clean_test_predictions.npz
      * outputs/predictions/<model_name>_clean_metrics.json
    Returns the metrics dict.
    """
    import keras
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score,
                                 precision_score, recall_score)

    log = get_logger(f"reeval.{model_name}")
    model_path = MODELS_DIR / f"{model_name}.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Cannot re-eval '{model_name}': {model_path} not found")

    log.info("Loading %s", model_path)
    model = keras.models.load_model(model_path)
    log.info("Loaded; native input shape: %s", getattr(model.inputs[0], "shape", "?"))

    test_df = pd.read_csv(SPLITS_DIR / split_csv)
    test_df = _resolve_paths(test_df)
    log.info("Test set: %d images @ %dx%d", len(test_df), input_size, input_size)

    paths = test_df["filepath"].tolist()
    labels = test_df["label"].astype(int).values

    y_proba = np.zeros((len(paths), len(CLASS_NAMES)), dtype=np.float32)
    t0 = time.time()
    for start in range(0, len(paths), batch_size):
        end = min(start + batch_size, len(paths))
        X = _load_batch(paths[start:end], size=input_size)
        proba = model.predict(X, verbose=0)
        if hasattr(proba, "detach"):
            proba = proba.detach().cpu().numpy()
        y_proba[start:end] = np.asarray(proba)
        if (start // batch_size) % 25 == 0:
            log.info("  %d / %d  (%.1fs)", end, len(paths), time.time() - t0)
    log.info("Inference done in %.1fs", time.time() - t0)

    y_true = labels
    y_pred = y_proba.argmax(axis=1)

    metrics = {
        "model_name": model_name,
        "split": split_csv,
        "input_size": input_size,
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            "precision": precision_score(y_true, y_pred, average=None, zero_division=0,
                                         labels=range(len(CLASS_NAMES))).tolist(),
            "recall": recall_score(y_true, y_pred, average=None, zero_division=0,
                                   labels=range(len(CLASS_NAMES))).tolist(),
            "f1": f1_score(y_true, y_pred, average=None, zero_division=0,
                            labels=range(len(CLASS_NAMES))).tolist(),
            "labels": list(CLASS_NAMES),
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred,
                                             labels=range(len(CLASS_NAMES))).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=range(len(CLASS_NAMES)),
            target_names=CLASS_NAMES, zero_division=0, output_dict=True,
        ),
    }

    pred_path = PREDICTIONS_DIR / f"{model_name}_clean_test_predictions.npz"
    np.savez_compressed(pred_path,
                        y_true=y_true, y_pred=y_pred, y_proba=y_proba,
                        relpaths=np.asarray(test_df["relpath"].tolist(), dtype=object))
    metrics_path = PREDICTIONS_DIR / f"{model_name}_clean_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("Saved %s and %s", pred_path.name, metrics_path.name)
    log.info("Test acc=%.4f  macroF1=%.4f", metrics["accuracy"], metrics["macro_f1"])
    return metrics


def reeval_all(models=("baseline_cnn", "mobilenetv2", "vgg16"), input_size=128) -> dict:
    out = {}
    for m in models:
        out[m] = reeval_model(m, input_size=input_size)
    summary_path = REPORTS_DIR / "phaseB_clean_reeval_summary.json"
    summary = {m: {"accuracy": out[m]["accuracy"], "macro_f1": out[m]["macro_f1"],
                   "n_test": out[m]["n_test"]} for m in out}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    s = reeval_all()
    for m, info in s.items():
        print(f"{m:14s} clean_acc={info['accuracy']:.4f}  clean_macroF1={info['macro_f1']:.4f}")
