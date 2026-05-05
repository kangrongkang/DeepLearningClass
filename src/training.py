"""
Shared training / evaluation helpers used by Days 3-5.

* ``train_model``  — compile-already done in factory; this just runs ``fit`` with
                     EarlyStopping + ReduceLROnPlateau + ModelCheckpoint.
* ``evaluate_and_save``  — compute metrics on the test set and persist everything
                            the comparison notebook expects.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from src.config import CLASS_NAMES, MODELS_DIR, REPORTS_DIR
from src.io_utils import get_logger, save_metrics, save_predictions, write_report
from src.plot_utils import plot_confusion_matrix, plot_history


def configure_gpu(memory_growth: bool = True) -> dict:
    """Detect the active backend's GPU and report what the kernel actually sees.

    Works for both Keras 3 + PyTorch backend (default in this project) and
    Keras 3 + TensorFlow backend. Returns a dict with ``device`` set to
    ``"GPU"`` or ``"CPU"`` so the notebooks can branch on it.
    """
    info = {"backend": "?", "gpus": [], "device": "CPU"}
    try:
        import keras
        info["keras_version"] = keras.__version__
        info["backend"] = keras.backend.backend()
    except ImportError:
        return info

    if info["backend"] == "torch":
        try:
            import torch
            info["torch_version"] = torch.__version__
            if torch.cuda.is_available():
                info["device"] = "GPU"
                info["gpus"] = [torch.cuda.get_device_name(i)
                                for i in range(torch.cuda.device_count())]
                info["cuda"] = torch.version.cuda
                info["compute_capability"] = list(torch.cuda.get_device_capability(0))
        except Exception as e:  # pragma: no cover
            info["error"] = str(e)
    elif info["backend"] == "tensorflow":
        try:
            import tensorflow as tf
            info["tf_version"] = tf.__version__
            gpus = tf.config.list_physical_devices("GPU")
            info["gpus"] = [g.name for g in gpus]
            if gpus:
                info["device"] = "GPU"
                for g in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(g, memory_growth)
                    except RuntimeError:
                        pass
        except Exception as e:  # pragma: no cover
            info["error"] = str(e)
    return info


def cosine_warmup_schedule(initial_lr: float, total_epochs: int,
                            warmup_epochs: int = 2, min_lr_factor: float = 0.01):
    """Linear warmup for ``warmup_epochs`` epochs, then cosine decay to
    ``initial_lr * min_lr_factor``.

    **Convention.** ``epoch`` is the 0-indexed epoch number Keras passes in.
    Warmup runs over epochs ``[0, warmup_epochs - 1]`` inclusive, so the LR
    on epoch 0 is ``initial_lr / warmup_epochs`` and the LR on epoch
    ``warmup_epochs - 1`` is ``initial_lr``. Cosine decay then starts on
    epoch ``warmup_epochs``.

    Returned as a closure usable with ``keras.callbacks.LearningRateScheduler``.
    Reference: Loshchilov & Hutter, ICLR 2017 / 2019 cosine-restart family.
    """
    import math

    def schedule(epoch: int, lr: float) -> float:
        if epoch < warmup_epochs:
            # epoch 0 -> initial/warmup; epoch warmup-1 -> initial
            return float(initial_lr * (epoch + 1) / max(warmup_epochs, 1))
        progress = (epoch - warmup_epochs + 1) / max(total_epochs - warmup_epochs, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(initial_lr * (min_lr_factor + (1.0 - min_lr_factor) * cosine))

    return schedule


def make_swa_callback(start_epoch: int, model):
    """Stochastic Weight Averaging callback (Izmailov et al. UAI 2018,
    arXiv:1803.05407).

    **Convention.** ``start_epoch`` is **1-indexed and inclusive** — the
    first epoch whose weights get sampled into the running average. With
    ``start_epoch=2`` and 5 total epochs, weights from epochs 2, 3, 4, 5 are
    averaged (4 samples).

    The averaged weights are written back into ``model`` in
    ``on_train_end``, so the model emerges from training with the averaged
    weights and ``model.save`` will persist them.
    """
    import keras
    import numpy as np

    class _SWA(keras.callbacks.Callback):
        def __init__(self, start_epoch: int):
            super().__init__()
            self.start_epoch = int(start_epoch)
            self._n = 0
            self._avg: list[np.ndarray] | None = None

        def on_epoch_end(self, epoch, logs=None):
            # epoch is 0-indexed; SWA convention is 1-indexed inclusive
            if (epoch + 1) < self.start_epoch:
                return
            current = [np.asarray(w) for w in self.model.get_weights()]
            if self._avg is None:
                self._avg = current
                self._n = 1
            else:
                self._n += 1
                self._avg = [
                    avg + (cur - avg) / self._n
                    for avg, cur in zip(self._avg, current)
                ]

        def on_train_end(self, logs=None):
            if self._avg is None:
                return
            self.model.set_weights(self._avg)

    return _SWA(start_epoch)


def make_callbacks_v2(model_name: str, total_epochs: int,
                      initial_lr: float, warmup_epochs: int = 2,
                      monitor: str = "val_accuracy", mode: str = "max",
                      patience_es: int = 5, swa_start_epoch: int | None = None,
                      model=None) -> list:
    """Refined callback set for Phase D (cosine warmup + EarlyStopping + ModelCheckpoint
    + optional SWA). Returns a list of callbacks; the caller adds them to
    ``model.fit(..., callbacks=cbs)``."""
    import keras
    ckpt_path = MODELS_DIR / f"{model_name}.keras"
    cbs = [
        keras.callbacks.LearningRateScheduler(
            cosine_warmup_schedule(initial_lr=initial_lr, total_epochs=total_epochs,
                                    warmup_epochs=warmup_epochs),
            verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor, mode=mode, patience=patience_es,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path), monitor=monitor, mode=mode,
            save_best_only=True, verbose=0,
        ),
    ]
    if swa_start_epoch is not None and model is not None:
        cbs.append(make_swa_callback(swa_start_epoch, model))
    return cbs


def predict_with_tta(model, ds, n_aug: int = 4, seed: int = 42) -> np.ndarray:
    """Test-time augmentation: average softmax over the original images plus
    ``n_aug`` lightly-augmented copies.

    **Correctness note:** we apply augmentation *outside* the model graph
    (a separate ``keras.Sequential`` block) and always call
    ``model.predict(..., verbose=0)`` (which runs in inference mode), so
    BatchNorm running stats stay frozen at their training-end values across
    every TTA pass. The earlier version called ``model(x, training=True)``
    which both fires the model's *internal* augment layers AND mutates BN
    running statistics — the latter is unsafe and was a real bug.

    The dataset is expected to be ``shuffle=False`` (every TTA pass needs to
    iterate the same image order). We assert that here.
    """
    import keras

    log = get_logger("predict.tta")
    if getattr(ds, "shuffle", False):
        raise ValueError("predict_with_tta requires a shuffle=False dataset; "
                         "passes must iterate images in identical order.")

    # Build a TTA-only augment block. Slightly stronger than training-time
    # because TTA wants visible-but-small perturbations across passes.
    tta_aug = keras.Sequential([
        keras.layers.RandomRotation(factor=0.04, seed=seed),
        keras.layers.RandomTranslation(0.05, 0.05, seed=seed),
        keras.layers.RandomZoom(0.05, seed=seed),
    ], name="tta_augment")

    def _to_numpy(t):
        if hasattr(t, "detach"):
            return t.detach().cpu().numpy()
        return np.asarray(t)

    log.info("TTA: deterministic pass + %d augmented passes (BN frozen)", n_aug)
    chunks: list[np.ndarray] = [None] * (n_aug + 1)
    for pass_idx in range(n_aug + 1):
        per_pass: list[np.ndarray] = []
        for i in range(len(ds)):
            xb, _ = ds[i]
            if pass_idx > 0:
                # Apply augmentation OUTSIDE the model so BN stays in eval.
                xb_aug = _to_numpy(tta_aug(xb, training=True)).astype(np.float32, copy=False)
            else:
                xb_aug = xb  # deterministic pass
            out = model.predict(xb_aug, verbose=0)
            per_pass.append(_to_numpy(out))
        chunks[pass_idx] = np.concatenate(per_pass, axis=0)

    return np.mean(np.stack(chunks, axis=0), axis=0).astype(np.float32)


def make_callbacks(model_name: str, monitor: str = "val_accuracy",
                   patience_es: int = 5, patience_lr: int = 2,
                   factor_lr: float = 0.5, min_lr: float = 1e-6) -> list:
    """Standard callback set used for every model.

    * EarlyStopping with ``patience_es`` epochs on ``monitor`` (max mode for accuracy).
    * ReduceLROnPlateau with ``patience_lr`` epochs.
    * ModelCheckpoint that keeps the *best* weights only at outputs/models/<name>.keras.
    """
    import keras

    ckpt_path = MODELS_DIR / f"{model_name}.keras"
    return [
        keras.callbacks.EarlyStopping(
            monitor=monitor, patience=patience_es,
            restore_best_weights=True, mode="max", verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=factor_lr, patience=patience_lr,
            min_lr=min_lr, mode="max", verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path), monitor=monitor, save_best_only=True,
            mode="max", verbose=0,
        ),
    ]


def train_model(model, train_ds, val_ds, model_name: str, epochs: int,
                callbacks: list | None = None) -> dict:
    """Run ``model.fit`` and return the history dict (with elapsed time)."""
    log = get_logger(f"train.{model_name}", logfile=REPORTS_DIR / f"{model_name}_train.log")
    if callbacks is None:
        callbacks = make_callbacks(model_name)
    log.info("Training '%s' for up to %d epochs", model_name, epochs)
    t0 = time.time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                     callbacks=callbacks, verbose=2)
    elapsed = round(time.time() - t0, 2)
    out = dict(hist.history)
    out["elapsed_seconds"] = elapsed
    log.info("Done: %.2f s, %d epochs", elapsed, len(hist.history.get("loss", [])))
    return out


def evaluate_and_save(model, test_ds, model_name: str, history: dict | None = None,
                      save_curves: bool = True,
                      test_df=None) -> dict:
    """Compute test metrics, persist predictions / metrics / figures, return a dict.

    Side effects:
      * ``outputs/predictions/<name>_test_predictions.npz`` — y_true / y_pred / y_proba.
      * ``outputs/predictions/<name>_metrics.json`` — accuracy / precision / recall / F1
        (macro & per-class) + confusion matrix.
      * ``outputs/figures/<name>_confusion_matrix.png``.
      * ``outputs/figures/<name>_training_curves.png`` (if history given).
    """
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, f1_score,
        precision_score, recall_score,
    )

    log = get_logger(f"eval.{model_name}")
    log.info("Evaluating '%s' on test set", model_name)

    # test_ds is a keras.utils.PyDataset; iterate batch-by-batch.
    y_true_list, y_proba_list = [], []
    for i in range(len(test_ds)):
        xb, yb = test_ds[i]
        proba = model.predict(xb, verbose=0)
        y_true_list.append(np.asarray(yb))
        # Some Keras 3 backends may return a torch/tf tensor from predict()
        if hasattr(proba, "detach"):
            proba = proba.detach().cpu().numpy()
        else:
            proba = np.asarray(proba)
        y_proba_list.append(proba)
    y_true = np.concatenate(y_true_list)
    # Handle one-hot labels (refined-recipe path) vs integer labels (legacy path).
    if y_true.ndim == 2 and y_true.shape[-1] > 1:
        y_true = y_true.argmax(axis=1)
    y_true = y_true.astype(np.int64)
    y_proba = np.concatenate(y_proba_list)
    y_pred = y_proba.argmax(axis=1)

    metrics = {
        "model_name": model_name,
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
    if history is not None:
        metrics["training"] = {
            "epochs_run": len(history.get("loss", [])),
            "elapsed_seconds": history.get("elapsed_seconds"),
            "best_val_accuracy": float(np.max(history.get("val_accuracy", [0.0]))),
            "best_val_loss": float(np.min(history.get("val_loss", [np.inf]))),
        }

    relpaths = None
    if test_df is not None and "relpath" in getattr(test_df, "columns", []):
        rp = test_df["relpath"].astype(str).tolist()
        if len(rp) == len(y_true):
            relpaths = rp
        else:
            log.warning("test_df has %d rows but predictions have %d — skipping relpaths.",
                        len(rp), len(y_true))
    save_predictions(model_name, y_true, y_pred, y_proba, relpaths=relpaths)
    save_metrics(model_name, metrics)

    cm_path = plot_confusion_matrix(
        np.asarray(metrics["confusion_matrix"]),
        title=f"{model_name} — confusion matrix",
        save_name=f"{model_name}_confusion_matrix",
    )
    log.info("Confusion matrix figure: %s", cm_path)

    if history is not None and save_curves:
        curves_path = plot_history(history, title_prefix=model_name,
                                   save_name=f"{model_name}_training_curves")
        log.info("Training curves figure: %s", curves_path)

    log.info("Test accuracy: %.4f   macro F1: %.4f", metrics["accuracy"], metrics["macro_f1"])
    return metrics
