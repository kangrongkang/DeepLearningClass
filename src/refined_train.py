"""
Phase D — train EfficientNet-B0 on the clean split with the refined recipe.

Three configs are run sequentially so each gain is *attributable*:

    a — bare:    AdamW + cosine + warmup, no Mixup, no smoothing, no SWA
    b — mixup:   a + Mixup alpha=0.2 (label smoothing OFF)
    c — full:    b + SWA on last 25% of epochs + TTA at inference

Each config does the standard two-stage schedule: frozen-backbone head
warm-up, then full fine-tune. Best weights from each stage are compared
and the better one is saved as the canonical ``efficientnet_b0_<cfg>.keras``.

Run from the project root:

    KERAS_BACKEND=torch python -m src.refined_train --config a
    KERAS_BACKEND=torch python -m src.refined_train --config b
    KERAS_BACKEND=torch python -m src.refined_train --config c
    KERAS_BACKEND=torch python -m src.refined_train --config all
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (BATCH_SIZE, MODELS_DIR, NUM_CLASSES, PREDICTIONS_DIR,
                        REPORTS_DIR, SEED, set_global_seed)
from src.io_utils import (get_logger, make_keras_dataset, save_metrics,
                          save_predictions)
from src.plot_utils import plot_confusion_matrix, plot_history


CONFIGS: dict[str, dict] = {
    "a": dict(mixup_alpha=0.0, label_smoothing=0.0, swa_start_frac=None,
              loss="categorical_crossentropy",
              tta=False, description="bare: AdamW + cosine, no recipe extras"),
    "b": dict(mixup_alpha=0.2, label_smoothing=0.0, swa_start_frac=None,
              loss="categorical_crossentropy",
              tta=False, description="+ Mixup alpha=0.2"),
    "c": dict(mixup_alpha=0.2, label_smoothing=0.0, swa_start_frac=0.75,
              loss="categorical_crossentropy",
              tta=True, description="+ SWA last 25% epochs + TTA at inference"),
}

CLEAN_SPLITS = {
    "train": "train_clean.csv",
    "val": "val_clean.csv",
    "test": "test_clean.csv",
}

# Default schedule (override via env or CLI for quick verification)
EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 15
LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-4
WARMUP = 2
INPUT_SIZE = 224


def _load_split(name: str) -> pd.DataFrame:
    from src.config import SPLITS_DIR
    return pd.read_csv(SPLITS_DIR / CLEAN_SPLITS[name])


def _build_datasets(train_df, val_df, test_df, mixup_alpha: float, batch_size: int):
    train_ds = make_keras_dataset(train_df, batch_size=batch_size, shuffle=True,
                                  augment=False, image_size=INPUT_SIZE,
                                  one_hot=True, num_classes=NUM_CLASSES,
                                  mixup_alpha=mixup_alpha, num_workers=2)
    val_ds = make_keras_dataset(val_df, batch_size=batch_size, shuffle=False,
                                augment=False, image_size=INPUT_SIZE,
                                one_hot=True, num_classes=NUM_CLASSES,
                                num_workers=1)
    test_ds = make_keras_dataset(test_df, batch_size=batch_size, shuffle=False,
                                 augment=False, image_size=INPUT_SIZE,
                                 one_hot=True, num_classes=NUM_CLASSES,
                                 num_workers=1)
    return train_ds, val_ds, test_ds


def _train_one_config(cfg_name: str, train_df, val_df, test_df,
                      epochs_stage1: int = EPOCHS_STAGE1,
                      epochs_stage2: int = EPOCHS_STAGE2,
                      batch_size: int = BATCH_SIZE) -> dict:
    import keras
    from src.models import build_efficientnet_b0
    from src.training import (configure_gpu, evaluate_and_save,
                              make_callbacks_v2, predict_with_tta)

    log = get_logger(f"refined.{cfg_name}",
                     logfile=REPORTS_DIR / f"efficientnet_b0_{cfg_name}_train.log")
    cfg = CONFIGS[cfg_name]
    log.info("=== Config %s — %s ===", cfg_name, cfg["description"])
    log.info("Schedule: stage1=%d epochs @ lr=%g, stage2=%d epochs @ lr=%g, warmup=%d",
             epochs_stage1, LR_STAGE1, epochs_stage2, LR_STAGE2, WARMUP)
    log.info("GPU: %s", configure_gpu())

    set_global_seed(SEED)

    # Datasets
    train_ds, val_ds, test_ds = _build_datasets(
        train_df, val_df, test_df, mixup_alpha=cfg["mixup_alpha"],
        batch_size=batch_size,
    )

    # ------- Stage 1: head warm-up, backbone frozen -------
    model = build_efficientnet_b0(
        freeze_backbone=True,
        learning_rate=LR_STAGE1,
        weight_decay=1e-4,
        loss=cfg["loss"],
        label_smoothing=cfg["label_smoothing"],
    )
    model_name = f"efficientnet_b0_{cfg_name}"
    log.info("Stage 1 — backbone frozen, %d trainable params",
             sum(int(v.numpy().size if hasattr(v, "numpy") else getattr(v, "shape", []) and 1)
                 for v in model.trainable_variables))

    cbs1 = make_callbacks_v2(
        model_name=f"{model_name}_stage1",
        total_epochs=epochs_stage1, initial_lr=LR_STAGE1,
        warmup_epochs=WARMUP,
        monitor="val_accuracy", mode="max",
        patience_es=5, swa_start_epoch=None, model=None,
    )
    t0 = time.time()
    hist1 = model.fit(train_ds, validation_data=val_ds,
                      epochs=epochs_stage1, callbacks=cbs1, verbose=2)
    s1_time = time.time() - t0

    # ------- Stage 2: full fine-tune -------
    # Unfreeze the entire backbone for fine-tuning at the smaller LR.
    backbone = None
    for layer in model.layers:
        if "efficientnetb0" in layer.name.lower() or "efficientnet-b0" in layer.name.lower():
            backbone = layer
            break
    if backbone is not None and hasattr(backbone, "layers"):
        for l in backbone.layers:
            l.trainable = True
        backbone.trainable = True
    else:
        # Inlined backbone (TF 2.21-style): mark all but the head trainable.
        head_names = {"image", "augment", "effnet_to_pixels", "gap", "dropout", "probs"}
        for l in model.layers:
            l.trainable = l.name not in head_names

    # Recompile with the smaller LR (AdamW preserved).
    optimizer = keras.optimizers.AdamW(learning_rate=LR_STAGE2, weight_decay=1e-4)
    if cfg["loss"] == "categorical_crossentropy":
        loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=cfg["label_smoothing"])
    elif cfg["loss"] == "focal_categorical":
        loss_fn = keras.losses.CategoricalFocalCrossentropy(
            alpha=0.25, gamma=2.0, label_smoothing=cfg["label_smoothing"])
    else:
        raise ValueError(f"Unsupported loss '{cfg['loss']}'")
    model.compile(optimizer=optimizer, loss=loss_fn,
                  metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")])

    swa_start = None
    if cfg["swa_start_frac"] is not None:
        swa_start = max(int(round(epochs_stage2 * cfg["swa_start_frac"])), 1)
        log.info("Stage 2 — SWA starting at epoch %d (=%.0f%% of %d)",
                 swa_start, cfg["swa_start_frac"] * 100, epochs_stage2)

    cbs2 = make_callbacks_v2(
        model_name=model_name, total_epochs=epochs_stage2, initial_lr=LR_STAGE2,
        warmup_epochs=max(1, WARMUP // 2),
        monitor="val_accuracy", mode="max",
        patience_es=5, swa_start_epoch=swa_start, model=model,
    )
    t1 = time.time()
    hist2 = model.fit(train_ds, validation_data=val_ds,
                      epochs=epochs_stage2, callbacks=cbs2, verbose=2)
    s2_time = time.time() - t1

    # Pick the best of the two stages
    best_s1 = max(hist1.history.get("val_accuracy", [0.0]))
    best_s2 = max(hist2.history.get("val_accuracy", [0.0]))
    log.info("Stage 1 best val_acc=%.4f, Stage 2 best=%.4f", best_s1, best_s2)
    final_path = MODELS_DIR / f"{model_name}.keras"
    if best_s1 > best_s2:
        s1_path = MODELS_DIR / f"{model_name}_stage1.keras"
        if s1_path.exists():
            shutil.copy(s1_path, final_path)
            model = keras.models.load_model(final_path)
            log.info("Stage 1 won; copied %s -> %s and reloaded.",
                     s1_path.name, final_path.name)

    # ------- Evaluation on the clean test set -------
    log.info("Evaluating on clean test set...")
    metrics = evaluate_and_save(model, test_ds, model_name=model_name,
                                history={**hist1.history, **hist2.history,
                                         "elapsed_seconds": s1_time + s2_time},
                                test_df=test_df)
    log.info("Test acc=%.4f  macroF1=%.4f", metrics["accuracy"], metrics["macro_f1"])

    if cfg["tta"]:
        log.info("Running TTA inference...")
        proba_tta = predict_with_tta(model, test_ds, n_aug=4, seed=SEED)
        y_true = np.asarray(test_df["label"].values)
        y_pred_tta = proba_tta.argmax(axis=1)
        from sklearn.metrics import (accuracy_score, classification_report,
                                     confusion_matrix, f1_score,
                                     precision_score, recall_score)
        from src.config import CLASS_NAMES
        tta_metrics = {
            "model_name": f"{model_name}_tta",
            "n_test": int(len(y_true)),
            "accuracy": float(accuracy_score(y_true, y_pred_tta)),
            "macro_precision": float(precision_score(y_true, y_pred_tta,
                                                     average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_true, y_pred_tta,
                                                average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred_tta,
                                       average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred_tta,
                                          average="weighted", zero_division=0)),
            "per_class": {
                "precision": precision_score(y_true, y_pred_tta, average=None,
                                             zero_division=0,
                                             labels=range(len(CLASS_NAMES))).tolist(),
                "recall": recall_score(y_true, y_pred_tta, average=None,
                                       zero_division=0,
                                       labels=range(len(CLASS_NAMES))).tolist(),
                "f1": f1_score(y_true, y_pred_tta, average=None, zero_division=0,
                                labels=range(len(CLASS_NAMES))).tolist(),
                "labels": list(CLASS_NAMES),
            },
            "confusion_matrix": confusion_matrix(y_true, y_pred_tta,
                                                 labels=range(len(CLASS_NAMES))).tolist(),
        }
        save_predictions(f"{model_name}_tta", y_true, y_pred_tta, proba_tta,
                         relpaths=test_df["relpath"].tolist()
                         if "relpath" in test_df.columns else None)
        save_metrics(f"{model_name}_tta", tta_metrics)
        plot_confusion_matrix(np.asarray(tta_metrics["confusion_matrix"]),
                              title=f"{model_name}_tta — confusion matrix",
                              save_name=f"{model_name}_tta_confusion_matrix")
        log.info("TTA test acc=%.4f  macroF1=%.4f",
                 tta_metrics["accuracy"], tta_metrics["macro_f1"])

    return {
        "config": cfg_name,
        "stage1_seconds": round(s1_time, 1),
        "stage2_seconds": round(s2_time, 1),
        "stage1_best_val_accuracy": best_s1,
        "stage2_best_val_accuracy": best_s2,
        "test_accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIGS.keys()) + ["all"],
                        default="all")
    parser.add_argument("--epochs-stage1", type=int, default=EPOCHS_STAGE1)
    parser.add_argument("--epochs-stage2", type=int, default=EPOCHS_STAGE2)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--sample-frac", type=float, default=1.0,
                        help="Subsample the clean split for fast verification.")
    args = parser.parse_args()

    log = get_logger("refined.driver",
                     logfile=REPORTS_DIR / "phaseD_refined_run.log")

    train_df = _load_split("train")
    val_df = _load_split("val")
    test_df = _load_split("test")

    if args.sample_frac < 1.0:
        train_df = train_df.sample(frac=args.sample_frac, random_state=SEED)
        val_df = val_df.sample(frac=args.sample_frac, random_state=SEED)
        test_df = test_df.sample(frac=args.sample_frac, random_state=SEED)
        log.info("Subsampled to frac=%.2f -> train=%d val=%d test=%d",
                 args.sample_frac, len(train_df), len(val_df), len(test_df))

    configs = list(CONFIGS.keys()) if args.config == "all" else [args.config]
    summary: list[dict] = []
    for c in configs:
        log.info("\n>>> Running config %s", c)
        try:
            res = _train_one_config(c, train_df, val_df, test_df,
                                    epochs_stage1=args.epochs_stage1,
                                    epochs_stage2=args.epochs_stage2,
                                    batch_size=args.batch_size)
            summary.append(res)
            log.info("Config %s done: %s", c, res)
        except Exception as e:
            log.exception("Config %s failed: %s", c, e)
            summary.append({"config": c, "error": str(e)})

    out = REPORTS_DIR / "phaseD_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Phase D summary saved to %s", out)
    return summary


if __name__ == "__main__":
    main()
