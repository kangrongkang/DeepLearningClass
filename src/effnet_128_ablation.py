"""
Reviewer-requested ablation: train EfficientNet-B0 at 128x128 (the same input
resolution as the baselines) with the same recipe as config 'a'. The point is
to disentangle "architecture" from "resolution" in the +8.4 pp gain.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (BATCH_SIZE, MODELS_DIR, NUM_CLASSES, PREDICTIONS_DIR,
                        REPORTS_DIR, SEED, set_global_seed)
from src.io_utils import get_logger, make_keras_dataset


def main():
    import keras
    from src.config import SPLITS_DIR
    from src.models import build_efficientnet_b0
    from src.training import (configure_gpu, evaluate_and_save,
                              make_callbacks_v2)

    log = get_logger("effnet128",
                     logfile=REPORTS_DIR / "efficientnet_b0_128_train.log")
    log.info("=== Reviewer-requested ablation: EfficientNet-B0 @ 128x128, recipe=a ===")
    log.info("GPU: %s", configure_gpu())

    set_global_seed(SEED)

    train_df = pd.read_csv(SPLITS_DIR / "train_clean.csv")
    val_df   = pd.read_csv(SPLITS_DIR / "val_clean.csv")
    test_df  = pd.read_csv(SPLITS_DIR / "test_clean.csv")
    log.info("Splits: train=%d val=%d test=%d",
             len(train_df), len(val_df), len(test_df))

    INPUT_SIZE = 128  # the only knob we change vs config 'a'
    train_ds = make_keras_dataset(train_df, batch_size=BATCH_SIZE, shuffle=True,
                                  augment=False, image_size=INPUT_SIZE,
                                  one_hot=True, num_classes=NUM_CLASSES,
                                  num_workers=2)
    val_ds = make_keras_dataset(val_df, batch_size=BATCH_SIZE, shuffle=False,
                                augment=False, image_size=INPUT_SIZE,
                                one_hot=True, num_classes=NUM_CLASSES,
                                num_workers=1)
    test_ds = make_keras_dataset(test_df, batch_size=BATCH_SIZE, shuffle=False,
                                 augment=False, image_size=INPUT_SIZE,
                                 one_hot=True, num_classes=NUM_CLASSES,
                                 num_workers=1)

    EPOCHS_STAGE1, EPOCHS_STAGE2 = 10, 15
    LR1, LR2 = 1e-3, 1e-4

    # Stage 1 — frozen backbone at 128x128
    model = build_efficientnet_b0(
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        freeze_backbone=True, learning_rate=LR1, weight_decay=1e-4,
        loss="categorical_crossentropy", label_smoothing=0.0,
    )
    log.info("Stage 1 — params=%d", model.count_params())
    cbs1 = make_callbacks_v2(model_name="efficientnet_b0_128_stage1",
                              total_epochs=EPOCHS_STAGE1, initial_lr=LR1,
                              warmup_epochs=2, monitor="val_accuracy", mode="max",
                              patience_es=5)
    t0 = time.time()
    hist1 = model.fit(train_ds, validation_data=val_ds,
                      epochs=EPOCHS_STAGE1, callbacks=cbs1, verbose=2)
    s1 = time.time() - t0

    # Stage 2 — full fine-tune
    head_names = {"image", "augment", "effnet_to_pixels", "gap", "dropout", "probs"}
    for layer in model.layers:
        layer.trainable = layer.name not in head_names
    optimizer = keras.optimizers.AdamW(learning_rate=LR2, weight_decay=1e-4)
    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=0.0)
    model.compile(optimizer=optimizer, loss=loss_fn,
                  metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")])

    cbs2 = make_callbacks_v2(model_name="efficientnet_b0_128",
                              total_epochs=EPOCHS_STAGE2, initial_lr=LR2,
                              warmup_epochs=1, monitor="val_accuracy", mode="max",
                              patience_es=5)
    t1 = time.time()
    hist2 = model.fit(train_ds, validation_data=val_ds,
                      epochs=EPOCHS_STAGE2, callbacks=cbs2, verbose=2)
    s2 = time.time() - t1

    log.info("Stage 1 best val_acc=%.4f, Stage 2 best=%.4f",
             max(hist1.history.get("val_accuracy", [0.0])),
             max(hist2.history.get("val_accuracy", [0.0])))

    metrics = evaluate_and_save(
        model, test_ds, model_name="efficientnet_b0_128",
        history={**hist1.history, **hist2.history,
                 "elapsed_seconds": s1 + s2},
        test_df=test_df,
    )
    log.info("Test acc=%.4f  macroF1=%.4f", metrics["accuracy"], metrics["macro_f1"])

    out = REPORTS_DIR / "effnet_128_ablation_summary.json"
    out.write_text(json.dumps(
        {"input_size": INPUT_SIZE,
         "stage1_seconds": round(s1, 1), "stage2_seconds": round(s2, 1),
         "stage1_best_val_accuracy": max(hist1.history.get("val_accuracy", [0.0])),
         "stage2_best_val_accuracy": max(hist2.history.get("val_accuracy", [0.0])),
         "test_accuracy": metrics["accuracy"],
         "macro_f1": metrics["macro_f1"]},
        indent=2), encoding="utf-8")
    log.info("Saved summary: %s", out)


if __name__ == "__main__":
    main()
