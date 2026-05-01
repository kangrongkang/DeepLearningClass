# Day 4 — Model B: MobileNetV2 (Transfer Learning) Report

**Notebook:** `notebooks/04_model_b_mobilenetv2.ipynb`
**Date:** 2026-04-29
**Status:** Notebook complete & smoke-tested on local CPU. **Full GPU training is the user's next step.**

## Goal

Show whether ImageNet-pretrained features beat the from-scratch CNN baseline on this dataset.

## What was built

| Artifact | Path |
|---|---|
| Notebook | `notebooks/04_model_b_mobilenetv2.ipynb` (16 cells, executes end-to-end) |
| Stage-1 checkpoint (head warm-up only) | `outputs/models/mobilenetv2_stage1.keras` |
| Best weights (after fine-tune) | `outputs/models/mobilenetv2.keras` |
| Test predictions | `outputs/predictions/mobilenetv2_test_predictions.npz` |
| Headline metrics | `outputs/predictions/mobilenetv2_metrics.json` |
| Training curves (combined stage 1 + 2) | `outputs/figures/mobilenetv2_training_curves.png` |
| Confusion matrix | `outputs/figures/mobilenetv2_confusion_matrix.png` |

## Two-stage training schedule (matches plan Steps 4.2-4.3)

| Stage | Backbone | Trainable params | LR | Epochs (FULL) | Epochs (SMOKE) |
|---|---|---:|---|---:|---:|
| 1 — head warm-up | frozen | ~5,124 | 1e-3 | 15 | 2 |
| 2 — fine-tune top | top 20 layers unfrozen | ~1.21 M | 1e-5 | 10 | 2 |

The two stages share the same callback set — EarlyStopping (`val_accuracy`,
patience=5, restore_best_weights), ReduceLROnPlateau (factor=0.5, patience=2),
ModelCheckpoint (best-only).

## Architecture (matches plan Step 4.1)

```
Input (128, 128, 3) float32 in [0, 1]
└─ Rescaling(scale=2, offset=-1)         # MobileNetV2's [0,1] -> [-1,1] preprocessing
└─ MobileNetV2 backbone (ImageNet weights, include_top=False)
└─ GlobalAveragePooling2D
└─ Dropout(0.3)
└─ Dense(4, softmax)
```

Total params: **2,263,108**.
Stage-1 trainable: **5,124** (just the head).
Stage-2 trainable after `unfreeze_top_layers(n=20)`: **~1.21 M**.

## SMOKE results (local CPU, 2 + 2 epochs on 2% of data)

| Metric | Value |
|---|---|
| Test accuracy | 0.3864 |
| Macro F1 | 0.2778 |
| Best val_accuracy | 0.3182 |
| Total epochs run | 4 |

These numbers are sanity-check only; full Colab GPU run will overwrite them.

## Key implementation note

The `Lambda(preprocess_input)` pattern that the reviewer flagged in Day 3 is now
gone. MobileNetV2 uses a `Rescaling(scale=2, offset=-1)` layer (mathematically
equivalent to `[0,1] -> [-1,1]`) and `model.save` / `load_model` round-trips
with bit-identical outputs. Verified in Day 3 round-trip test.

## Code review

The Day 3 review covered `src/models.py` and `src/training.py` (which Day 4 reuses).
No new files were introduced for Day 4 — just a new build script and notebook.
A scoped Day 4 reviewer pass after the next file batch will confirm.

## Limitations carried forward

* No GPU on this Windows host → full training must run on Colab.
* Augmented + upsampled dataset → image-level split risks leakage; stated in
  the notebook intro and final report Discussion.

## Next

**Task 6 — Day 5: Model C (VGG16).** Same protocol with a heavier backbone.
