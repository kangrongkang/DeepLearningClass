# Day 3 — Model A: Custom CNN Baseline Report

**Notebook:** `notebooks/03_model_a_baseline_cnn.ipynb`
**Date:** 2026-04-29
**Status:** Notebook complete & smoke-tested on local CPU. **Full GPU training is the user's next step (run on Colab).**

## Goal

Train a small CNN from scratch on the Day 2 splits to set a floor every later model has to beat.

## What was built

| Artifact | Path |
|---|---|
| Notebook | `notebooks/03_model_a_baseline_cnn.ipynb` (executes end-to-end on CPU smoke mode) |
| Model factory | `src/models.py` → `build_baseline_cnn` |
| Shared training loop | `src/training.py` → `train_model`, `evaluate_and_save`, `make_callbacks`, `configure_gpu` |
| Best weights | `outputs/models/baseline_cnn.keras` |
| Test predictions | `outputs/predictions/baseline_cnn_test_predictions.npz` (y_true, y_pred, y_proba) |
| Headline metrics | `outputs/predictions/baseline_cnn_metrics.json` |
| Training curves | `outputs/figures/baseline_cnn_training_curves.png` |
| Confusion matrix | `outputs/figures/baseline_cnn_confusion_matrix.png` |
| Train log | `outputs/reports/baseline_cnn_train.log` |

## Architecture (matches plan Step 3.1)

```
Input (128, 128, 3) float32 in [0, 1]
└─ Conv2D(32, 3x3, same, no_bias) → BatchNorm → ReLU → MaxPool 2x2
└─ Conv2D(64, 3x3, same, no_bias) → BatchNorm → ReLU → MaxPool 2x2
└─ Conv2D(128, 3x3, same, no_bias) → BatchNorm → ReLU → MaxPool 2x2
└─ Conv2D(256, 3x3, same, no_bias) → BatchNorm → ReLU → MaxPool 2x2
└─ GlobalAveragePooling2D
└─ Dense(128, ReLU) → Dropout(0.3)
└─ Dense(4, softmax)
```

Total trainable parameters: **423,268**.
He-normal initialization throughout (matches BN+ReLU statistics).

## Training settings (matches plan Step 3.2)

| Knob | Value |
|---|---|
| optimizer | Adam |
| learning rate | 1e-3 |
| batch size | 32 |
| max epochs | 30 |
| EarlyStopping | `val_accuracy`, patience=5, `restore_best_weights=True`, mode=max |
| ReduceLROnPlateau | `val_accuracy`, factor=0.5, patience=2, min_lr=1e-6, mode=max |
| ModelCheckpoint | best-only, written to `outputs/models/baseline_cnn.keras` |

## SMOKE / FULL toggle

The notebook detects whether a GPU is visible and toggles `SMOKE` automatically:

| Mode | Trigger | epochs | sample fraction | wall-clock |
|---|---|---:|---:|---:|
| SMOKE | no GPU | 2 | 2 % (~615 / 132 / 132) | ~10 s on CPU |
| FULL  | GPU visible | 30 | 100 % (30,799 / 6,600 / 6,601) | ~10-15 min on Colab T4 |

Local smoke result: **acc 0.36, macro F1 0.22 on 132 test samples after 2 epochs on 615 train samples** — the pipeline works; the numbers are uninformative until the full Colab run lands.

## Code review summary (Reviewer pass 3)

The reviewer flagged two MUST-FIX items affecting Days 4-5 reuse:

| Severity | Issue | Fix |
|---|---|---|
| MUST FIX | `Lambda(lambda t: preprocess_input(t * 255.0))` in MobileNetV2 / VGG16 — Lambda serializes the function reference, which breaks `model.save` / `load_model` across kernel restarts. | MobileNetV2 → `Rescaling(scale=2.0, offset=-1.0)` (mathematical equivalent of MobileNetV2's `[0,255]→[-1,1]`). VGG16 → call `vgg16.preprocess_input(x*255)` directly on the symbolic tensor (ops bake into the graph as standard nodes; no Lambda layer). Verified by save/load round-trip — all three models reload with **bit-identical outputs**. |
| MUST FIX | Possible `CLASS_NAMES` ↔ label mismatch if config drifts | `load_split` now asserts that `(class, label)` rows in the CSV match `CLASS_NAMES` order; raises `ValueError` with remediation if they don't. |
| Resolved | VGG16 docstring said "Flatten -> Dense(256)" but code used GAP | Docstring rewritten to explain the GAP choice (Flatten on 4×4×512 → Dense(256) blows up to ~2 M head params and overfits fast on this dataset). |

Five SHOULD-CONSIDER items were noted; minor polish (predict-loop microefficiency, CSVLogger callback, kernel_initializer seed) — left alone for now since they don't affect correctness.

## Round-trip verification

```
baseline_cnn   save/load OK   max output diff = 0.00e+00
mobilenetv2    save/load OK   max output diff = 0.00e+00
vgg16          save/load OK   max output diff = 0.00e+00
```

All three architectures save to `.keras` and reload with bit-identical outputs.

## Limitations carried forward

* Full training was not run locally (no GPU). Numbers in the metrics JSON are smoke-mode only — the user must re-run on Colab GPU before Days 4-6 are meaningful.
* Image-level split → augmented dataset → optimistic accuracy — repeated in the notebook intro and final report.

## Next

**Task 5 — Day 4: Model B (MobileNetV2).** Two-stage training: frozen backbone (lr 1e-3, 10-15 epochs) → fine-tune last 20-30 layers (lr 1e-5, 5-10 epochs). Reuses `train_model` / `evaluate_and_save` from Day 3.
