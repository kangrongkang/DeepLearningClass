# Day 5 — Model C: VGG16 (Transfer Learning) Report

**Notebook:** `notebooks/05_model_c_vgg16.ipynb`
**Date:** 2026-04-29
**Status:** Notebook complete & smoke-tested on local CPU. **Full GPU training is the user's next step.**

## Goal

Add a heavier, very widely cited transfer-learning backbone so the final report
can be framed as a comparison study (CNN-from-scratch vs MobileNetV2 vs VGG16).

## What was built

| Artifact | Path |
|---|---|
| Notebook | `notebooks/05_model_c_vgg16.ipynb` (16 cells, executes end-to-end) |
| Stage-1 checkpoint | `outputs/models/vgg16_stage1.keras` |
| Best weights (frozen-only by default) | `outputs/models/vgg16.keras` |
| Test predictions | `outputs/predictions/vgg16_test_predictions.npz` |
| Headline metrics | `outputs/predictions/vgg16_metrics.json` |
| Training curves | `outputs/figures/vgg16_training_curves.png` |
| Confusion matrix | `outputs/figures/vgg16_confusion_matrix.png` |

## Architecture (matches plan Steps 5.1-5.2 with the GAP option)

```
Input (128, 128, 3) float32 in [0, 1]
└─ Rescaling(scale=255)                       # [0,1] -> [0,255]
└─ vgg16.preprocess_input (RGB->BGR + ImageNet mean subtraction; not Lambda-wrapped)
└─ VGG16 backbone (ImageNet weights, include_top=False)
└─ GlobalAveragePooling2D
└─ Dense(256, ReLU)
└─ Dropout(0.5)                               # higher than MobileNetV2 — VGG overfits faster
└─ Dense(4, softmax)
```

Total params: **14,847,044**.
Stage-1 trainable: **132,356** (just the head — `Dense(256) + Dense(4)`).

## Training schedule

| Stage | Backbone | LR | Epochs (FULL) | Epochs (SMOKE) | Default |
|---|---|---|---:|---:|---|
| 1 — head only | frozen | 1e-3 | 15 | 1 | always run |
| 2 — fine-tune top 8 (last conv block) | unfrozen | 1e-5 | 8 | 1 | **off** by default (`DO_FINETUNE = False`) |

The plan explicitly calls Stage 2 *optional* for VGG16 because of overfit risk.
The notebook ships with `DO_FINETUNE = False`; the user can flip it to `True`
after seeing the Stage-1 val curve.

## Architectural choice — Flatten vs GAP

The plan permits either `Flatten -> Dense(256) -> Dropout` or
`GlobalAveragePooling -> Dense(256) -> Dropout`. We chose **GAP** for two reasons:

1. Flatten on a 4×4×512 feature map gives an 8,192-dim vector → Dense(256) becomes
   ~2 M parameters in the head alone. With GAP this drops to 131 K parameters.
2. Per the plan, "VGG16 is more prone to overfitting" — a smaller head reduces
   that risk.

This decision is documented in the model factory's docstring and the notebook intro.

## SMOKE results (local CPU, 1 epoch, 2% of data, frozen backbone only)

| Metric | Value |
|---|---|
| Test accuracy | 0.4015 |
| Macro F1 | 0.3507 |
| Total epochs run | 1 |

Uninformative scientifically; full Colab GPU run will overwrite.

## Comparison snapshot (smoke mode — illustrative only)

| Model | Acc | Macro F1 | Epochs | n_test |
|---|---:|---:|---:|---:|
| baseline_cnn | 0.2803 | 0.1943 | 2 | 132 |
| mobilenetv2 | 0.3864 | 0.2778 | 4 | 132 |
| vgg16 | 0.4015 | 0.3507 | 1 | 132 |

After full Colab GPU runs, the comparison notebook (Day 6) will produce the real
table; ordering may shift.

## Code review

Day 3 covered `src/models.py` (`build_vgg16` was reviewed as part of that pass —
the docstring drift was already fixed and the `Lambda(preprocess_input)` swap
was verified by the bit-identical save/load round-trip).
No new files for Day 5 — just the build script and notebook. A short Day 5
reviewer pass after the next batch will confirm the notebook flow.

## Next

**Task 7 — Day 6: Model Comparison.** Load all three models' metrics + preds,
build a single comparison table, render side-by-side confusion matrices, and
note observations (e.g. which classes the models confuse).
