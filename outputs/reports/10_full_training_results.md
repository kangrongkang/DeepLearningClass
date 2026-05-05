# Full Training Results — Local RTX 5080

**Date:** 2026-04-30 (training completed at 08:04 local time)
**Total wall-clock:** 6h 19min on local NVIDIA GeForce RTX 5080 (Blackwell, sm_120)
**Stack:** Keras 3.14 + PyTorch 2.11 (cu128), native Windows GPU
**Test set:** 6,601 images, fully held-out

## Headline numbers

| Model | Test accuracy | Macro F1 | Macro precision | Macro recall | Epochs run | Wall-clock |
|---|---:|---:|---:|---:|---:|---:|
| **baseline_cnn** | **0.9053** | **0.9084** | 0.9104 | 0.9080 | 30 (no early stop) | 3 h 14 min |
| mobilenetv2 | 0.7426 | 0.7430 | 0.7511 | 0.7467 | 20 (early-stopped) | 1 h 51 min |
| vgg16 | 0.7290 | 0.7348 | 0.7391 | 0.7332 | 15 (early-stopped) | 1 h 11 min |

> **The from-scratch CNN beat both transfer-learning models by ~17 percentage
> points on test accuracy.** This is the *opposite* of the project plan's
> prediction. It is the most interesting finding of the project and deserves
> careful discussion in the report.

## Per-class F1 — adjacent severity levels are the hardest

| Class | baseline_cnn | mobilenetv2 | vgg16 |
|---|---:|---:|---:|
| ModerateDemented | 0.991 | 0.956 | 0.947 |
| MildDemented | 0.935 | 0.728 | 0.717 |
| NonDemented | 0.887 | 0.740 | 0.714 |
| **VeryMildDemented** | **0.820** | **0.548** | **0.561** |

* **ModerateDemented** is easy for everyone — visually distinct, F1 ≥ 0.95 across all three models.
* **VeryMildDemented** is hardest for the transfer models (F1 ~0.55) because
  the distinguishing features (subtle hippocampal atrophy, mild ventricular
  enlargement) are not in ImageNet's repertoire.
* The **VeryMildDemented vs NonDemented** confusion the plan predicted as the
  hardest pair is exactly what shows up — see normalized confusion below.

## Confusion matrices (row-normalized, true → predicted)

### baseline_cnn
```
            NonD   VeryM  Mild   Moder
NonD        0.919  0.071  0.010  0.000
VeryMild    0.161  0.785  0.054  0.000
Mild        0.012  0.042  0.946  0.000
Moderate    0.004  0.011  0.003  0.982
```
Largest confusion: **VeryMild → NonDemented (16.1%)** — the predicted hardest pair.

### mobilenetv2
```
            NonD   VeryM  Mild   Moder
NonD        0.800  0.098  0.100  0.002
VeryMild    0.324  0.472  0.198  0.006
Mild        0.091  0.117  0.787  0.005
Moderate    0.008  0.038  0.026  0.928
```
Largest confusion: **VeryMild → NonDemented (32.4%)** — same pair, twice as bad as baseline.

### vgg16
```
            NonD   VeryM  Mild   Moder
NonD        0.754  0.161  0.083  0.002
VeryMild    0.298  0.530  0.163  0.009
Mild        0.116  0.146  0.735  0.003
Moderate    0.009  0.051  0.027  0.914
```
Largest confusion: **VeryMild → NonDemented (29.8%)** — same pair, slightly better than MobileNetV2.

## Why did transfer learning underperform?

Three plausible explanations, ordered by how much they likely contribute:

1. **Domain gap.** ImageNet pretraining encodes filters tuned to natural-photo
   gradients, edges, and textures (dogs, cars, food). Brain MRI has very
   different low- and mid-level statistics: grayscale-dominated intensity,
   smooth gradients, anatomical symmetry, no color cues. The pretrained
   conv stack is at best partially useful, at worst actively misleading. The
   custom CNN, with no preconceptions, learned MRI-native filters directly.
2. **Augmented + upsampled dataset, image-level split.** The dataset
   description says the publisher already augmented and upsampled the images.
   If we split at the image level (we have to — the dataset doesn't expose
   patient or scan IDs), near-duplicates can leak across train / val / test.
   A small, high-capacity model trained from scratch can fit those leaked
   duplicates more aggressively than a frozen-backbone transfer model. The
   gap we measure here is therefore an *upper bound* of the real
   architecture difference, not a clean comparison.
3. **128×128 inputs are smaller than the transfer models expect.** MobileNetV2
   and VGG16 were trained at 224×224. At 128×128 the receptive fields they
   learned are partly mis-scaled. The custom CNN was designed for 128×128
   from the ground up, so its receptive field math is exactly right.

The right framing in the discussion is: **"For this dataset, a small,
domain-specific CNN trained from scratch outperforms ImageNet-pretrained
backbones — likely a combination of domain mismatch and dataset-level
duplication risk."** This is a defensible, nuanced finding that engages with
the literature instead of just reporting numbers.

## Training-curve observations

* **baseline_cnn** ran the full 30 epochs without triggering EarlyStopping —
  it was still improving at the cap. Suggests room for more epochs or higher
  capacity if we wanted to push further.
* **mobilenetv2** stage 1 (15 epochs frozen) was followed by stage 2 fine-tuning
  on the top 20 layers. Fine-tuning helped modestly; the comparison logic in
  the notebook picked stage 2's weights (best val_acc ≈ 0.74).
* **vgg16** early-stopped at 15 epochs (frozen-only path; `DO_FINETUNE=False`).
  Per the plan, fine-tuning VGG16 on this dataset is risky, so we did not
  enable it.

## How this updates the existing reports

`outputs/reports/00_final_report.md` was auto-regenerated from the artifacts
above. It already contains:

- The corrected headline-metrics table
- Per-class F1 with VeryMildDemented as the hardest class for transfer models
- The "biggest confusion" sentences for each model (auto-extracted)
- Section 9 (Discussion & Limitations) with the augmented + upsampled caveat,
  image-level split caveat, 2D-vs-3D caveat, and benchmark-comparison framing

The Discussion section in the final report should be lightly updated to
include the surprising-finding framing above. I'm leaving the existing
auto-generated text intact so the discussion notebook doesn't drift, and
captures the surprising finding here for the *manuscript* version of the
report.

## Reviewer-audit checklist (final pass)

| Item | Status |
|---|---|
| All 8 notebooks executed end-to-end on local GPU | ✓ |
| Three .keras best-weights files saved | ✓ |
| Three metrics JSONs with full classification reports | ✓ |
| Three test-prediction npz files (with relpaths) | ✓ |
| Three training-curve PNGs | ✓ |
| Three confusion-matrix PNGs | ✓ |
| Three Grad-CAM panels with mixed correct/wrong examples | ✓ |
| Comparison report auto-built from artifacts | ✓ |
| Final report auto-built from artifacts | ✓ |
| All four required limitation sentences present | ✓ |
| Reproducibility checklist in final report | ✓ |

## Project closing note

The 9-day plan from `project plan.docx` is fully executed. Day 1-9 deliverables
are in `notebooks/01..08*.ipynb` and `outputs/`, and the entire pipeline now
runs on native Windows GPU (RTX 5080) via Keras 3 + PyTorch backend in roughly
6 hours end-to-end.

The most scientifically interesting finding — custom CNN outperforming
transfer learning — is documented here and should be the centerpiece of the
manuscript discussion.
