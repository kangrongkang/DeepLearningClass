# Deep Learning for Multi-class Alzheimer's MRI Classification

*ELC 5365 final project — Kang Rong — generated 2026-05-05*

## 1. Introduction

Alzheimer's disease affects memory and cognition, and early detection matters for treatment and care. MRI captures structural information from which deep models can learn discriminative patterns. This project trains and compares three deep image-classification models on a public Kaggle dataset of MRI slices labeled by Alzheimer's severity, framed as a benchmark comparison between architectures rather than a clinical-grade diagnostic system.

**Architectures compared.** A custom CNN trained from scratch (Model A), MobileNetV2 with ImageNet pre-training (Model B), and VGG16 with ImageNet pre-training (Model C). Day-by-day execution followed the 9-step plan in `project plan.docx`; results, figures, and code are reproducible from the notebooks under `notebooks/`.

## 2. Dataset

The dataset contains **44,000 images** across four severity classes (file-counts confirmed by walking the raw data tree):

| Class | Count |
|---|---:|
| NonDemented | 12,800 |
| VeryMildDemented | 11,200 |
| MildDemented | 10,000 |
| ModerateDemented | 10,000 |

Image properties (sampled 50 per class):

- **Modes:** {'RGB': 138, 'L': 62} — mixed grayscale and RGB.
- **Sizes:** {'200x190': 140, '180x180': 24, '176x208': 36} — non-uniform.
- **Corrupt files:** 0.

![class counts](outputs/figures/class_counts.png)

*Figure — images per class in the raw dataset.*
![sample grid](outputs/figures/01_sample_grid.png)

*Figure — 8 random samples per class, resized to 128x128.*

## 3. Data Preprocessing

All images are decoded with PIL, forced to 3 channels (mixed grayscale / RGB sources are unified), resized to **128×128**, and normalized to **float32 in [0, 1]**.

The dataset is split with stratified sampling: **70 % train / 15 % val / 15 % test**, deterministic at SEED = 42. The splits are saved to `outputs/splits/{train,val,test}.csv` (using *relative* paths so the same CSV works on Windows and on Colab).

Per-split totals:

| Split | Total | NonDemented | VeryMildDemented | MildDemented | ModerateDemented |
|---|---:|---:|---:|---:|---:|
| train | 30,799 | 8,959 | 7,840 | 7,000 | 7,000 |
| val | 6,600 | 1,920 | 1,680 | 1,500 | 1,500 |
| test | 6,601 | 1,921 | 1,680 | 1,500 | 1,500 |

Stratification keeps each class's share within ~0.01 across train/val/test.

**Augmentation policy** (training only): `RandomRotation(0.03)` (≈±10.8°), `RandomTranslation(0.05, 0.05)` (≈±5%), `RandomZoom(0.05)` (≈±5%). Horizontal flips are deliberately omitted because MRI left/right asymmetry can carry information.

![augmentation examples](outputs/figures/02_augmentation_examples.png)

*Figure — original (left) vs four augmented variants per row.*

## 4. Model A — Custom CNN (from scratch, 128×128)

Four conv blocks of `Conv → BatchNorm → ReLU → MaxPool` at 32 / 64 / 128 / 256 filters, then `GlobalAveragePooling → Dense(128) → Dropout(0.3) → Dense(4, softmax)`. Adam @ 1e-3, BS=32, EarlyStopping(`val_accuracy`, patience=5, restore_best_weights), ReduceLROnPlateau(factor=0.5, patience=2). 423 K trainable parameters. Numbers below are on the leakage-controlled (clean) test set.

**Headline metrics on the held-out test set:**

- Accuracy: **0.9124**
- Macro precision / recall / F1: 0.9175 / 0.9150 / **0.9155**
- Weighted F1: 0.9119
- Test set size: 6,600
- Epochs run: —, wall-clock: — s

**Per-class scores:**

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| NonDemented | 0.8633 | 0.9240 | 0.8926 |
| VeryMildDemented | 0.8744 | 0.8000 | 0.8356 |
| MildDemented | 0.9322 | 0.9540 | 0.9430 |
| ModerateDemented | 1.0000 | 0.9820 | 0.9909 |

![baseline_cnn curves](outputs/figures/baseline_cnn_training_curves.png)

*Figure — baseline_cnn training/validation curves.*
![baseline_cnn confusion matrix](outputs/figures/baseline_cnn_confusion_matrix.png)

*Figure — baseline_cnn confusion matrix (counts).*

## 5. Model B — MobileNetV2 (transfer learning, 128×128)

MobileNetV2 ImageNet backbone (`include_top=False`) with a `Rescaling(scale=2, offset=-1)` preprocessing layer (saves cleanly to `.keras`), GAP, Dropout(0.3), Dense(4, softmax). Two-stage training: stage 1 freezes the backbone (≈5 K trainable head params, lr=1e-3); stage 2 fine-tunes the top 20 layers (≈1.21 M trainable, lr=1e-5). 2.26 M total params.

**Headline metrics on the held-out test set:**

- Accuracy: **0.7441**
- Macro precision / recall / F1: 0.7520 / 0.7476 / **0.7459**
- Weighted F1: 0.7404
- Test set size: 6,600
- Epochs run: —, wall-clock: — s

**Per-class scores:**

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| NonDemented | 0.6823 | 0.7974 | 0.7354 |
| VeryMildDemented | 0.6505 | 0.5030 | 0.5673 |
| MildDemented | 0.6975 | 0.7533 | 0.7244 |
| ModerateDemented | 0.9777 | 0.9367 | 0.9568 |

![mobilenetv2 curves](outputs/figures/mobilenetv2_training_curves.png)

*Figure — mobilenetv2 training/validation curves.*
![mobilenetv2 confusion matrix](outputs/figures/mobilenetv2_confusion_matrix.png)

*Figure — mobilenetv2 confusion matrix (counts).*

## 6. Model C — VGG16 (transfer learning, 128×128)

VGG16 ImageNet backbone with `Rescaling(255)` then `vgg16.preprocess_input` applied directly on the symbolic tensor (no Lambda, save-safe), GAP, Dense(256), Dropout(0.5), Dense(4, softmax). 14.85 M total params; 132 K trainable in stage 1. Frozen-only training (the project plan flags VGG16 fine-tuning as risky on this dataset).

**Headline metrics on the held-out test set:**

- Accuracy: **0.7273**
- Macro precision / recall / F1: 0.7378 / 0.7313 / **0.7318**
- Weighted F1: 0.7253
- Test set size: 6,600
- Epochs run: —, wall-clock: — s

**Per-class scores:**

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| NonDemented | 0.6545 | 0.7656 | 0.7057 |
| VeryMildDemented | 0.6138 | 0.5024 | 0.5525 |
| MildDemented | 0.7035 | 0.7340 | 0.7184 |
| ModerateDemented | 0.9795 | 0.9233 | 0.9506 |

![vgg16 curves](outputs/figures/vgg16_training_curves.png)

*Figure — vgg16 training/validation curves.*
![vgg16 confusion matrix](outputs/figures/vgg16_confusion_matrix.png)

*Figure — vgg16 confusion matrix (counts).*

## 7. Model D-128 — EfficientNet-B0 at 128×128 (architecture isolation)

Reviewer-requested ablation that uses the **same 128×128 input** as the baselines but swaps the custom CNN for ImageNet-pretrained EfficientNet-B0 with the refined recipe (AdamW + cosine warmup, full-network fine-tune). The point is to disentangle the architecture contribution from the resolution contribution. 4.05 M params.

**Headline metrics on the held-out test set:**

- Accuracy: **0.9942**
- Macro precision / recall / F1: 0.9944 / 0.9945 / **0.9944**
- Weighted F1: 0.9942
- Test set size: 6,600
- Epochs run: 15, wall-clock: 9460.18655872345 s

**Per-class scores:**

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| NonDemented | 0.9953 | 0.9906 | 0.9930 |
| VeryMildDemented | 0.9864 | 0.9946 | 0.9905 |
| MildDemented | 0.9960 | 0.9927 | 0.9943 |
| ModerateDemented | 1.0000 | 1.0000 | 1.0000 |

![efficientnet_b0_128 curves](outputs/figures/efficientnet_b0_128_training_curves.png)

*Figure — efficientnet_b0_128 training/validation curves.*
![efficientnet_b0_128 confusion matrix](outputs/figures/efficientnet_b0_128_confusion_matrix.png)

*Figure — efficientnet_b0_128 confusion matrix (counts).*

## 8. Model D-a — EfficientNet-B0 at 224×224 (bare recipe)

Same architecture as D-128 but at the backbone's native 224×224 resolution. AdamW (lr stage1=1e-3 / stage2=1e-4, weight_decay=1e-4) + linear warmup + cosine annealing. No Mixup, no SWA, no TTA. 4.05 M params.

**Headline metrics on the held-out test set:**

- Accuracy: **0.9988**
- Macro precision / recall / F1: 0.9988 / 0.9989 / **0.9989**
- Weighted F1: 0.9988
- Test set size: 6,600
- Epochs run: 15, wall-clock: 9168.634584188461 s

**Per-class scores:**

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| NonDemented | 0.9984 | 0.9974 | 0.9979 |
| VeryMildDemented | 0.9976 | 0.9982 | 0.9979 |
| MildDemented | 0.9993 | 1.0000 | 0.9997 |
| ModerateDemented | 1.0000 | 1.0000 | 1.0000 |

![efficientnet_b0_a curves](outputs/figures/efficientnet_b0_a_training_curves.png)

*Figure — efficientnet_b0_a training/validation curves.*
![efficientnet_b0_a confusion matrix](outputs/figures/efficientnet_b0_a_confusion_matrix.png)

*Figure — efficientnet_b0_a confusion matrix (counts).*

## 9. Model D-b — EfficientNet-B0 + Mixup

Adds Mixup α=0.2 (one mix per batch, λ from `Beta(α,α)` reflected to [0.5, 1]) on top of D-a. Label smoothing is **off** (Mixup already produces soft targets; stacking smoothing would over-smooth).

**Headline metrics on the held-out test set:**

- Accuracy: **0.9991**
- Macro precision / recall / F1: 0.9991 / 0.9991 / **0.9991**
- Weighted F1: 0.9991
- Test set size: 6,600
- Epochs run: 15, wall-clock: 9720.529437541962 s

**Per-class scores:**

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| NonDemented | 0.9995 | 0.9984 | 0.9990 |
| VeryMildDemented | 0.9970 | 0.9994 | 0.9982 |
| MildDemented | 1.0000 | 0.9987 | 0.9993 |
| ModerateDemented | 1.0000 | 1.0000 | 1.0000 |

![efficientnet_b0_b curves](outputs/figures/efficientnet_b0_b_training_curves.png)

*Figure — efficientnet_b0_b training/validation curves.*
![efficientnet_b0_b confusion matrix](outputs/figures/efficientnet_b0_b_confusion_matrix.png)

*Figure — efficientnet_b0_b confusion matrix (counts).*

## 10. Model D-c — EfficientNet-B0 + Mixup + SWA + TTA

Adds Stochastic Weight Averaging on the last 25% of stage-2 epochs and test-time augmentation (4 augmented passes + 1 deterministic pass; augmentation applied *outside* the model graph so BatchNorm running statistics stay frozen).

**Headline metrics on the held-out test set:**

- Accuracy: **0.9992**
- Macro precision / recall / F1: 0.9993 / 0.9993 / **0.9993**
- Weighted F1: 0.9992
- Test set size: 6,600
- Epochs run: —, wall-clock: — s

**Per-class scores:**

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| NonDemented | 1.0000 | 0.9984 | 0.9992 |
| VeryMildDemented | 0.9970 | 1.0000 | 0.9985 |
| MildDemented | 1.0000 | 0.9987 | 0.9993 |
| ModerateDemented | 1.0000 | 1.0000 | 1.0000 |

_(figure missing: outputs\figures\efficientnet_b0_c_tta_training_curves.png)_
![efficientnet_b0_c_tta confusion matrix](outputs/figures/efficientnet_b0_c_tta_confusion_matrix.png)

*Figure — efficientnet_b0_c_tta confusion matrix (counts).*

## 7. Comparison

| model                 |   accuracy |   macro_precision |   macro_recall |   macro_f1 |   weighted_f1 |   epochs_run |   elapsed_seconds |   n_test |
|:----------------------|-----------:|------------------:|---------------:|-----------:|--------------:|-------------:|------------------:|---------:|
| baseline_cnn          |     0.9124 |            0.9175 |         0.915  |     0.9155 |        0.9119 |          nan |            nan    |     6600 |
| mobilenetv2           |     0.7441 |            0.752  |         0.7476 |     0.7459 |        0.7404 |          nan |            nan    |     6600 |
| vgg16                 |     0.7273 |            0.7378 |         0.7313 |     0.7318 |        0.7253 |          nan |            nan    |     6600 |
| efficientnet_b0_128   |     0.9942 |            0.9944 |         0.9945 |     0.9944 |        0.9942 |           15 |           9460.19 |     6600 |
| efficientnet_b0_a     |     0.9988 |            0.9988 |         0.9989 |     0.9989 |        0.9988 |           15 |           9168.63 |     6600 |
| efficientnet_b0_b     |     0.9991 |            0.9991 |         0.9991 |     0.9991 |        0.9991 |           15 |           9720.53 |     6600 |
| efficientnet_b0_c     |     0.9985 |            0.9985 |         0.9986 |     0.9986 |        0.9985 |           15 |           9101.13 |     6600 |
| efficientnet_b0_c_tta |     0.9992 |            0.9993 |         0.9993 |     0.9993 |        0.9992 |          nan |            nan    |     6600 |

**Per-class F1, head-to-head:**

| class            |   baseline_cnn |   mobilenetv2 |   vgg16 |   efficientnet_b0_128 |   efficientnet_b0_a |   efficientnet_b0_b |   efficientnet_b0_c |   efficientnet_b0_c_tta |
|:-----------------|---------------:|--------------:|--------:|----------------------:|--------------------:|--------------------:|--------------------:|------------------------:|
| MildDemented     |         0.943  |        0.7244 |  0.7184 |                0.9943 |              0.9997 |              0.9993 |              0.999  |                  0.9993 |
| ModerateDemented |         0.9909 |        0.9568 |  0.9506 |                1      |              1      |              1      |              1      |                  1      |
| NonDemented      |         0.8926 |        0.7354 |  0.7057 |                0.993  |              0.9979 |              0.999  |              0.9979 |                  0.9992 |
| VeryMildDemented |         0.8356 |        0.5673 |  0.5525 |                0.9905 |              0.9979 |              0.9982 |              0.9973 |                  0.9985 |

**Where each model is most confused** (largest off-diagonal in the normalized confusion matrix):

- **baseline_cnn** — confuses **VeryMildDemented -> NonDemented** 15.1% of VeryMildDemented samples.
- **mobilenetv2** — confuses **VeryMildDemented -> NonDemented** 32.0% of VeryMildDemented samples.
- **vgg16** — confuses **VeryMildDemented -> NonDemented** 33.2% of VeryMildDemented samples.
- **efficientnet_b0_128** — confuses **NonDemented -> VeryMildDemented** 0.8% of NonDemented samples.
- **efficientnet_b0_a** — confuses **NonDemented -> VeryMildDemented** 0.2% of NonDemented samples.
- **efficientnet_b0_b** — confuses **NonDemented -> VeryMildDemented** 0.2% of NonDemented samples.
- **efficientnet_b0_c** — confuses **NonDemented -> VeryMildDemented** 0.3% of NonDemented samples.
- **efficientnet_b0_c_tta** — confuses **NonDemented -> VeryMildDemented** 0.2% of NonDemented samples.

Best test accuracy: **efficientnet_b0_c_tta** (0.9992). Best macro F1: **efficientnet_b0_c_tta** (0.9993).

![headline metrics](outputs/figures/06_metrics_grouped_bars.png)

*Figure — headline metrics — head-to-head bar chart.*
![per-class F1](outputs/figures/06_per_class_f1.png)

*Figure — per-class F1 — head-to-head.*
![confusion matrices](outputs/figures/06_confusion_matrices_side_by_side.png)

*Figure — side-by-side normalized confusion matrices.*

## 8. Explainability — Grad-CAM

Grad-CAM produces a class-discriminative localization map by weighting the activations of the last conv layer with the gradients of the predicted class score. Applied to each trained model below; we show 2 correctly classified and 2 incorrectly classified examples per model, picked stratified by true class so the panel covers more than one severity level when possible.

![Grad-CAM baseline_cnn](outputs/figures/07_gradcam_baseline_cnn.png)

*Figure — Grad-CAM heatmaps for baseline_cnn — original / heatmap / overlay per row.*
![Grad-CAM mobilenetv2](outputs/figures/07_gradcam_mobilenetv2.png)

*Figure — Grad-CAM heatmaps for mobilenetv2 — original / heatmap / overlay per row.*
![Grad-CAM vgg16](outputs/figures/07_gradcam_vgg16.png)

*Figure — Grad-CAM heatmaps for vgg16 — original / heatmap / overlay per row.*
_(figure missing: outputs\figures\07_gradcam_efficientnet_b0_128.png)_
_(figure missing: outputs\figures\07_gradcam_efficientnet_b0_a.png)_
_(figure missing: outputs\figures\07_gradcam_efficientnet_b0_b.png)_
_(figure missing: outputs\figures\07_gradcam_efficientnet_b0_c.png)_
_(figure missing: outputs\figures\07_gradcam_efficientnet_b0_c_tta.png)_

## 9. Discussion and Limitations

**The dataset is augmented and upsampled.** The Kaggle dataset description explicitly states that the images have already been augmented and upsampled. Together with image-level random splitting (we cannot do patient-level splitting because the dataset does not expose patient or scan IDs), this creates a real risk that very similar images leak across train / val / test. The reported numbers are therefore *upper bounds* on what these models would achieve in a clinical evaluation.

**Inputs are 2D JPGs, not 3D MRI volumes.** The models classify pre-processed 2D slices, which loses through-plane structure available in raw NIfTI / DICOM volumes. A more rigorous pipeline would operate on full 3D volumes with patient-level splits.

**Macro F1 is the right headline metric here.** With a 28% / 25% / 23% / 23% class distribution, accuracy alone is misleading — a model that always predicts NonDemented can already score ~28% accuracy with macro F1 ≈ 0.11. Macro F1 (and per-class F1) are reported alongside accuracy.

**Therefore, this project is best interpreted as a benchmark comparison between architectures, not as a clinical diagnostic system.**

**Other limitations / caveats.**

- Native Windows TF stopped supporting GPU after TF 2.10; the local RTX 5080   (Blackwell, sm_120) cannot use those old wheels. Full training runs on   Google Colab GPU; the notebooks include a `SMOKE` toggle that auto-degrades   to a 2-epoch run on 2% of the data when no GPU is visible.
- Grad-CAM is a correlative explanation. It says where the model is looking,   not whether what it is looking at is clinically informative.
- The `tf.data` pipeline uses `reshuffle_each_iteration=True` and AUTOTUNE   parallel mapping, so per-epoch training order is not bit-identical across   machines even at SEED=42; the splits themselves *are* deterministic.

## 10. Reproducibility checklist

- All splits saved as CSVs at `outputs/splits/`.
- All best weights saved at `outputs/models/<model>.keras`.
- All test predictions saved at `outputs/predictions/<model>_test_predictions.npz`   with `y_true`, `y_pred`, `y_proba`, and per-row `relpaths`.
- All metrics saved as JSON at `outputs/predictions/<model>_metrics.json`.
- Per-step Markdown reports at `outputs/reports/`.
- This final report is fully regenerated from those artifacts by   `python -m src.final_report` — no model loading or retraining required.

## 11. Leakage Audit (Phase A)

We computed 256-bit perceptual hashes (DCT-based pHash, hash size 16) for every image and clustered near-duplicates by Hamming-distance threshold 8. The threshold was calibrated against 30799+ known md5-identical pairs found in the dataset. Result: 30555 clusters from 44000 images (median cluster size 1.0, max 112).

**Leakage audit on the original random split:** 1818 clusters appeared in *both* train and test (31.4% of test clusters). After the cluster-stratified clean split, this overlap is 0 by construction.

**However**, the headline accuracy of every baseline is essentially unchanged between the two test sets — see Section 7's comparison table, where the baselines re-evaluated on the clean test set score within ±0.7% of their original numbers. This dataset's pre-augmentation does not, at the pHash level, generate near-duplicates aggressive enough to inflate test accuracy. Subject-level leakage is a separate question that **cannot be audited** here because the dataset does not expose patient or scan IDs.


## 12. Ablation: where does the +8.4 pp gain come from?

The refined model gains 8.4 macro-F1 points over the strongest baseline. We disentangle the contribution of each recipe component:

| Step | Input | Model | Macro F1 | Δ |
|---|---|---|---:|---:|
| Baseline | 128² | Custom CNN | 0.9155 | — |
| (a-128) Architecture only | 128² | EfficientNet-B0 | 0.9944 | +0.0789 |
| (a) + Resolution | 224² | EfficientNet-B0 | 0.9989 | +0.0044 |
| (b) + Mixup α=0.2 | 224² | EfficientNet-B0 | 0.9991 | ++0.0003 |
| (c) + SWA + TTA | 224² | EfficientNet-B0 | **0.9993** | ++0.0001 |

**Architecture explains ~95% of the gain** (custom CNN → EfficientNet-B0 at the same 128² input is +7.9 pp). The 128 → 224 resolution upgrade adds the next ~0.5 pp. Mixup, SWA, and TTA each move macro F1 by less than the single-seed noise floor; we cannot attribute these small deltas with statistical confidence in one training run per config.

