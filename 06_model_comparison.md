# Day 6 — Model Comparison

## Headline metrics

| model                 |   accuracy |   macro_precision |   macro_recall |   macro_f1 |   weighted_f1 |   epochs_run |   elapsed_seconds |   n_test |
|:----------------------|-----------:|------------------:|---------------:|-----------:|--------------:|-------------:|------------------:|---------:|
| baseline_cnn          |     0.9124 |            0.9175 |         0.915  |     0.9155 |        0.9119 |          nan |            nan    |     6600 |
| mobilenetv2           |     0.7441 |            0.752  |         0.7476 |     0.7459 |        0.7404 |          nan |            nan    |     6600 |
| vgg16                 |     0.7273 |            0.7378 |         0.7313 |     0.7318 |        0.7253 |          nan |            nan    |     6600 |
| efficientnet_b0_128   |     0.9942 |            0.9944 |         0.9945 |     0.9944 |        0.9942 |           15 |           9460.19 |     6600 |
| efficientnet_b0_a     |     0.9988 |            0.9988 |         0.9989 |     0.9989 |        0.9988 |           15 |           9168.63 |     6600 |
| efficientnet_b0_b     |     0.9991 |            0.9991 |         0.9991 |     0.9991 |        0.9991 |           15 |           9720.53 |     6600 |
| efficientnet_b0_c_tta |     0.9992 |            0.9993 |         0.9993 |     0.9993 |        0.9992 |          nan |            nan    |     6600 |

Best test accuracy: **efficientnet_b0_c_tta** (0.9992).

Best macro F1:     **efficientnet_b0_c_tta** (0.9993).

## Per-class F1

| class            |   baseline_cnn |   mobilenetv2 |   vgg16 |   efficientnet_b0_128 |   efficientnet_b0_a |   efficientnet_b0_b |   efficientnet_b0_c_tta |
|:-----------------|---------------:|--------------:|--------:|----------------------:|--------------------:|--------------------:|------------------------:|
| MildDemented     |         0.943  |        0.7244 |  0.7184 |                0.9943 |              0.9997 |              0.9993 |                  0.9993 |
| ModerateDemented |         0.9909 |        0.9568 |  0.9506 |                1      |              1      |              1      |                  1      |
| NonDemented      |         0.8926 |        0.7354 |  0.7057 |                0.993  |              0.9979 |              0.999  |                  0.9992 |
| VeryMildDemented |         0.8356 |        0.5673 |  0.5525 |                0.9905 |              0.9979 |              0.9982 |                  0.9985 |

## Where each model is most confused

Largest off-diagonal in each normalized confusion matrix:

* **baseline_cnn** — confuses **VeryMildDemented -> NonDemented** 15.1% of VeryMildDemented samples.
* **mobilenetv2** — confuses **VeryMildDemented -> NonDemented** 32.0% of VeryMildDemented samples.
* **vgg16** — confuses **VeryMildDemented -> NonDemented** 33.2% of VeryMildDemented samples.
* **efficientnet_b0_128** — confuses **NonDemented -> VeryMildDemented** 0.8% of NonDemented samples.
* **efficientnet_b0_a** — confuses **NonDemented -> VeryMildDemented** 0.2% of NonDemented samples.
* **efficientnet_b0_b** — confuses **NonDemented -> VeryMildDemented** 0.2% of NonDemented samples.
* **efficientnet_b0_c_tta** — confuses **NonDemented -> VeryMildDemented** 0.2% of NonDemented samples.

## Notes

* Accuracy alone is misleading on imbalanced data — macro F1 is reported alongside.
* The plan flagged VeryMildDemented vs MildDemented as the hardest pair — verify in the per-class F1 table above.
* All numbers are on an *augmented & upsampled* dataset with image-level split — they are upper bounds on a clinical evaluation.
