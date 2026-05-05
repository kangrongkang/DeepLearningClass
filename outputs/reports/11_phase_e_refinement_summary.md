# Phase E — Refinement results & comparison report

**Date:** 2026-04-30
**Status:** Closed.
**Notebook / pipeline:**
- Auto-generated comparison: `outputs/reports/06_model_comparison.md`
- Auto-generated final report: `outputs/reports/00_final_report.md`
- This document: human-readable summary of Phase E findings.

## Goal

Produce the head-to-head comparison promised by `refine.md` Phase E: load
every saved model's metrics on the **clean** (leakage-controlled) test set
and write a comparison table that the refined model has to beat by ≥5 pp
macro F1.

## Headline result

The refined EfficientNet-B0 recipe **beat every baseline by +8.3 pp macro F1**.

| Model | Test acc | Macro F1 | Δ vs baseline_cnn |
|---|---:|---:|---:|
| baseline_cnn (custom 423K, 128²) | 0.9124 | 0.9155 | — |
| mobilenetv2 (frozen→FT, 128²) | 0.7441 | 0.7459 | −16.96 |
| vgg16 (frozen, 128²) | 0.7273 | 0.7318 | −18.37 |
| **efficientnet_b0 (a) bare, 224²** | **0.9988** | **0.9989** | **+8.34** |
| **efficientnet_b0 (b) +Mixup, 224²** | **0.9991** | **0.9991** | **+8.36** |
| **efficientnet_b0 (c) +SWA+TTA, 224²** | **0.9992** | **0.9993** | **+8.38** |

All numbers on the same leakage-controlled test set (`outputs/splits/test_clean.csv`,
n=6,600). Refer to `outputs/reports/06_model_comparison.md` for the
auto-generated version of this table.

## What worked, what didn't

### Worked

- **Resolution + architecture (config a).** The single biggest contributor.
  Bare EfficientNet-B0 at 224² already takes us from 0.9155 macro F1 to
  0.9989 — that's the entire +8.3 pp gain.
- **Cluster-stratified pHash split.** Was the right de-leakage primitive
  given the absence of patient IDs. 0% cluster overlap between train and
  test by construction.
- **AdamW + cosine + linear warmup.** Standard modern recipe; converged
  to >99% within ~3 epochs of stage 2 fine-tuning.
- **Two-stage training (frozen head warm-up → full fine-tune).** Stage 1
  hit ~68% val accuracy on its own; stage 2 jumped to >99% in the first
  few epochs.

### Didn't move the needle (within noise)

- **Mixup α=0.2.** Added marginal +0.0002 macro F1 on top of bare config.
  At ceiling already; can't attribute statistically with single seed.
- **SWA last 25%.** Same story: the (c) without TTA scored 0.9986, which
  is *lower* than (b)'s 0.9991. Single-seed noise.
- **TTA.** Recovered (c) back to 0.9993 (best single number) but the gap
  is well below run-to-run variance.

### Did NOT work as the leakage hypothesis predicted

The leakage hypothesis from `refine.md` predicted that the 91% baseline_cnn
result was inflated by augmentation duplicates leaking across train/test.
**That hypothesis is rejected at the pHash level**: re-evaluating the saved
baseline_cnn checkpoint on the clean (no-near-duplicate) test set gave
0.9124 — *0.7 pp higher* than the original 0.9053 leaky test number, not
lower. Same story for MobileNetV2 (+0.15 pp) and VGG16 (−0.17 pp).

**Caveat.** pHash deduplication does not test subject-level leakage. The
dataset doesn't expose patient IDs, so subject-level leakage cannot be
measured or eliminated here. It likely contributes to the high accuracy
ceiling, but we cannot prove it.

## Per-class breakdown (the clinically interesting view)

For the hardest pair — **VeryMildDemented ↔ NonDemented** (the early-stage
distinction that matters most for clinical intervention):

| Model | VeryMild → NonDem (%) | NonDem → VeryMild (%) |
|---|---:|---:|
| Custom CNN baseline | 15.12 | 6.77 |
| MobileNetV2 | 31.96 | 11.30 |
| VGG16 | 33.21 | 14.79 |
| EfficientNet-B0 (a) | 0.18 | 0.21 |
| EfficientNet-B0 (b) | 0.06 | 0.16 |
| EfficientNet-B0 (c) +TTA | 0.00 | 0.16 |

The transfer-learning baselines confuse this pair *worse* than the
from-scratch CNN. EfficientNet-B0 cuts both off-diagonals to ≤0.2%.

## Artifacts persisted (Phase E deliverables)

- `outputs/predictions/<model>_metrics.json` (clean test set) for all 6 models.
- `outputs/predictions/<model>_test_predictions.npz` with `y_true`, `y_pred`, `y_proba`, `relpaths`.
- `outputs/figures/06_metrics_grouped_bars.png` — headline bars.
- `outputs/figures/06_per_class_f1.png` — per-class F1.
- `outputs/figures/06_confusion_matrices_side_by_side.png` — 6-model CM panel.
- `outputs/figures/efficientnet_b0_<a|b|c|c_tta>_confusion_matrix.png`.
- `outputs/figures/efficientnet_b0_<a|b|c>_training_curves.png`.

## Why is 99.9% achievable?

Three combined factors, ranked by importance:

1. **The dataset is tractable at 224×224.** The four severity classes have
   visually distinct MRI appearance (ventricular enlargement, cortical
   thinning). At ImageNet's native resolution, a pretrained backbone
   extracts those features cleanly. The custom CNN at 128² did surprisingly
   well too (91%); EfficientNet-B0 at 224² simply has more headroom.
2. **The dataset is augmented and upsampled.** Even with a cluster-stratified
   pHash split, training and test draw from the same underlying slice
   distribution. Subject-level leakage cannot be ruled out.
3. **The recipe is well-matched.** AdamW + cosine warmup is a robust default
   for transfer-learning fine-tunes; Mixup + label smoothing trade-off is
   well-known.

## Discussion vs. literature

Several papers on the same/near-identical Kaggle corpus report 95%–99.9%:
Abd El-Latif et al. 2023 (95.93% 4-class), Hussain et al. 2025 (99.4%),
El-Assy et al. 2024 (99.13–99.57% on a related ADNI split), and a ViTAD
preprint (99.98%). All of these used **image-level** random splits with no
explicit deduplication. Our 99.9% — on a strictly *image-level deduplicated*
split — is in the same neighborhood, suggesting that the gap between
"image-level random" and "image-level deduplicated" is small for this
dataset, and that the *real* gap (vs subject-level evaluation) is what
remains uninvestigated.

## Next

Task F (paper) writes this up for an academic audience with proper
references and threats-to-validity discussion. Reviewer-requested
disambiguation experiment (EfficientNet-B0 at 128² to separate
architecture from resolution) is currently training in the background.
