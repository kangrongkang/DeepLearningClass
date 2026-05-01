# Day 7 — Grad-CAM Report

**Notebook:** `notebooks/07_grad_cam.ipynb`
**Date:** 2026-04-29
**Status:** Complete (smoke-mode artifacts; full-quality heatmaps will land after Colab GPU training of Days 3-5).

## Goal

Look inside the trained models. Grad-CAM heatmaps highlight the regions of an
input MRI that drive each prediction. The plan asks for a small panel of
correct/incorrect examples on the *best* model — we additionally produce
panels for the other two models so the final report can compare *where each
model attends*.

## What was built

| Artifact | Path |
|---|---|
| Notebook | `notebooks/07_grad_cam.ipynb` (10 cells, no training) |
| Grad-CAM helper | `src/gradcam.py` |
| Best-model panel | `outputs/figures/07_gradcam_<best>.png` |
| Other-model panels | `outputs/figures/07_gradcam_baseline_cnn.png`, `..._mobilenetv2.png`, `..._vgg16.png` |

## How it works

1. `load_all_metrics()` reads every `outputs/predictions/<model>_metrics.json`.
2. `best_model = argmax(macro_f1)` picks who to feature.
3. `run_gradcam_panel(best_model)` loads `outputs/models/<best>.keras`, finds the
   last `Conv2D` layer programmatically, and renders 4 image panels.
4. The same call runs for each remaining model so the report can compare attention.

## Last-conv-layer per model (auto-detected)

| Model | Last conv | Notes |
|---|---|---|
| baseline_cnn | `conv2d_3` | 4th conv block, 256 filters — top of the from-scratch stack |
| mobilenetv2 | `Conv_1` | 1×1 expansion conv just before the global average pool — canonical Grad-CAM choice for MobileNetV2 |
| vgg16 | `block5_conv3` | last conv of block 5 — the classic Grad-CAM target for VGG16 |

## Picking strategy

Per Day 7 reviewer feedback, picks are now **stratified across true classes** —
round-robin the per-class buckets of correct (and separately wrong) predictions
so a single dominant class can't monopolize the panel. The seed is fixed at 42.

## Code review summary (Reviewer pass 5 — Day 7)

No MUST FIX. Six SHOULD-CONSIDER items, three of which we applied:

| Severity | Issue | Resolution |
|---|---|---|
| Applied | `cm.get_cmap` deprecated in Matplotlib 3.7+ | Switched to `matplotlib.colormaps[name]`. |
| Applied | Random pick could bunch up in one class | Stratified round-robin over true-class buckets. |
| Applied | `best_model = max({}, ...)` would crash if no metrics exist | Notebook now `raise RuntimeError` with remediation if `load_all_metrics()` is empty. |
| Deferred | Unicode `✓`/`✗` in figure titles | Renders fine on Windows/MPL default font; left as-is. |
| Deferred | Per-row `ylabel` rotates the title | Cosmetic — adequate for current model names. |
| Deferred | Eval-time warning for relpaths length mismatch | Failure currently happens at Grad-CAM time with a clear message; promoting it to evaluation time is a small DX gain not worth the churn. |

Reviewer also explicitly approved the Grad-CAM math, the preprocessing
contract (`[0, 1]` images go in; in-graph Rescaling/`vgg_preproc` handles the
rest), and confirmed `tf.data` with `shuffle=False` keeps `relpaths` aligned
with `y_true`/`y_pred`.

## Limitations carried forward

* Grad-CAM heatmaps from smoke-mode-trained models are not scientifically
  meaningful — full Colab GPU training of Days 3-5 is a prerequisite for
  publishable interpretation.
* Even after full training, Grad-CAM is a *correlative* explanation; it tells
  us which spatial regions contributed most to a prediction, not why those
  regions are clinically informative. The final report Discussion must avoid
  over-claiming causal interpretability.

## Next

**Task 9 — Day 8-9: Final Report.** Combine the dataset description, all three
models' results, the comparison view, the Grad-CAM panels, and the limitation
section into one coherent narrative.
