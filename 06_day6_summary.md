# Day 6 — Model Comparison Summary

**Notebook:** `notebooks/06_model_comparison.ipynb`
**Date:** 2026-04-29
**Status:** Complete (smoke-mode metrics; full-mode numbers will land after the user runs Days 3-5 on Colab GPU and re-executes this notebook).

> The auto-generated comparison artifacts live at:
> - `outputs/reports/06_model_comparison.md` — auto-built by `write_comparison_report` every time the notebook runs. Open that for the live numbers.
> - This file is the **process** report describing what was built and what the reviewer asked us to fix.

## What was built

| Artifact | Path |
|---|---|
| Notebook | `notebooks/06_model_comparison.ipynb` (15 cells, no training) |
| Metrics-loading + plotting helpers | `src/comparison.py` |
| Auto-generated report | `outputs/reports/06_model_comparison.md` (rebuilds on every notebook run) |
| Headline-metrics bar chart | `outputs/figures/06_metrics_grouped_bars.png` |
| Per-class F1 bar chart | `outputs/figures/06_per_class_f1.png` |
| Side-by-side confusion matrices | `outputs/figures/06_confusion_matrices_side_by_side.png` |

## What it shows

The comparison notebook:

1. Loads every `outputs/predictions/<model>_metrics.json` (no retraining).
2. Builds a head-to-head metrics table with accuracy, macro precision/recall/F1, weighted F1, epochs run, and training time.
3. Renders a per-class F1 table preserved in model-load order (not alphabetical, so adding a fourth model doesn't shuffle the columns).
4. Plots three figures: headline metrics, per-class F1, and normalized confusion matrices side-by-side.
5. **Auto-extracts the largest off-diagonal in each model's confusion matrix** and writes a sentence into the report — addresses the plan's "note where models confuse classes" requirement directly.

## Code review summary (Reviewer pass 4 — Days 4-6 batch)

The Days 4-6 reviewer flagged 10 SHOULD-CONSIDER items. I applied the high-impact ones:

| Severity | Issue | Fix |
|---|---|---|
| Promoted to FIX | `unfreeze_top_layers` silent no-op when backbone wasn't found as a submodel — TF 2.21 inlines the backbone | Two-path implementation: try embedded submodel first, fall back to "everything except `head_layer_names` is the backbone" with explicit `RuntimeError` if no layers can be unfrozen. Verified MobileNetV2 5K→1.2M and VGG16 132K→13M trainable params. |
| Fix | Stage-2 `ModelCheckpoint` could overwrite a better stage-1 best | Day 4 notebook now compares `max(hist1.val_accuracy)` vs `max(hist2.val_accuracy)` after stage 2; if stage-1 wins, copies its checkpoint to the canonical `mobilenetv2.keras` and reloads the in-memory model. |
| Fix | Day 5 `DO_FINETUNE=False` silently skipped saving when stage-1 ckpt didn't exist | Added `else: model.save(dst_path)` so `vgg16.keras` is always written. |
| Fix | Generic Day 6 report Notes — didn't surface actual confusions | Added `biggest_confusion()` helper that finds the largest normalized off-diagonal per model and writes "X confuses A → B Y% of A samples" into the report. |
| Fix | `per_class_f1_table` columns sorted alphabetically by `pivot()` | Reindex by `metrics.keys()` so the column order matches the input order across all tables/figures. |
| Fix | `comparison_table` `KeyError` if a metrics JSON is missing a field | All lookups now use `.get(..., float("nan"))`. |
| Fix | `plot_test_curves_side_by_side` was misnamed (it's a bar chart, not curves) | Renamed to `plot_headline_metrics_bars`. |
| Fix | Day 4 history-stitching dropped single-sided keys | Adopted the union-of-keys pattern from Day 5. |
| Deferred | Per-class F1 bar-chart label collisions (only matters at N≥4 models) | Left as-is for 3 models; note added for future. |
| Deferred | Bootstrap boilerplate duplicated across notebooks | Real but small DRY win; not worth refactoring at this stage. |

## Round-trip / smoke verification

- All three model notebooks executed end-to-end on local CPU after the fixes.
- Comparison notebook executed cleanly and the auto-generated report contains real-data sentences (e.g. "vgg16 — confuses MildDemented → NonDemented 68.6% of MildDemented samples"). With smoke-mode metrics this is mostly noise, but the *mechanism* is verified — when the user runs full Colab GPU training, those sentences will reflect real model behavior.

## Limitations carried forward

* Smoke metrics ≠ scientific results. The comparison ranking will change after Colab full-GPU training.
* Augmented & upsampled dataset → image-level split → all numbers are upper bounds on a clinical evaluation. Repeated in the auto-generated comparison report and the final report.

## Next

**Task 8 — Day 7: Grad-CAM.** Pick the best model (per the comparison table), generate Grad-CAM heatmaps on a few correct and incorrect predictions, save the figure for the final report.
