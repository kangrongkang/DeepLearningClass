# Day 8-9 — Final Report Summary

**Notebook:** `notebooks/08_final_report.ipynb`
**Generator:** `python -m src.final_report` (also called from inside the notebook)
**Date:** 2026-04-29
**Status:** Complete. Project ready to be re-run end-to-end on Colab GPU; the report regenerates from artifacts without code changes.

## Goal

Pull every Day 1-7 artifact into a single coherent narrative covering:

1. Introduction & dataset
2. Preprocessing
3-5. Per-model results (CNN baseline / MobileNetV2 / VGG16)
6. Comparison
7. Grad-CAM
8. Discussion & limitations
9. Reproducibility checklist

## What was built

| Artifact | Path |
|---|---|
| Auto-generated final report | `outputs/reports/00_final_report.md` (213 lines) |
| Final notebook | `notebooks/08_final_report.ipynb` (renders the report inline + lists all artifacts) |
| Report generator | `src/final_report.py` (pure aggregation; no model loading or inference) |

## Why an auto-generator instead of a static Markdown file

Because the dataset, splits, and trained models live in separate places, the
final report has to stay in sync with whatever currently exists under
`outputs/`. By generating the report from the persisted artifacts:

- Re-running any model notebook → re-running `python -m src.final_report` →
  immediately gives you a fresh report without hand-editing.
- The report can never lie about a number that's been re-trained but not
  re-pasted.
- Reproducibility is structural, not promised: `00_final_report.md`,
  `06_model_comparison.md`, the comparison figures, and the per-class F1 chart
  all rebuild from the same JSON inputs.

## What the final report includes

Per the plan's Day 9 checklist:

| Plan item | Where in the report |
|---|---|
| dataset samples figure | Section 2 (`01_sample_grid.png`) |
| class count bar chart | Section 2 (`class_counts.png`) |
| three training curves | Sections 4 / 5 / 6 (per-model `*_training_curves.png`) |
| three confusion matrices | Sections 4 / 5 / 6 (per-model `*_confusion_matrix.png`) + side-by-side in Section 7 |
| final comparison table | Section 7 (auto-generated) |
| Grad-CAM figure | Section 8 (one panel per model) |
| augmented + upsampled caveat | Section 9 (Discussion) — explicit, with all four plan-required sentences |

## Project artifacts inventory

**Notebooks (8):** `01_data_inspection`, `02_data_preprocessing`, `03_model_a_baseline_cnn`, `04_model_b_mobilenetv2`, `05_model_c_vgg16`, `06_model_comparison`, `07_grad_cam`, `08_final_report`.

**Build scripts (8):** `notebooks/build_<NN>_*.py` — every notebook is regenerated from its build script. Stable cell IDs, clean diffs.

**Source modules (10):**

| Module | Purpose |
|---|---|
| `src/config.py` | Paths, classes, seeds, hyperparameters; auto-detects Colab vs local |
| `src/io_utils.py` | Image listing, splits I/O, tf.data pipeline |
| `src/plot_utils.py` | Shared matplotlib styling |
| `src/inspection.py` | Day 1 data inspection (runnable as `python -m src.inspection`) |
| `src/preprocessing.py` | Day 2 stratified split + augmentation visualization |
| `src/models.py` | `build_baseline_cnn`, `build_mobilenetv2`, `build_vgg16`, `unfreeze_top_layers` |
| `src/training.py` | `configure_gpu`, `make_callbacks`, `train_model`, `evaluate_and_save` |
| `src/comparison.py` | Day 6 comparison table, per-class F1, side-by-side CMs |
| `src/gradcam.py` | Day 7 Grad-CAM (last-conv auto-detect, stratified picks, jet overlay) |
| `src/final_report.py` | Day 9 final-report generator |
| `src/notebook_builder.py` | Helper for assembling notebooks from cell lists |

**Figures (15):** class counts, sample grid, augmentation examples, three training-curve PNGs, three per-model confusion matrices, three comparison figures (headline / per-class F1 / side-by-side CM), three Grad-CAM panels.

**Model checkpoints (5):** `baseline_cnn.keras`, `mobilenetv2_stage1.keras`, `mobilenetv2.keras`, `vgg16_stage1.keras`, `vgg16.keras`.

**Predictions (6):** three `*_metrics.json` and three `*_test_predictions.npz`.

**Reports (13):** one final report, one auto-comparison, one per-day write-up (×7), one inspection JSON, one split-summary JSON, plus per-stage train.log files.

**Splits (3):** `train.csv`, `val.csv`, `test.csv` (relative paths, portable across OSes).

## Reviewer passes — summary across the project

| Pass | Day | Files reviewed | MUST FIX | SHOULD CONSIDER applied | Outcome |
|---|---|---|---:|---:|---|
| 1 | Day 1 | inspection + helpers | 2 | 4 | All addressed |
| 2 | Day 2 | preprocessing + tf.data pipeline | 0 (1 promoted to fix: portable CSVs) | 4 | All addressed |
| 3 | Day 3 | baseline CNN + training/eval | 2 (Lambda preprocess, class-name assert) | 1 | All addressed |
| 4 | Days 4-6 | Mobilenet + VGG + comparison | 0 (3 promoted: silent unfreeze, ckpt overwrite, generic notes) | 5 | All addressed |
| 5 | Day 7 | Grad-CAM | 0 | 3 | Applied; 3 deferred as cosmetic |

## Verification

* All 8 notebooks execute end-to-end on local TF 2.21 CPU after every fix.
* All three model factories save and reload from `.keras` with bit-identical outputs.
* Predictions npz now stores per-row `relpaths` so Grad-CAM works against any subset (full or smoke-mode).
* `src/final_report.py` pulls everything together into a fully-regeneratable Markdown document.

## What the user does next

To get scientifically meaningful numbers (the smoke-mode metrics are uninformative):

1. Open `notebooks/03_model_a_baseline_cnn.ipynb` on Colab with a **GPU runtime**.
2. The `SMOKE` flag will auto-flip to `False` because the GPU is visible — the full 30-epoch training schedule runs (~10-15 min on T4).
3. Repeat for `04_…mobilenetv2` and `05_…vgg16`.
4. Re-run `06_model_comparison`, `07_grad_cam`, and `08_final_report` to refresh the report — no other changes needed; everything pulls from disk.

If running on the local Windows + RTX 5080 instead, set up WSL2 with TensorFlow 2.16+ and CUDA 12; the notebooks are identical there.

## Limitations baked into the report (per the plan)

> 1. This is a public augmented and upsampled dataset.
> 2. Image-level split may produce optimistic performance.
> 3. The data are processed 2D JPG MRI images, not raw 3D MRI volumes.
> 4. Therefore, this project should be interpreted as a benchmark comparison study, not a clinical diagnostic system.

All four sentences are explicitly stated in `outputs/reports/00_final_report.md` Section 9.

## Project closing note

The 9-step plan from `project plan.docx` is fully implemented. Every code
artifact has been reviewed by an independent agent at least once; the issues
they flagged are tracked in the per-day reports above. The project is ready
for a Colab-GPU pass that overwrites the smoke-mode artifacts with full-quality
numbers — the report will refresh automatically.
