# Day 1 — Data Inspection Report

**Notebook:** `notebooks/01_data_inspection.ipynb`
**Date:** 2026-04-29
**Status:** Complete

## Goal

Before any training, characterize the dataset: how many images per class,
what they look like, whether sizes/modes are uniform, and whether any files
are corrupt. Also write down the dataset's known limitations so they propagate
into the final report.

## What was built

| Artifact | Path |
|---|---|
| Notebook | `notebooks/01_data_inspection.ipynb` (executed end-to-end) |
| Class-count figure | `outputs/figures/class_counts.png` |
| Sample grid (8 per class, 32 total) | `outputs/figures/01_sample_grid.png` |
| Machine-readable summary | `outputs/reports/01_data_inspection_summary.json` |
| Reusable inspection module | `src/inspection.py` (importable + runnable as `python -m src.inspection`) |
| Shared config / IO / plot helpers | `src/config.py`, `src/io_utils.py`, `src/plot_utils.py` |

## Findings

### Class counts (matches the project plan exactly)

| Class | Count | Share |
|---|---:|---:|
| NonDemented | 12,800 | 29.1 % |
| VeryMildDemented | 11,200 | 25.5 % |
| MildDemented | 10,000 | 22.7 % |
| ModerateDemented | 10,000 | 22.7 % |
| **Total** | **44,000** | **100 %** |

Imbalance is mild (largest:smallest ≈ 1.28×). Macro F1 is the right reporting
metric alongside accuracy.

### Image properties (sampled 50 per class = 200 total)

| Property | Observation |
|---|---|
| Modes | 138/200 RGB, 62/200 grayscale (`L`) — *mixed* |
| Sizes | 200×190 (140), 180×180 (24), 176×208 (36) — *not uniform* |
| Corrupt | 0 (verified with `Image.verify()` then `img.load()`) |

### Decisions for downstream steps

1. **Force every image to 3 channels** at load time so MobileNetV2 / VGG16 don't
   need a per-image channel branch.
2. **Resize every image to 128×128** at load time. Plan baseline; can revisit at
   160 or 224 if GPU budget allows.
3. **Track macro F1**, not just accuracy.
4. **Defensive try/except** when reading the full dataset later, even though the
   sample showed no corruption.

## Code review summary (Reviewer pass 1)

The reviewer agent flagged five issues. Fixes applied:

| Severity | Issue | Fix |
|---|---|---|
| MUST FIX | `list_images` used filesystem-order iteration → splits would be non-deterministic across OSes | `src/io_utils.py:list_images` now sorts within each class folder |
| MUST FIX | `plot_sample_grid` ignored shuffled order (`idxs[c % len(idxs)]`) → "random" was a lie | `src/plot_utils.py:plot_sample_grid` now takes inputs at face value |
| SHOULD | Plan asks 8-12 samples per class; we shipped 4 | Bumped to 8 per class |
| SHOULD | Colab fallback could silently mis-route `DATA_ROOT` | `src/config.py` raises `FileNotFoundError` with clear remediation |
| SHOULD | `assert DATA_ROOT.exists()` strips under `python -O` | Replaced with explicit `if … raise` |
| SHOULD | `Image.verify()` misses truncated files | Added `img.load()` second-pass |

The TF-decode-image hint (re: `decode_image` not always honoring `channels=`) is
parked for Day 2 when TF actually gets used.

## Limitations carried forward

The Kaggle dataset is **augmented and upsampled**. With image-level splitting,
near-duplicates can leak across train / val / test, inflating test accuracy.
This is repeated in:

- the intro markdown of `01_data_inspection.ipynb`
- the conclusions table in the same notebook
- this report
- the README

It will be repeated again in every later notebook and in the final report's
Discussion.

## Verification

* Notebook validated via `nbformat.validate()` and executed end-to-end via
  `jupyter nbconvert --execute` — no errors.
* Standalone module form (`python -m src.inspection`) reproduces the same JSON
  summary in 1.2 s on the local machine.
* Sample grid renders 8 images for every class — visually inspected to match
  expected MRI appearance.

## Next

**Task 3 — Day 2: Data Preprocessing & Split.** Build a stratified 70/15/15 split
saved as CSVs, define the light-augmentation policy, and prepare a smoke test
that loads a single batch via `tf.data` to confirm the pipeline produces
`(batch_size, 128, 128, 3)` float tensors.
