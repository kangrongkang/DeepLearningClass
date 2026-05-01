# Day 2 — Data Preprocessing & Stratified Split Report

**Notebook:** `notebooks/02_data_preprocessing.ipynb`
**Date:** 2026-04-29
**Status:** Complete

## Goal

Convert the raw 4-folder dataset into deterministic CSV splits that every Day 3-5
model loads identically, and lock in the input format and augmentation policy.

## What was built

| Artifact | Path |
|---|---|
| Notebook | `notebooks/02_data_preprocessing.ipynb` |
| Train split (relpath, class, label) | `outputs/splits/train.csv` |
| Val split | `outputs/splits/val.csv` |
| Test split | `outputs/splits/test.csv` |
| Augmentation example figure | `outputs/figures/02_augmentation_examples.png` |
| Machine-readable summary | `outputs/reports/02_split_summary.json` |
| Reusable preprocessing module | `src/preprocessing.py` (runnable as `python -m src.preprocessing`) |

## Decisions

| Knob | Value | Why |
|---|---|---|
| Image size | 128×128 | Day 1 found non-uniform native sizes (200×190 / 180×180 / 176×208). 128² is enough for the baseline CNN, fits MobileNetV2 / VGG16, and trains fast on Colab. |
| Channels | 3 (RGB) | Day 1 found ~30% of images come in as grayscale. Forcing 3 channels lets one pipeline serve all three models. |
| Value range | [0, 1] | `/255.0` at decode; per-backbone preprocessing is added inside the model graph for MobileNetV2 / VGG16. |
| Train fraction | 0.70 | Plan default. |
| Val fraction | 0.15 | Plan default. |
| Test fraction | 0.15 | Plan default. |
| Seed | 42 | Plan default. |

## Augmentation policy (training only)

| Transform | Magnitude |
|---|---|
| Rotation | ±10.8° (`RandomRotation(factor=0.03)` — `factor` is fraction of a full turn) |
| Translation | ±5% width / height |
| Zoom | ±5% |
| Horizontal flip | **NOT applied** (MRI left/right asymmetry can be informative) |

Visualized in `outputs/figures/02_augmentation_examples.png` — augmented copies look
nearly identical to the originals, confirming the policy is "light".

## Stratified split — actual counts

| Split | Total | NonDemented | VeryMildDemented | MildDemented | ModerateDemented |
|---|---:|---:|---:|---:|---:|
| train (70%) | 30,799 | 8,959 | 7,840 | 7,000 | 7,000 |
| val (15%) | 6,600 | 1,920 | 1,680 | 1,500 | 1,500 |
| test (15%) | 6,601 | 1,921 | 1,680 | 1,500 | 1,500 |

Per-class share is identical to ~4 decimal places across the three splits — stratification worked.

## Verification (smoke test)

Notebook executes end-to-end on TF 2.21 (CPU). The pipeline contract is enforced
by an assertion:

```
train batch -> shape (32, 128, 128, 3)  dtype float32  min 0.000  max 1.000  labels [0, 1, 2, 3]
val   batch -> shape (32, 128, 128, 3)  dtype float32  min 0.000  max 1.000  labels [0, 1, 2, 3]
test  batch -> shape (32, 128, 128, 3)  dtype float32  min 0.000  max 1.000  labels [0, 1, 2, 3]
PASS: tf.data pipeline produces (32, 128, 128, 3) float32 tensors in [0, 1].
```

## Code review summary (Reviewer pass 2)

The reviewer agent flagged five SHOULD-CONSIDER items, no MUST-FIX. The biggest one
was a real Colab-portability landmine; that was promoted to a fix:

| Severity | Issue | Fix |
|---|---|---|
| Promoted to fix | Split CSVs stored absolute Windows paths → would silently break on Colab | `src/io_utils.py` now stores `relpath` only; `load_split` re-prefixes with the local `DATA_ROOT`. CSVs are now portable. |
| Doc tweak | Notebook said "fraction of 2π" for `RandomRotation(factor=)`; Keras docs say "fraction of a full turn" | Reworded the augmentation table |
| Noted | `tf.data` `reshuffle_each_iteration=True` + `AUTOTUNE` map is not bit-identical across machines | Acceptable for an academic project; splits themselves *are* deterministic since splitting is in pandas/sklearn, not tf.data |
| Noted | Drive mount could prompt for auth on Colab | Existing `force_remount=False` is the right default |
| Noted | All three augmentation layers share `seed=42` | Independent RNGs per layer make this fine |

## Limitations carried forward

- **Image-level split** still risks duplicate leakage because the dataset doesn't
  expose patient or scan IDs. Stated in the notebook intro and will be repeated
  in the final report Discussion.
- **TF training is CPU-only on this Windows host** (RTX 5080 + Blackwell needs
  WSL2 or Colab). Each training notebook will detect the device and warn if
  no GPU is present.

## Next

**Task 4 — Day 3: Custom CNN baseline.** A 4-block conv stack with BN + GAP head,
trained from scratch with Adam 1e-3, BS=32, EarlyStopping, ReduceLROnPlateau.
Save weights, history, predictions, and a per-model report.
