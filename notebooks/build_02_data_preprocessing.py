"""Build notebooks/02_data_preprocessing.ipynb."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.notebook_builder import build_notebook, md, code


cells = [
    md(
"""# 02 — Data Preprocessing & Stratified Split

**Day:** 2 of the project plan
**Goal:** turn the raw image folders into deterministic train / val / test CSVs that
every model in Days 3-5 will load identically. Also nail down the augmentation policy.

This notebook has three jobs:
1. Decide and document the input format (size, channels, value range).
2. Build a stratified 70 / 15 / 15 split — *file-list level only, no pixels copied*.
3. Define a light augmentation policy (rotation / shift / zoom; **no horizontal flip**)
   and visualize a few augmented examples so a human can confirm "light" is actually light.

> **Limitation reminder:** the dataset is augmented and upsampled. Image-level splitting
> can leak near-duplicates across train / val / test. We split by image only because the
> dataset doesn't expose patient/scan IDs — the limitation belongs in the final report.
"""),

    md("## 0. Environment setup"),

    code(
"""import os, sys
from pathlib import Path

# Select Keras backend BEFORE importing anything that touches keras.
# 'torch' = native Windows GPU on RTX 5080. Override to 'tensorflow' on Colab if desired.
os.environ.setdefault("KERAS_BACKEND", "torch")

def _find_project_root():
    here = Path(os.getcwd()).resolve()
    for cand in [here, *here.parents]:
        if (cand / "src" / "config.py").exists():
            return cand
    raise RuntimeError("Could not find src/config.py.")

PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import google.colab; IN_COLAB = True
    from google.colab import drive; drive.mount("/content/drive", force_remount=False)
except ImportError:
    IN_COLAB = False
print("In Colab:", IN_COLAB, "  Keras backend:", os.environ["KERAS_BACKEND"])
"""),

    code(
"""from src.config import (
    summary, set_global_seed, DATA_ROOT, IMG_SIZE, IMG_CHANNELS,
    BATCH_SIZE, TRAIN_FRAC, VAL_FRAC, TEST_FRAC, SEED, CLASS_NAMES, NUM_CLASSES,
)
print(summary())
print(f"Splits: train={TRAIN_FRAC}, val={VAL_FRAC}, test={TEST_FRAC}")
set_global_seed()
"""),

    md("## 1. Input format decisions\n\n| Knob | Value | Why |\n|---|---|---|\n| Image size | 128×128 | Day 1 showed sizes are non-uniform (200×190, 180×180, 176×208). 128² is enough for the baseline CNN and lets MobileNetV2 / VGG16 fit comfortably in Colab. |\n| Channels | 3 (RGB) | Day 1 showed ~30% of images come in as grayscale. Forcing 3 channels everywhere lets us reuse one pipeline for the baseline CNN *and* the ImageNet-pretrained backbones. |\n| Value range | [0, 1] | Plain `/255.0` normalization. We rely on Keras's per-backbone preprocessing layer (`preprocess_input`) where it matters (MobileNetV2 / VGG16). |"),

    md("## 2. Stratified 70 / 15 / 15 split\n\nWe do this in two steps with `sklearn.model_selection.train_test_split`:\n1. `train_test_split(df, train_size=0.70, stratify=labels)` → train vs (val+test).\n2. `train_test_split(rest, test_size=0.50, stratify=labels)` → val vs test.\n\nBoth calls pass `stratify=`, so each class keeps the same proportion in every split.\nThe inputs come from `list_images`, which sorts within each class folder, so the\nresulting splits are fully deterministic at `seed=42`."),

    code(
"""from src.preprocessing import run_preprocessing
import json
split_summary = run_preprocessing()
print(json.dumps(split_summary, indent=2))
"""),

    md("### Sanity check: per-class fractions are stable across splits"),

    code(
"""import pandas as pd
rows = []
for split in ["train", "val", "test"]:
    counts = split_summary[split]["by_class"]
    total = split_summary[split]["total"]
    for c, n in counts.items():
        rows.append({"split": split, "class": c, "count": n,
                     "share_within_split": round(n / total, 4)})
counts_df = pd.DataFrame(rows)
counts_df.pivot(index="class", columns="split", values="share_within_split")
"""),

    md("If the table above shows roughly the same fractions across train / val / test for each class, the stratification worked."),

    md("## 3. Persisted split files\n\nAfter this notebook runs, every Day 3-5 model loads the splits via `src.io_utils.load_split('train')`. There is no per-model splitting code — that would be the easiest way to introduce a subtle leak."),

    code(
"""from src.io_utils import load_split
train_df = load_split("train"); val_df = load_split("val"); test_df = load_split("test")
print(f"train: {len(train_df):,}   val: {len(val_df):,}   test: {len(test_df):,}")
train_df.head()
"""),

    md("""## 4. Augmentation policy

The dataset has *already* been augmented and upsampled by the publisher, so our own
augmentation should be **light**. We use only:

| Augmentation | Magnitude | Notes |
|---|---|---|
| Rotation | ±10.8° (`RandomRotation(factor=0.03)`, where `factor` is a fraction of a full turn → 0.03 × 360°) | Plan recommends ≤10° |
| Translation | ±5% width / height | Small shifts only |
| Zoom | ±5% | Small zoom only |
| **Horizontal flip** | **NOT USED** | Plan flags it as risky for MRI — left/right brain asymmetry can be informative |

Augmentation is applied **only in the training pipeline**; val and test are clean.
"""),

    code(
"""from src.preprocessing import visualize_augmentation
fig_path = visualize_augmentation(train_df, n_examples=4, n_aug_per_example=4)
print("Saved:", fig_path)
from IPython.display import Image as IPyImage
IPyImage(filename=str(fig_path))
"""),

    md("Visually the augmented copies should look almost identical to the original — only a tiny rotation / shift / zoom. If they look distorted, the policy is too strong."),

    md("## 5. Data pipeline smoke test\n\nLoad one batch from each split and verify the shapes / dtype / value range.\nThis is the contract every model notebook will rely on. The pipeline is a Keras 3 `PyDataset` so it works on either TF or PyTorch backend."),

    code(
"""from src.io_utils import make_keras_dataset

# small subsets for the smoke test so this cell is fast
train_ds = make_keras_dataset(train_df.head(96),  batch_size=32, shuffle=True,  augment=True)
val_ds   = make_keras_dataset(val_df.head(64),    batch_size=32, shuffle=False, augment=False)
test_ds  = make_keras_dataset(test_df.head(64),   batch_size=32, shuffle=False, augment=False)

import numpy as np
for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
    xb, yb = ds[0]   # PyDataset.__getitem__ returns one batch
    xb = np.asarray(xb); yb = np.asarray(yb)
    print(f"{name:5s} batch -> shape {xb.shape}  dtype {xb.dtype}  "
          f"min {float(xb.min()):.3f}  max {float(xb.max()):.3f}  "
          f"labels {sorted(set(int(v) for v in yb.tolist()))}")

assert xb.shape == (32, 128, 128, 3) and xb.dtype == np.float32
print("PASS: pipeline produces (32, 128, 128, 3) float32 batches in [0, 1].")
"""),

    md("""## 6. Conclusions for Day 2

- Splits saved at `outputs/splits/{train,val,test}.csv` (deterministic at SEED=42).
- Stratification preserves the class fractions almost exactly.
- Augmentation policy is light per the plan (no horizontal flip).
- `make_tf_dataset` is the single entry point for every Day 3-5 model.

Next: **03_model_a_baseline_cnn.ipynb** — custom 4-block CNN trained from scratch.
"""),
]


if __name__ == "__main__":
    out = build_notebook(cells, ROOT / "notebooks" / "02_data_preprocessing.ipynb")
    print("Wrote:", out)
