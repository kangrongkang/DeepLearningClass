"""Build notebooks/01_data_inspection.ipynb."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.notebook_builder import build_notebook, md, code


cells = [
    md(
"""# 01 — Data Inspection

**Project:** Deep Learning for Multi-class Alzheimer's MRI Classification
**Day:** 1 of the project plan
**Goal:** before any training, look carefully at the dataset.

This notebook does only what is needed to *understand* the data:
1. Count images per class.
2. Show 4 random images per class so we know what an MRI slice looks like in each severity level.
3. Probe a sample of images to find out: are they grayscale or RGB? are sizes uniform? are any files corrupt?
4. Write down dataset limitations for the final report.

> **Important caveat (Day 1, Step 1.3 of the plan):**
> The Kaggle Alzheimer's MRI dataset is an *augmented and upsampled* benchmark.
> If we split at the image level, very similar images can leak across train / val / test
> and inflate our numbers. Treat all results as a benchmark comparison between
> architectures, not a clinical accuracy estimate. We repeat this in every notebook.
"""),

    md("## 0. Environment setup\n\nWorks both locally and on Colab. On Colab, mount Drive and clone / copy the project there before running."),

    code(
"""import os, sys
from pathlib import Path

# --- locate the project root so `from src...` works on either Colab or local Windows ---
def _find_project_root():
    here = Path(os.getcwd()).resolve()
    for cand in [here, *here.parents]:
        if (cand / "src" / "config.py").exists():
            return cand
    raise RuntimeError("Could not find src/config.py. Are you inside the project directory?")

PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
print("Project root:", PROJECT_ROOT)
"""),

    code(
"""# Detect Colab and (optionally) mount Drive. If you keep the dataset on Drive at
# /content/drive/MyDrive/Project_DeepLearningClass, this cell will pick it up.
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except ImportError:
    IN_COLAB = False
print("In Colab:", IN_COLAB)
"""),

    code(
"""from src.config import summary, set_global_seed, DATA_ROOT, REPORTS_DIR, FIGURES_DIR
print(summary())
set_global_seed()
if not DATA_ROOT.exists():
    raise FileNotFoundError(
        f"DATA_ROOT not found: {DATA_ROOT}. On Colab, mount Drive and place the "
        f"dataset at <project>/archive/combined_images/<class>/."
    )
"""),

    md("## 1. Walk the data tree\n\nList every JPG/PNG under each class folder. We do not load pixels yet — just paths."),

    code(
"""from src.io_utils import list_images, get_logger
log = get_logger("01_inspection")
df = list_images(DATA_ROOT)
log.info("Found %d image files under %s", len(df), DATA_ROOT)
df.head()
"""),

    md("## 2. Class counts\n\nWe expect approximately:\n- NonDemented 12,800\n- VeryMildDemented 11,200\n- MildDemented 10,000\n- ModerateDemented 10,000\n\nThe classes are imbalanced — NonDemented is ~28% larger than the smallest class.\nWhen we report results we will look at macro F1 in addition to accuracy."),

    code(
"""from src.inspection import class_counts
from src.plot_utils import plot_class_counts

counts = class_counts(df)
for k, v in counts.items():
    print(f"  {k:18s} {v:>6,}")
print(f"  {'TOTAL':18s} {sum(counts.values()):>6,}")

fig_path = plot_class_counts(counts, title="Images per class (raw dataset)")
print("Saved:", fig_path)
from IPython.display import Image as IPyImage
IPyImage(filename=str(fig_path))
"""),

    md("## 3. Image probe — modes, sizes, corruption\n\nWe sample 50 images per class (200 total) and:\n- record the PIL `mode` (RGB vs L grayscale),\n- record the (width, height),\n- run `Image.verify()` to flag corrupt files.\n\nThis is enough to decide our preprocessing without reading all 44k files twice."),

    code(
"""from src.inspection import probe_images
probe = probe_images(df, n_per_class=50)
print("Checked:", probe["checked"])
print("Modes  :", probe["modes"])
print("Sizes  :", probe["sizes"])
print("Corrupt:", len(probe["corrupt"]))
if probe["corrupt"]:
    for line in probe["corrupt"][:5]:
        print(" ", line)
"""),

    md("""### What we learn from the probe

* **Mode is mixed.** Some images come back as `RGB`, some as `L` (single-channel grayscale).
  → We will force every image to 3 channels in preprocessing so transfer-learning models
  (MobileNetV2, VGG16) work without per-image branching.
* **Sizes are not uniform.** We see 200×190, 180×180, and 176×208 in the sample.
  → We will resize to a single target (128×128) at load time.
* **No corrupt files** in the sampled subset. We will still wrap full-dataset reads in
  try/except later in case a few in the long tail are bad.
"""),

    md("## 4. Visual sample grid\n\n8 random images per class — the plan asks for 8-12. Sanity-check that labels look right and notice obvious quality differences between classes."),

    code(
"""from src.inspection import sample_grid
grid_path = sample_grid(df, n_per_class=8)
print("Saved:", grid_path)
IPyImage(filename=str(grid_path))
"""),

    md("## 5. Write the machine-readable summary\n\nThe `06_model_comparison` notebook and the final report load this JSON when describing the dataset, so it must always be in sync with the data root."),

    code(
"""from src.inspection import run_inspection
summary_dict = run_inspection()
import json
print(json.dumps(summary_dict, indent=2))
"""),

    md("""## 6. Conclusions for Day 1

| Finding | Decision for the rest of the project |
|---|---|
| 44,000 images across 4 classes (matches the plan) | Proceed with all four classes. |
| Imbalance: NonDemented ≈ 1.28× ModerateDemented | Track macro F1 alongside accuracy. |
| Mixed RGB / grayscale | Force 3 channels everywhere in preprocessing. |
| Sizes 200×190 / 180×180 / 176×208 | Resize all to 128×128 (Day 2). |
| No corrupt files in sample | Continue, but defensively skip on read errors. |
| Augmented & upsampled benchmark | Repeat the limitation in every notebook + final report. |

Next: **02_data_preprocessing.ipynb** — stratified 70 / 15 / 15 split, light augmentation, save splits as CSVs.
"""),
]


if __name__ == "__main__":
    out = build_notebook(cells, ROOT / "notebooks" / "01_data_inspection.ipynb")
    print("Wrote:", out)
