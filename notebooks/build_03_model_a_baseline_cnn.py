"""Build notebooks/03_model_a_baseline_cnn.ipynb."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.notebook_builder import build_notebook, md, code


cells = [
    md(
"""# 03 — Model A: Custom CNN Baseline

**Day:** 3 of the project plan
**Goal:** train a small CNN *from scratch* on the 70/15/15 splits we built in Day 2.
This sets the floor every later model has to beat.

> **GPU is required for full training.** On CPU the full 25-epoch run takes hours;
> on a Colab T4 it finishes in ~10-15 minutes. The first cell after setup checks
> the active device and prints a warning if no GPU is visible.

> **Limitation reminder:** the dataset is augmented and upsampled. Image-level
> splitting can leak near-duplicates across train / val / test, inflating test
> accuracy. We treat all numbers as a comparison between architectures, not a
> clinical estimate. This caveat is repeated in every later notebook.
"""),

    md("## 0. Environment setup"),

    code(
"""import os, sys
from pathlib import Path

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
"""from src.config import (summary, set_global_seed, BATCH_SIZE, IMG_SIZE,
                          IMG_CHANNELS, NUM_CLASSES, CLASS_NAMES)
from src.training import configure_gpu
print(summary())
set_global_seed()
gpu_info = configure_gpu()
print("Device info:", gpu_info)
if gpu_info["device"] != "GPU":
    print("\\n*** WARNING: no GPU detected. Full training on CPU is impractical. ***")
    print("    On Colab: Runtime -> Change runtime type -> GPU.")
    print("    Locally on Windows: TF needs WSL2 for GPU.")
"""),

    md("## 1. Load the splits and build the data pipeline\n\nWe load the CSVs written by `02_data_preprocessing.ipynb` and turn them into `tf.data.Dataset`s.\nTraining gets light augmentation; val and test are clean."),

    code(
"""from src.io_utils import load_split, make_tf_dataset

train_df = load_split("train")
val_df   = load_split("val")
test_df  = load_split("test")
print(f"train: {len(train_df):,}   val: {len(val_df):,}   test: {len(test_df):,}")

train_ds = make_tf_dataset(train_df, batch_size=BATCH_SIZE, shuffle=True,  augment=True)
val_ds   = make_tf_dataset(val_df,   batch_size=BATCH_SIZE, shuffle=False, augment=False)
test_ds  = make_tf_dataset(test_df,  batch_size=BATCH_SIZE, shuffle=False, augment=False)
"""),

    md("""## 2. Build the model

Architecture (matches Day 3, Step 3.1 of the project plan):

| Block | Layers |
|---|---|
| Conv block 1 | Conv(32, 3×3, same) → BN → ReLU → MaxPool 2×2 |
| Conv block 2 | Conv(64, 3×3, same) → BN → ReLU → MaxPool 2×2 |
| Conv block 3 | Conv(128, 3×3, same) → BN → ReLU → MaxPool 2×2 |
| Conv block 4 | Conv(256, 3×3, same) → BN → ReLU → MaxPool 2×2 |
| Head | GlobalAveragePooling → Dense(128, ReLU) → Dropout(0.3) → Dense(4, softmax) |

GAP instead of Flatten keeps the parameter count low (~423k) so we can train from
scratch without the model overfitting on day one.
"""),

    code(
"""from src.models import build_baseline_cnn
model = build_baseline_cnn()
model.summary(line_length=110)
"""),

    md("""## 3. Training settings

| Knob | Value |
|---|---|
| optimizer | Adam |
| learning rate | 1e-3 |
| batch size | 32 |
| epochs | 30 (capped) |
| EarlyStopping | `monitor="val_accuracy", patience=5, restore_best_weights=True` |
| ReduceLROnPlateau | `factor=0.5, patience=2, min_lr=1e-6` |
| ModelCheckpoint | best-only, saved at `outputs/models/baseline_cnn.keras` |

> **Smoke vs full mode.** If you don't have a GPU, set `EPOCHS = 2` and `SAMPLE_FRAC = 0.02` to do
> a 1-minute smoke test on a tiny subset just to verify the pipeline. Otherwise leave them at the
> full values for a real training run.
"""),

    code(
"""# Toggle between smoke (CPU-friendly) and full (GPU) modes
SMOKE = (gpu_info["device"] != "GPU")
EPOCHS = 2 if SMOKE else 30
SAMPLE_FRAC = 0.02 if SMOKE else 1.0
# Optional env override for quick CI-style verification on GPU machines
if "OVERRIDE_EPOCHS" in os.environ:
    EPOCHS = int(os.environ["OVERRIDE_EPOCHS"])
if "OVERRIDE_SAMPLE_FRAC" in os.environ:
    SAMPLE_FRAC = float(os.environ["OVERRIDE_SAMPLE_FRAC"])
print(f"Mode: {'SMOKE' if SMOKE else 'FULL'}   epochs={EPOCHS}   sample_frac={SAMPLE_FRAC}")
"""),

    code(
"""if SAMPLE_FRAC < 1.0:
    train_df_used = train_df.sample(frac=SAMPLE_FRAC, random_state=42)
    val_df_used   = val_df.sample(frac=SAMPLE_FRAC, random_state=42)
    test_df_used  = test_df.sample(frac=SAMPLE_FRAC, random_state=42)
    train_ds = make_tf_dataset(train_df_used, batch_size=BATCH_SIZE, shuffle=True,  augment=True)
    val_ds   = make_tf_dataset(val_df_used,   batch_size=BATCH_SIZE, shuffle=False, augment=False)
    test_ds  = make_tf_dataset(test_df_used,  batch_size=BATCH_SIZE, shuffle=False, augment=False)
    print(f"Subsampled: train={len(train_df_used)} val={len(val_df_used)} test={len(test_df_used)}")
"""),

    code(
"""from src.training import train_model, make_callbacks

MODEL_NAME = "baseline_cnn"
callbacks = make_callbacks(MODEL_NAME, monitor="val_accuracy",
                           patience_es=5, patience_lr=2)
history = train_model(model, train_ds, val_ds, model_name=MODEL_NAME,
                      epochs=EPOCHS, callbacks=callbacks)
print(f"Trained {len(history['loss'])} epochs in {history['elapsed_seconds']}s")
"""),

    md("## 4. Curves & test-set evaluation\n\nWe evaluate on the held-out test set, save predictions / metrics / figures, and produce the confusion matrix."),

    code(
"""from src.training import evaluate_and_save
metrics = evaluate_and_save(model, test_ds, model_name=MODEL_NAME, history=history,
                            test_df=(train_df_used.iloc[:0] if False else (test_df_used if SAMPLE_FRAC < 1.0 else test_df)))

print(f"Accuracy:        {metrics['accuracy']:.4f}")
print(f"Macro precision: {metrics['macro_precision']:.4f}")
print(f"Macro recall:    {metrics['macro_recall']:.4f}")
print(f"Macro F1:        {metrics['macro_f1']:.4f}")
"""),

    code(
"""from IPython.display import Image as IPyImage
from src.config import FIGURES_DIR
IPyImage(filename=str(FIGURES_DIR / f"{MODEL_NAME}_training_curves.png"))
"""),

    code(
"""IPyImage(filename=str(FIGURES_DIR / f"{MODEL_NAME}_confusion_matrix.png"))
"""),

    md("## 5. Per-class scores\n\nAccuracy alone hides the class-imbalance picture. Per-class precision / recall / F1 tell us where the model is actually struggling."),

    code(
"""import pandas as pd
per = metrics["per_class"]
pd.DataFrame({
    "class": per["labels"],
    "precision": [round(v, 4) for v in per["precision"]],
    "recall":    [round(v, 4) for v in per["recall"]],
    "f1":        [round(v, 4) for v in per["f1"]],
})
"""),

    md("""## 6. Conclusion for Day 3

After full training (run on Colab GPU), expectations:
* The custom CNN typically reaches **>0.9 test accuracy** on this dataset because
  upsampled augmented copies make the task easier than a held-out clinical evaluation.
* The hardest pair to separate is usually **VeryMildDemented vs MildDemented** —
  expect off-diagonal cells there in the confusion matrix.

What gets persisted for the comparison notebook:
* `outputs/models/baseline_cnn.keras` — best weights.
* `outputs/predictions/baseline_cnn_test_predictions.npz` — y_true / y_pred / y_proba.
* `outputs/predictions/baseline_cnn_metrics.json` — accuracy / precision / recall / F1
  (macro and per-class) + confusion matrix + training info.
* `outputs/figures/baseline_cnn_training_curves.png` and `..._confusion_matrix.png`.

Next: **04_model_b_mobilenetv2.ipynb** — does a pretrained backbone help?
"""),
]


if __name__ == "__main__":
    out = build_notebook(cells, ROOT / "notebooks" / "03_model_a_baseline_cnn.ipynb")
    print("Wrote:", out)
