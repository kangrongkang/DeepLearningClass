"""Build notebooks/06_model_comparison.ipynb."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.notebook_builder import build_notebook, md, code


cells = [
    md(
"""# 06 — Model Comparison

**Day:** 6 of the project plan
**Goal:** stop training, start comparing. Pull every per-model metrics JSON
and predictions npz that Days 3-5 saved, and produce a single summary view.

This notebook produces:

1. A side-by-side metrics table (accuracy, macro precision/recall/F1, weighted F1, epochs, runtime).
2. Three confusion matrices in one figure for visual comparison.
3. A grouped bar chart of headline metrics.
4. A grouped bar chart of per-class F1.
5. A short observations block — which model is best, where the confusions are, and how to interpret them given the dataset's known limitations.

> **Limitation reminder.** The dataset is augmented and upsampled. Image-level
> splitting can leak near-duplicates → all numbers below are *upper bounds* on
> a clinical evaluation. We compare architectures on equal footing, not the
> models' clinical fitness.
"""),

    md("## 0. Environment setup"),

    code(
"""import os, sys
from pathlib import Path

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
print("In Colab:", IN_COLAB)
"""),

    code(
"""from src.config import summary, FIGURES_DIR, REPORTS_DIR, PREDICTIONS_DIR
from src.comparison import (
    load_all_metrics, comparison_table, per_class_f1_table,
    plot_confusion_matrices_side_by_side, plot_per_class_f1_bars,
    plot_headline_metrics_bars, write_comparison_report,
)
print(summary())
"""),

    md("## 1. Load every model's persisted metrics\n\nWe just read the JSON / npz files that Days 3-5 wrote — there is no retraining here. If you re-run a model on Colab, just rerun this notebook to refresh the comparison."),

    code(
"""metrics = load_all_metrics()
print("Loaded metrics for:", list(metrics.keys()))
"""),

    md("## 2. Headline metrics table"),

    code(
"""table_df = comparison_table(metrics)
table_df
"""),

    md("## 3. Per-class F1\n\nMacro F1 hides which class is hard. Plan flags VeryMildDemented vs MildDemented as the most-likely confusable pair."),

    code(
"""per_class_f1_table(metrics).round(4)
"""),

    md("## 4. Headline-metric bar chart"),

    code(
"""bar_path = plot_headline_metrics_bars(metrics)
print("Saved:", bar_path)
from IPython.display import Image as IPyImage
IPyImage(filename=str(bar_path))
"""),

    md("## 5. Per-class F1 bar chart"),

    code(
"""f1_path = plot_per_class_f1_bars(metrics)
print("Saved:", f1_path)
IPyImage(filename=str(f1_path))
"""),

    md("## 6. Side-by-side confusion matrices (normalized)"),

    code(
"""cm_path = plot_confusion_matrices_side_by_side(metrics, normalize=True)
print("Saved:", cm_path)
IPyImage(filename=str(cm_path))
"""),

    md("## 7. Observations\n\nWith a *full* GPU run for all three models, expect:\n- The custom CNN baseline reaches a respectable but lower ceiling than transfer-learning models.\n- MobileNetV2 is usually the sweet spot — fastest to train, most stable.\n- VGG16 is competitive, but its much larger parameter count typically does NOT translate into a proportional accuracy lift on this dataset.\n- The hardest classes are usually the two adjacent severity levels — VeryMildDemented vs MildDemented.\n\nWith the smoke-mode CSVs we currently have, the numbers are too noisy to draw real conclusions. Run the three model notebooks on Colab GPU and rerun this notebook to refresh."),

    md("## 8. Save Markdown report"),

    code(
"""report_path = write_comparison_report(metrics)
print("Saved:", report_path)
"""),

    md("""## 9. Conclusion for Day 6

Persisted artifacts:
* `outputs/figures/06_metrics_grouped_bars.png`
* `outputs/figures/06_per_class_f1.png`
* `outputs/figures/06_confusion_matrices_side_by_side.png`
* `outputs/reports/06_model_comparison.md`

Next: **07_grad_cam.ipynb** — pick the best model, generate Grad-CAM heatmaps on a few correct and incorrect predictions to see *where* the model is looking.
"""),
]


if __name__ == "__main__":
    out = build_notebook(cells, ROOT / "notebooks" / "06_model_comparison.ipynb")
    print("Wrote:", out)
