"""Build notebooks/08_final_report.ipynb — pulls every artifact together."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.notebook_builder import build_notebook, md, code


cells = [
    md(
"""# 08 — Final Report

**Day:** 8-9 of the project plan
**Goal:** consolidate Day 1-7 artifacts into a single, coherent narrative.

This notebook does NOT train, evaluate, or generate Grad-CAMs. It loads the
files that the previous notebooks already produced (metrics JSONs, figures,
splits, predictions) and assembles the final report — both as a
`outputs/reports/00_final_report.md` file (fully regeneratable from artifacts)
and as inline cells you can read here.

If any cell looks empty (e.g. no Grad-CAM panel for VGG16), it means that
model's notebook hasn't been run yet. Run notebooks 01-07 in order on Colab
GPU and re-execute this notebook to refresh.
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

from src.config import summary, FIGURES_DIR, REPORTS_DIR, PREDICTIONS_DIR
print(summary())
"""),

    md("## 1. Refresh comparison artifacts and the final report markdown"),

    code(
"""# Rebuild the comparison view and the final report from whatever metrics exist on disk
from src.comparison import (
    load_all_metrics, write_comparison_report,
    plot_confusion_matrices_side_by_side, plot_per_class_f1_bars,
    plot_headline_metrics_bars,
)
from src.final_report import build_final_report

metrics = load_all_metrics()
print("Models with metrics on disk:", list(metrics.keys()))

if metrics:
    plot_headline_metrics_bars(metrics)
    plot_per_class_f1_bars(metrics)
    plot_confusion_matrices_side_by_side(metrics, normalize=True)
    write_comparison_report(metrics)

report_path = build_final_report()
print("Final report:", report_path)
"""),

    md("## 2. Render the final report inline"),

    code(
"""from IPython.display import Markdown
Markdown(report_path.read_text(encoding="utf-8"))
"""),

    md("## 3. Index of artifacts\n\nRun this cell to see every file the project produces."),

    code(
"""def list_dir(d):
    if not d.exists(): return []
    return sorted(p.name for p in d.iterdir() if p.is_file())

print("Figures:")
for f in list_dir(FIGURES_DIR):
    print(f"  {f}")
print()
print("Predictions / metrics:")
for f in list_dir(PREDICTIONS_DIR):
    print(f"  {f}")
print()
print("Reports:")
for f in list_dir(REPORTS_DIR):
    print(f"  {f}")
"""),

    md("""## 4. What to hand in

For the course submission, the deliverables are:
1. **This notebook** (`08_final_report.ipynb`) — single page that renders the full report.
2. **`outputs/reports/00_final_report.md`** — same content as a standalone Markdown file.
3. **All seven prior notebooks** (`01_…` to `07_…`) — show the day-by-day work.
4. **`outputs/figures/`** — all PNGs referenced by the report.
5. **`outputs/predictions/`** — JSON / npz files for grading reproducibility.
6. **`outputs/models/<model>.keras`** — the best-weights checkpoints. (These can be re-trained from the notebooks if file size is a problem; nothing else depends on having them locally except Grad-CAM.)

**Reproducibility:** `python -m src.final_report` regenerates the Markdown report from the artifacts above; nothing in the report is hand-edited.
"""),
]


if __name__ == "__main__":
    out = build_notebook(cells, ROOT / "notebooks" / "08_final_report.ipynb")
    print("Wrote:", out)
