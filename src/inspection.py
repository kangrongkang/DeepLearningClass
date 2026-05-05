"""
Pure-Python data inspection routines that don't need TF — used by 01_data_inspection.ipynb
and runnable as a script from the project root for evidence generation.

Run as: ``PYTHONIOENCODING=utf-8 python -m src.inspection``
"""
from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.config import CLASS_NAMES, DATA_ROOT, FIGURES_DIR, REPORTS_DIR, SEED
from src.io_utils import get_logger, list_images
from src.plot_utils import plot_class_counts, plot_sample_grid


def class_counts(df: pd.DataFrame) -> dict:
    return {c: int((df["class"] == c).sum()) for c in CLASS_NAMES}


def probe_images(df: pd.DataFrame, n_per_class: int = 50, seed: int = SEED) -> dict:
    """Sample a handful of images per class and capture mode / size / corruption.

    ``Image.verify()`` is structural (catches broken JPEG headers); we follow it
    with ``img.load()`` to also catch truncated pixel data, which `verify` misses.
    """
    rng = random.Random(seed)
    modes: Counter[str] = Counter()
    sizes: Counter[tuple] = Counter()
    bad: list[str] = []
    sampled = []
    for cname in CLASS_NAMES:
        sub = df[df["class"] == cname]["filepath"].tolist()
        rng.shuffle(sub)
        sampled.extend(sub[:n_per_class])
    for p in sampled:
        try:
            with Image.open(p) as img:
                img.verify()
            with Image.open(p) as img:
                img.load()  # force pixel decode; catches truncated files
                modes[img.mode] += 1
                sizes[img.size] += 1
        except (UnidentifiedImageError, OSError) as e:
            bad.append(f"{p}: {e}")
    return {
        "checked": len(sampled),
        "modes": dict(modes),
        "sizes": {f"{w}x{h}": n for (w, h), n in sizes.items()},
        "corrupt": bad,
    }


def sample_grid(df: pd.DataFrame, n_per_class: int = 8, seed: int = SEED) -> Path:
    """Pick ``n_per_class`` random images per class and render them in a grid.

    Plan calls for 8-12 per class. We default to 8: enough to spot label noise,
    still readable in a single figure.
    """
    rng = random.Random(seed)
    images, labels = [], []
    for cls_idx, cname in enumerate(CLASS_NAMES):
        sub = df[df["class"] == cname]["filepath"].tolist()
        rng.shuffle(sub)
        for p in sub[:n_per_class]:
            with Image.open(p) as img:
                arr = np.asarray(img.convert("RGB").resize((128, 128)), dtype=np.uint8)
            images.append(arr)
            labels.append(cls_idx)
    return plot_sample_grid(images, labels, n_per_class=n_per_class,
                            save_name="01_sample_grid",
                            title=f"{n_per_class} random samples per class (resized to 128x128)")


def run_inspection() -> dict:
    log = get_logger("inspection")
    t0 = time.time()
    log.info("Walking data tree at %s", DATA_ROOT)
    df = list_images(DATA_ROOT)
    log.info("Found %d image files total", len(df))

    counts = class_counts(df)
    counts_fig = plot_class_counts(counts, title="Images per class (raw dataset)")
    log.info("Saved class counts figure: %s", counts_fig)

    log.info("Probing image modes / sizes (sampled)")
    probe = probe_images(df, n_per_class=50)
    log.info("Modes seen: %s", probe["modes"])
    log.info("Sizes seen: %s", probe["sizes"])
    if probe["corrupt"]:
        log.warning("Found %d corrupt files", len(probe["corrupt"]))
    else:
        log.info("No corrupt images found in the sampled set")

    grid_path = sample_grid(df, n_per_class=4)
    log.info("Saved sample grid: %s", grid_path)

    elapsed = time.time() - t0
    summary = {
        "data_root": str(DATA_ROOT),
        "total_images": int(len(df)),
        "class_counts": counts,
        "image_probe": probe,
        "figures": {
            "class_counts": str(counts_fig),
            "sample_grid": str(grid_path),
        },
        "elapsed_seconds": round(elapsed, 2),
    }
    out = REPORTS_DIR / "01_data_inspection_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Wrote machine-readable summary: %s", out)
    return summary


if __name__ == "__main__":
    s = run_inspection()
    print(json.dumps(s, indent=2))
