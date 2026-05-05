"""
Shared configuration for the Alzheimer's MRI classification project.

Used by every notebook so paths, seeds, hyperparameters, and class definitions stay consistent.
Works both locally (Windows) and on Google Colab — paths are auto-detected.
"""
from __future__ import annotations

import os
import random
from pathlib import Path


def detect_environment() -> str:
    """Return 'colab' or 'local'."""
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        return "local"


ENV = detect_environment()


def _resolve_project_root() -> Path:
    """Find the project root regardless of where a notebook is started from.

    Strategy: start from the current working directory, walk up looking for a
    folder that contains an 'archive' subdirectory or a 'notebooks' subdirectory.
    Falls back to a hard-coded Colab default when running on Colab.
    """
    if ENV == "colab":
        candidates = [
            Path("/content/Project_DeepLearningClass"),
            Path("/content/drive/MyDrive/Project_DeepLearningClass"),
        ]
        for c in candidates:
            if c.exists() and ((c / "archive").exists() or (c / "notebooks").exists()):
                return c
        raise FileNotFoundError(
            "Project root not found on Colab. Expected to find the project at one of: "
            f"{candidates}. Either clone the repo to /content/Project_DeepLearningClass "
            "or upload the project folder to Drive at "
            "/content/drive/MyDrive/Project_DeepLearningClass."
        )

    here = Path(os.getcwd()).resolve()
    for cand in [here, *here.parents]:
        if (cand / "archive").exists() or (cand / "notebooks").exists():
            return cand
    return here


PROJECT_ROOT = _resolve_project_root()
DATA_ROOT = PROJECT_ROOT / "archive" / "combined_images"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
SPLITS_DIR = OUTPUTS_DIR / "splits"

for d in [OUTPUTS_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, PREDICTIONS_DIR, SPLITS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# Class definitions — the order here is the canonical label index across the project.
CLASS_NAMES = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented",
]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}


# Image / training hyperparameters
IMG_SIZE = 128
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
BATCH_SIZE = 32

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

SEED = 42


def set_global_seed(seed: int = SEED) -> None:
    """Best-effort deterministic seeding across numpy/python/tf."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def summary() -> str:
    return (
        f"Environment: {ENV}\n"
        f"Project root: {PROJECT_ROOT}\n"
        f"Data root:    {DATA_ROOT}\n"
        f"Outputs:      {OUTPUTS_DIR}\n"
        f"Classes:      {CLASS_NAMES}\n"
        f"Image size:   {IMG_SIZE}x{IMG_SIZE}x{IMG_CHANNELS}\n"
        f"Seed:         {SEED}\n"
    )
