"""
Boilerplate every notebook runs in its first cell so it can `from src.config import ...`
on either Colab or local Windows. Adjusts sys.path and (on Colab) clones / mounts.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def bootstrap() -> Path:
    """Add the project root to sys.path; return the project root."""
    here = Path(os.getcwd()).resolve()
    candidates = [here, *here.parents]
    for cand in candidates:
        if (cand / "src" / "config.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand
    raise RuntimeError(
        "Could not locate src/config.py. Make sure you launched the notebook "
        "from inside the Project_DeepLearningClass directory."
    )
