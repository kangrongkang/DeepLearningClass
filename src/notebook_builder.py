"""
Helpers for assembling .ipynb files from a list of (kind, source) tuples.

Each notebook script (e.g. ``build_01_data_inspection.py``) calls ``build_notebook``
with a list of cells; the writer ensures the JSON matches Jupyter's nbformat 4.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def _cell_id(prefix: str, source: str) -> str:
    """Stable cell id derived from the source — keeps diffs small across rebuilds."""
    h = hashlib.md5(source.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}-{h}"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _cell_id("md", source),
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": _cell_id("code", source),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_notebook(cells: Iterable[dict], path: Path | str,
                   kernel_name: str = "python3", kernel_display: str = "Python 3") -> Path:
    nb = {
        "cells": list(cells),
        "metadata": {
            "kernelspec": {
                "display_name": kernel_display,
                "language": "python",
                "name": kernel_name,
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    return path
