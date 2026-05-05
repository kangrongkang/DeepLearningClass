#!/usr/bin/env bash
# Reviewer-requested ablation driver: EfficientNet-B0 at 128x128 to disambiguate
# the architecture-vs-resolution contribution.
set -e
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
export KERAS_BACKEND=torch
LOG="outputs/reports/effnet_128_run.log"
mkdir -p outputs/reports

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
echo "[$(stamp)] === EfficientNet-B0 @ 128x128 ablation START ===" | tee -a "$LOG"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv | tee -a "$LOG"

python -c "import sys; sys.path.insert(0, '.'); from src.effnet_128_ablation import main; main()" 2>&1 | tee -a "$LOG"

echo "[$(stamp)] === EfficientNet-B0 @ 128x128 ablation DONE ===" | tee -a "$LOG"
