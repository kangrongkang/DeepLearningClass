#!/usr/bin/env bash
# Drives the full training pipeline on the local RTX 5080.
# Usage: bash run_full_training.sh
#
# Each notebook writes its own log under outputs/reports/<model>_train.log;
# the high-level progress is appended to outputs/reports/full_run.log.
set -e

cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
export KERAS_BACKEND=torch
LOG="outputs/reports/full_run.log"
mkdir -p outputs/reports

stamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(stamp)] === FULL TRAINING RUN START ===" | tee -a "$LOG"
echo "[$(stamp)] GPU info:" | tee -a "$LOG"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv | tee -a "$LOG"

run_nb () {
  local nb="$1"
  local timeout="$2"
  echo "[$(stamp)] >>> running $nb (timeout=${timeout}s)" | tee -a "$LOG"
  jupyter nbconvert --to notebook --execute "$nb" --output "$(basename "$nb")" \
    --ExecutePreprocessor.timeout="$timeout" 2>&1 | tail -5 | tee -a "$LOG"
  echo "[$(stamp)] <<< done $nb" | tee -a "$LOG"
}

run_nb notebooks/03_model_a_baseline_cnn.ipynb 21600   # 6h cap
run_nb notebooks/04_model_b_mobilenetv2.ipynb 21600
run_nb notebooks/05_model_c_vgg16.ipynb       21600
run_nb notebooks/06_model_comparison.ipynb     900
run_nb notebooks/07_grad_cam.ipynb            1800
run_nb notebooks/08_final_report.ipynb         900

# Regenerate the standalone Markdown report from artifacts
python -m src.final_report 2>&1 | tee -a "$LOG"

echo "[$(stamp)] === FULL TRAINING RUN COMPLETE ===" | tee -a "$LOG"
