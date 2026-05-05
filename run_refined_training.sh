#!/usr/bin/env bash
# Drives the Phase D refined-recipe training on the local RTX 5080.
# Three EfficientNet-B0 configs sequentially: a (bare) -> b (+Mixup) -> c (+SWA+TTA).
set -e

cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
export KERAS_BACKEND=torch
LOG="outputs/reports/refined_full_run.log"
mkdir -p outputs/reports

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
echo "[$(stamp)] === REFINED TRAINING RUN START ===" | tee -a "$LOG"
echo "[$(stamp)] GPU info:" | tee -a "$LOG"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv | tee -a "$LOG"

run_cfg () {
  local cfg="$1"
  echo "[$(stamp)] >>> running config $cfg" | tee -a "$LOG"
  python -m src.refined_train --config "$cfg" 2>&1 | tail -30 | tee -a "$LOG"
  echo "[$(stamp)] <<< done config $cfg" | tee -a "$LOG"
}

run_cfg a
run_cfg b
run_cfg c

# Regenerate the comparison + final-report artifacts now that new metrics exist
echo "[$(stamp)] Regenerating comparison + report..." | tee -a "$LOG"
python -m src.final_report 2>&1 | tee -a "$LOG"

echo "[$(stamp)] === REFINED TRAINING RUN COMPLETE ===" | tee -a "$LOG"
