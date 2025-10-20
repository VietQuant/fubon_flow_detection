#!/usr/bin/env bash
set -euo pipefail

# Editable parameters
PY="python"
OUT_DIR="results/tuning/exp1"

# Trial budgets
STAGE1_TRIALS=80
STAGE2_TRIALS=80

# Sampler config
N_STARTUP=25
SEED=42

# Optuna persistent storage and study names
S1_STORAGE="sqlite:///results/tuning/optuna_v3_stage1_exp1.db"
S1_NAME="fubon_v3_stage1_exp1"
S2_STORAGE="sqlite:///results/tuning/optuna_v3_stage2_exp1.db"
S2_NAME="fubon_v3_stage2_exp1"

# Ensure output directory exists
mkdir -p "$OUT_DIR"

# Run Optimizer v3
$PY algorithm/fubon_hyperparameter_tuning_ver3.py \
  --stage1-trials "$STAGE1_TRIALS" \
  --stage2-trials "$STAGE2_TRIALS" \
  --stage1-storage "$S1_STORAGE" \
  --stage1-study-name "$S1_NAME" \
  --stage2-storage "$S2_STORAGE" \
  --stage2-study-name "$S2_NAME" \
  --output-dir "$OUT_DIR" \
  --n-startup-trials "$N_STARTUP" \
  --random-seed "$SEED"

