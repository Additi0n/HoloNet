#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-RGBA}"
OUTPUT_DIR="${OUTPUT_DIR:-runs}"
DEVICE="${DEVICE:-cuda:0}"
CONFIG_PATH="${CONFIG_PATH:-experiments/ablation_suite.json}"

${PYTHON_BIN} run_experiment.py \
  --config "${CONFIG_PATH}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --all-datasets