#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" "${ROOT_DIR}/main.py" \
  --study "demo_survival" \
  --feature "GRACE" \
  --root "${ROOT_DIR}/demo_features" \
  --excel_file "${ROOT_DIR}/dataset_excel/demo_survival_labels.csv" \
  --num_epoch 2 \
  --batch_size 1 \
  --patience 2 \
  --seed 7 \
  --output_dir "${ROOT_DIR}/result" \
  "$@"
