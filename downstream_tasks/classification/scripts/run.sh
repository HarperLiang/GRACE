#!/bin/bash
set -e

# ============ Config ============
PROJECT_ROOT="/path/to/project/root"  # Adjust to your project root
FEATURE_PATH="/path/to/feature/files"  # Adjust the feature directory as needed

FEATURES=("GRACE")
SEEDS=(1234) 
STUDIES=("demo")

NUM_EPOCH=50
BATCH_SIZE=1
LR=2e-4
GPU_ID=0

cd "${PROJECT_ROOT}"
mkdir -p logs results

for study in "${STUDIES[@]}"; do
  for feature in "${FEATURES[@]}"; do
    for seed in "${SEEDS[@]}"; do

      LOG_FILE="logs/${study}_${feature}_seed${seed}.log"

      CUDA_VISIBLE_DEVICES=${GPU_ID} python "${PROJECT_ROOT}/main.py" \
        --model ABMIL \
        --root "${FEATURE_PATH}" \
        --feature "${feature}" \
        --study "${study}" \
        --excel_file "${PROJECT_ROOT}/dataset_excel/${study}.xlsx" \
        --num_epoch ${NUM_EPOCH} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --seed ${seed} \
        --tqdm \
        > "${LOG_FILE}" 2>&1

    done
  done
done