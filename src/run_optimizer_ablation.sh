#!/bin/bash
# Experiment 6: Adam vs AdamW, followed by unseen-cohort evaluation.
set -euo pipefail
cd /media/rokny/DATA3/Jade/HTG_Model_ICI_Prediction/src

PYTHON="${PYTHON:-/home/rokny/miniconda3/envs/htg_model/bin/python}"
DATA="${DATA:-/media/rokny/DATA6/Jade/sc_training_dataset_latest/sample_h5ad/Tumor/pre}"
OUT="${OUT:-/media/rokny/DATA3/Jade/HTG_Training_Results/1001_optimizer_ablation}"
UNSEEN="${UNSEEN:-/media/rokny/DATA6/Jade/validation_cohort_unseen_final}"

mkdir -p "$OUT"
"$PYTHON" -u tests/test_run_parameters.py --experiment 6 \
  --dataset-root "$DATA" \
  --output-root "$OUT"

for optimizer in adam adamw; do
  "$PYTHON" -u -m src.train predict \
    --checkpoint "$OUT/$optimizer/best.pt" \
    --dataset-root "$UNSEEN" \
    --output-dir "$OUT/$optimizer/unseen" \
    --tissue all \
    --ici-phase all
done
