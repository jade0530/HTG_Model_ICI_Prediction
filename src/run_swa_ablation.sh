#!/bin/bash
# Baseline vs Baseline+SWA using the current HTG training command.
# After both runs, score the unseen cohort with each retained checkpoint.
# The val-tuned threshold stored in the checkpoint is reused; unseen is not used to pick a cutoff.
set -euo pipefail
cd /media/rokny/DATA3/Jade/HTG_Model_ICI_Prediction/src

PYTHON="${PYTHON:-/home/rokny/miniconda3/envs/htg_model/bin/python}"
DATA="${DATA:-/media/rokny/DATA6/Jade/sc_training_dataset_latest/sample_h5ad/Tumor/pre}"
OUT="${OUT:-/media/rokny/DATA3/Jade/HTG_Training_Results/0930_swa_ablation}"
UNSEEN="${UNSEEN:-/media/rokny/DATA6/Jade/validation_cohort_unseen_final}"

mkdir -p "$OUT"
echo "Starting SWA ablation -> $OUT"

"$PYTHON" -u tests/test_run_parameters.py --experiment 5 \
  --dataset-root "$DATA" \
  --output-root "$OUT"

echo "Scoring unseen cohort with retained checkpoints"
"$PYTHON" -u -m src.train predict \
  --checkpoint "$OUT/baseline/best.pt" \
  --dataset-root "$UNSEEN" \
  --output-dir "$OUT/baseline/unseen" \
  --tissue all \
  --ici-phase all

"$PYTHON" -u -m src.train predict \
  --checkpoint "$OUT/swa/best.pt" \
  --dataset-root "$UNSEEN" \
  --output-dir "$OUT/swa/unseen_best" \
  --tissue all \
  --ici-phase all

"$PYTHON" -u -m src.train predict \
  --checkpoint "$OUT/swa/swa.pt" \
  --dataset-root "$UNSEEN" \
  --output-dir "$OUT/swa/unseen_swa" \
  --tissue all \
  --ici-phase all

echo "SWA ablation finished. Compare $OUT/swa/ablation_compare.csv and unseen predict.json files."
