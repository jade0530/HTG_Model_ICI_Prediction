#!/bin/bash
set -euo pipefail
cd /media/rokny/DATA3/Jade/HTG_Model_ICI_Prediction/src

PYTHON=/home/rokny/miniconda3/envs/htg_model/bin/python
DATA=/media/rokny/DATA6/Jade/sc_training_dataset_latest/sample_h5ad/Tumor/pre
UNSEEN=/media/rokny/DATA6/Jade/validation_cohort_unseen_final
ROOT=/media/rokny/DATA3/Jade/HTG_Training_Results/1002_repeated_benchmarks_rerun
SPLITS=/media/rokny/DATA3/Jade/HTG_Training_Results/1002_repeated_benchmarks/splits

for model in random_forest linear_regression mlp gnn; do
  "$PYTHON" -u benchmark/run_repeated_benchmark.py \
    --model "$model" \
    --dataset-root "$DATA" \
    --split-dir "$SPLITS" \
    --output-dir "$ROOT/$model" \
    --training-seeds 42 43 44 \
    --n-hvg 3000 \
    --unseen-dataset-root "$UNSEEN"
done
