#!/bin/bash
set -euo pipefail
cd /media/rokny/DATA3/Jade/HTG-interpretability/src

PYTHON=/home/rokny/miniconda3/envs/htg_model/bin/python
DATA=/media/rokny/DATA6/Jade/sc_training_dataset_latest/sample_h5ad/Tumor/pre
UNSEEN=/media/rokny/DATA6/Jade/validation_cohort_unseen_final
SPLITS=/media/rokny/DATA3/Jade/HTG_Training_Results/1002_repeated_benchmarks/splits
PARAMS=/media/rokny/DATA3/Jade/HTG_Training_Results/1014_repeated_benchmarks_hvg3000
ROOT=/media/rokny/DATA3/Jade/HTG_Training_Results/1016_selected_benchmarks

N_HVG=3000
SPLIT_SEEDS=(9485 1043 1045 8173 2873)
TRAINING_SEEDS=(42 43 44)

declare -A SPLIT_JSON=(
  [9485]="$SPLITS/split_001.json"
  [1043]="$SPLITS/split_002.json"
  [1045]="$SPLITS/split_003.json"
  [8173]="$SPLITS/split_004.json"
  [2873]="$SPLITS/split_005.json"
)

for seed in "${SPLIT_SEEDS[@]}"; do
  found="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["split_seed"])' "${SPLIT_JSON[$seed]}")"
  if [[ "$found" != "$seed" ]]; then
    echo "split file ${SPLIT_JSON[$seed]} has split_seed=$found, expected $seed" >&2
    exit 1
  fi
done

for model in random_forest linear_regression mlp gnn; do
  params="$PARAMS/$model/selected_params.json"
  for split_seed in "${SPLIT_SEEDS[@]}"; do
    for training_seed in "${TRAINING_SEEDS[@]}"; do
      out="$ROOT/$model/split_${split_seed}/seed_${training_seed}"
      echo "model=$model split_seed=$split_seed training_seed=$training_seed"
      cmd=(
        "$PYTHON" -u "benchmark/benchmark_${model}.py"
        --dataset-root "$DATA"
        --split-json "${SPLIT_JSON[$split_seed]}"
        --params-json "$params"
        --output-dir "$out"
        --n-hvg "$N_HVG"
        --seed "$training_seed"
      )
      if [[ "$model" != "random_forest" ]]; then
        cmd+=(--epochs 60)
      fi
      if [[ "$model" == "gnn" ]]; then
        cmd+=(--num-cells 128 --batch-size 2 --device cuda)
      fi
      if [[ ! -f "$out/model.joblib" ]]; then
        "${cmd[@]}"
      else
        echo "train exists model=$model split_seed=$split_seed training_seed=$training_seed"
      fi
      if [[ ! -f "$out/unseen/predict.json" ]]; then
        echo "predict model=$model split_seed=$split_seed training_seed=$training_seed"
        predict_cmd=(
          "$PYTHON" -u "benchmark/benchmark_${model}.py"
          predict
          --checkpoint "$out"
          --dataset-root "$UNSEEN"
          --output-dir "$out/unseen"
          --tissue all
          --ici-phase all
        )
        if [[ "$model" == "gnn" ]]; then
          predict_cmd+=(--device cpu --batch-size 2)
        fi
        "${predict_cmd[@]}"
      fi
    done
  done
done
