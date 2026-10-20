#!/bin/bash
# Score the unseen cohort for each finished selected-benchmark run.
# Safe to run beside run_selected_benchmarks.sh: a directory is scored only
# after its training process has exited and model.joblib is present.
set -euo pipefail
cd /media/rokny/DATA3/Jade/HTG-interpretability/src

PYTHON=/home/rokny/miniconda3/envs/htg_model/bin/python
UNSEEN=/media/rokny/DATA6/Jade/validation_cohort_unseen_final
ROOT=/media/rokny/DATA3/Jade/HTG_Training_Results/1016_selected_benchmarks
MODELS=(random_forest linear_regression mlp gnn)
SPLIT_SEEDS=(9485 1043 1045 8173 2873)
TRAINING_SEEDS=(42 43 44)

training_this_dir() {
  local out="$1"
  local pid cmd
  for pid in $(pgrep -f "benchmark_.*--output-dir ${out}( |$)" || true); do
    cmd=$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)
    if [[ "$cmd" == *"--output-dir ${out}"* && "$cmd" != *" predict "* ]]; then
      return 0
    fi
  done
  return 1
}

pending=1
while [[ "$pending" -gt 0 ]]; do
  pending=0
  for model in "${MODELS[@]}"; do
    for split_seed in "${SPLIT_SEEDS[@]}"; do
      for training_seed in "${TRAINING_SEEDS[@]}"; do
        out="$ROOT/$model/split_${split_seed}/seed_${training_seed}"
        if [[ -f "$out/unseen/predict.json" ]]; then
          continue
        fi
        pending=$((pending + 1))
        if [[ ! -f "$out/model.joblib" ]]; then
          continue
        fi
        if training_this_dir "$out"; then
          continue
        fi
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
      done
    done
  done
  if [[ "$pending" -gt 0 ]]; then
    echo "waiting pending=$pending"
    sleep 60
  fi
done
echo "predict complete"
