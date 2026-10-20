#!/bin/bash
set -euo pipefail
cd /media/rokny/DATA3/Jade/HTG-interpretability/src

PYTHON=/home/rokny/miniconda3/envs/htg_model/bin/python
DATA=/media/rokny/DATA6/Jade/sc_training_dataset_latest/sample_h5ad/Tumor/pre
UNSEEN=/media/rokny/DATA6/Jade/validation_cohort_unseen_final
HVG=/media/rokny/DATA3/Jade/HTG-interpretability/outputs/fi_prior/none/gene_universe.txt
ROOT=/media/rokny/DATA3/Jade/HTG-interpretability/outputs/fi_modes
MODES=(baseline fi_topology fi_relation_aware)

run_one() {
  local seed="$1"
  local mode="$2"
  local out="$ROOT/seed_${seed}/${mode}"
  mkdir -p "$out"
  if [[ ! -f "$out/best.pt" ]]; then
    echo "train mode=$mode seed=$seed"
    "$PYTHON" -u -m src.train \
      --dataset-root "$DATA" \
      --output-dir "$out" \
      --encoder sage \
      --pooling mean \
      --readout cell \
      --num-gnn-layers 2 \
      --gat-heads 4 \
      --hidden-dim 128 \
      --dropout 0.1 \
      --gene-strategy hvg \
      --n-hvg 5000 \
      --hvg-file "$HVG" \
      --sampling-mode random \
      --num-cells 128 \
      --batch-size 2 \
      --epochs 60 \
      --lr 0.001 \
      --val-fraction 0.25 \
      --test-fraction 0.0 \
      --no-early-stopping \
      --threshold-strategy max_f1 \
      --threshold 0.5 \
      --cancer-proportional-split \
      --split-seed 1043 \
      --training-seed "$seed" \
      --gene-prior none \
      --fi-mode "$mode" \
      --device cuda
  else
    echo "train exists mode=$mode seed=$seed"
  fi
  if [[ ! -f "$out/unseen/predict.json" ]]; then
    echo "predict mode=$mode seed=$seed"
    "$PYTHON" -u -m src.train predict \
      --checkpoint "$out" \
      --dataset-root "$UNSEEN" \
      --output-dir "$out/unseen" \
      --tissue all \
      --ici-phase all \
      --device cuda
  else
    echo "predict exists mode=$mode seed=$seed"
  fi
}

for mode in "${MODES[@]}"; do
  run_one 1 "$mode"
done

for seed in 2 3; do
  for mode in "${MODES[@]}"; do
    run_one "$seed" "$mode"
  done
done

"$PYTHON" -m src.fi.summarize_fi_modes --root "$ROOT" --seeds 1 2 3
