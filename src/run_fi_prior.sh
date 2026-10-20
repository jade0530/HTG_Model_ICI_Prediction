#!/bin/bash
set -euo pipefail
cd /media/rokny/DATA3/Jade/HTG-interpretability/src

PYTHON=/home/rokny/miniconda3/envs/htg_model/bin/python
DATA=/media/rokny/DATA6/Jade/sc_training_dataset_latest/sample_h5ad/Tumor/pre
ROOT=/media/rokny/DATA3/Jade/HTG-interpretability/outputs/fi_prior

for prior in none fi_topology_frozen fi_semantic_frozen; do
  "$PYTHON" -u -m src.train \
    --dataset-root "$DATA" \
    --output-dir "$ROOT/$prior" \
    --encoder sage \
    --pooling mean \
    --readout cell \
    --num-gnn-layers 2 \
    --gat-heads 4 \
    --hidden-dim 128 \
    --dropout 0.1 \
    --gene-strategy hvg \
    --n-hvg 5000 \
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
    --training-seed 42 \
    --gene-prior "$prior" \
    --device cuda
done
