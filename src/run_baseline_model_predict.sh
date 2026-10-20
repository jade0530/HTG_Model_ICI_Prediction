#!/bin/bash
# Score the unseen cohort with the finished HTG baselines that use the
# benchmark splits. Folders live under Anish's seeds_splits_exp/baseline:
#   seed_<training_seed>_split_<n>
# split 1 = 9485, split 2 = 1043, split 3 = 1045, split 4 = 8173, split 5 = 2873.
# seed_*_split_2_gene_symbols and trial are not part of this set.
set -euo pipefail
cd /media/rokny/DATA3/Jade/HTG-interpretability/src

PYTHON=/home/rokny/miniconda3/envs/htg_model/bin/python
UNSEEN=/media/rokny/DATA6/Jade/validation_cohort_unseen_final
ROOT=/media/rokny/DATA3/Anish/HTG_Model_ICI_Results/seeds_splits_exp/baseline
SPLIT_SEEDS=(9485 1043 1045 8173 2873)
FOLDERS=(1 2 3 4 5)
TRAINING_SEEDS=(42 43 44)

for folder in "${FOLDERS[@]}"; do
  for training_seed in "${TRAINING_SEEDS[@]}"; do
    split_seed=${SPLIT_SEEDS[$folder-1]}
    out="$ROOT/seed_${training_seed}_split_${folder}"
    if [[ -f "$out/unseen/predict.json" ]]; then
      continue
    fi
    if [[ ! -f "$out/best.pt" ]]; then
      echo "missing checkpoint: $out/best.pt" >&2
      exit 1
    fi
    echo "predict split=${folder} split_seed=${split_seed} training_seed=${training_seed}"
    # Split 5 seeds 43 and 44 were trained on gene symbols. The other
    # baselines in this folder were trained on Ensembl IDs.
    predict_cmd=(
      "$PYTHON" -m src.train predict
      --checkpoint "$out"
      --dataset-root "$UNSEEN"
      --output-dir "$out/unseen"
      --tissue all
      --ici-phase all
    )
    if [[ "$folder" != "5" || "$training_seed" == "42" ]]; then
      predict_cmd+=(--no-gene-symbols)
    fi
    "${predict_cmd[@]}"
  done
done
echo "predict complete"
