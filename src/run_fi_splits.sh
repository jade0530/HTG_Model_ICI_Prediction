#!/bin/bash
# FI modes on the benchmark splits and training seeds.
# Gene symbols, 5000 HVGs, and the sample-level HTG settings stay fixed.
# The first baseline run of each split selects the GeneUniverse; later runs reuse it.
set -euo pipefail
cd /media/rokny/DATA3/Jade/HTG-interpretability/src

PYTHON=/home/rokny/miniconda3/envs/htg_model/bin/python
DATA=/media/rokny/DATA6/Jade/sc_training_dataset_latest/sample_h5ad/Tumor/pre
UNSEEN=/media/rokny/DATA6/Jade/validation_cohort_unseen_final
ROOT=/media/rokny/DATA3/Jade/HTG-interpretability/outputs/fi_splits
MODES=(baseline fi_topology fi_relation_aware)
SPLIT_SEEDS=(9485 1043 1045 8173 2873)
TRAINING_SEEDS=(42 43 44)

for split_seed in "${SPLIT_SEEDS[@]}"; do
  universe="$ROOT/split_${split_seed}/seed_42/baseline/gene_universe.txt"
  for training_seed in "${TRAINING_SEEDS[@]}"; do
    for mode in "${MODES[@]}"; do
      out="$ROOT/split_${split_seed}/seed_${training_seed}/${mode}"
      mkdir -p "$out"
      echo "train mode=$mode split_seed=$split_seed training_seed=$training_seed"
      if [[ ! -f "$out/best.pt" ]]; then
        cmd=(
          "$PYTHON" -u -m src.train
          --dataset-root "$DATA"
          --output-dir "$out"
          --encoder sage
          --pooling mean
          --readout cell
          --num-gnn-layers 2
          --gat-heads 4
          --hidden-dim 128
          --dropout 0.1
          --gene-strategy hvg
          --n-hvg 5000
          --sampling-mode random
          --num-cells 128
          --batch-size 2
          --epochs 60
          --lr 0.001
          --val-fraction 0.25
          --test-fraction 0.0
          --no-early-stopping
          --threshold-strategy max_f1
          --threshold 0.5
          --cancer-proportional-split
          --split-seed "$split_seed"
          --training-seed "$training_seed"
          --gene-prior none
          --fi-mode "$mode"
          --device cuda
        )
        if [[ -f "$universe" && ! ( "$mode" == "baseline" && "$training_seed" == "42" ) ]]; then
          cmd+=(--hvg-file "$universe")
        fi
        "${cmd[@]}"
      else
        echo "train exists mode=$mode split_seed=$split_seed training_seed=$training_seed"
      fi
      if [[ ! -f "$out/unseen/predict.json" ]]; then
        echo "predict mode=$mode split_seed=$split_seed training_seed=$training_seed"
        "$PYTHON" -u -m src.train predict \
          --checkpoint "$out" \
          --dataset-root "$UNSEEN" \
          --output-dir "$out/unseen" \
          --tissue all \
          --ici-phase all \
          --device cuda
      fi
    done
  done
done

"$PYTHON" - <<'PY'
import csv, json
from pathlib import Path
root = Path("/media/rokny/DATA3/Jade/HTG-interpretability/outputs/fi_splits")
rows = []
for split in (9485, 1043, 1045, 8173, 2873):
    for seed in (42, 43, 44):
        for mode in ("baseline", "fi_topology", "fi_relation_aware"):
            run = root / f"split_{split}" / f"seed_{seed}" / mode
            val = next(row for row in csv.DictReader((run / "results.csv").open()) if row["split"] == "val")
            early = json.loads((run / "early_stopping.json").read_text())
            test = json.loads((run / "unseen" / "predict.json").read_text())["metrics"]
            rows.append({
                "mode": mode,
                "split_seed": split,
                "training_seed": seed,
                "best_epoch": early["best_epoch"],
                "val_auprc": val["auprc"],
                "val_auroc": val["auroc"],
                "test_auprc": test["auprc"],
                "test_auroc": test["auroc"],
                "test_f1": test["f1"],
                "test_balanced_acc": test["balanced_acc"],
            })
out = root / "summary.csv"
with out.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {out}")
PY
