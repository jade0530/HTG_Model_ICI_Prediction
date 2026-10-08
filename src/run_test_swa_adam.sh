#!/bin/bash 

# late swa + adam
python -m src.train   --dataset-root /media/rokny/DATA6/Jade/sc_training_dataset_latest/sample_h5ad/Tumor/pre \
    --output-dir /media/rokny/DATA3/Jade/HTG_Training_Results/1001_adam_late_swa_128 \
    --encoder sage  \
    --pooling mean  \
     --readout cell  \
     --num-gnn-layers 2  \
     --gat-heads 4  \
     --hidden-dim 128  \
     --dropout 0.1  \
     --gene-strategy hvg  \
     --n-hvg 5000  \
     --sampling-mode random  \
     --num-cells 128  \
     --batch-size 2  \
     --epochs 60  \
     --lr 0.001    \
     --swa  \
     --swa-start 45  \
     --swa-freq 1   \
     --swa-lr 0.001  \
     --val-fraction 0.25   \
     --test-fraction 0.0   \
     --no-early-stopping   \
     --threshold-strategy max_f1   \
     --threshold 0.5     \
     --cancer-proportional-split \
     --split-seed 42 \
     --training-seed 1000 \
     --optimizer adam

# early swa + adamw
python -m src.train   --dataset-root /media/rokny/DATA6/Jade/sc_training_dataset_latest/sample_h5ad/Tumor/pre \
    --output-dir /media/rokny/DATA3/Jade/HTG_Training_Results/1001_adamw_early_swa_128 \
    --encoder sage  \
    --pooling mean  \
     --readout cell  \
     --num-gnn-layers 2  \
     --gat-heads 4  \
     --hidden-dim 128  \
     --dropout 0.1  \
     --gene-strategy hvg  \
     --n-hvg 5000  \
     --sampling-mode random  \
     --num-cells 128  \
     --batch-size 2  \
     --epochs 60  \
     --lr 0.001    \
     --swa  \
     --swa-start 15  \
     --swa-freq 1   \
     --swa-lr 0.001  \
     --val-fraction 0.25   \
     --test-fraction 0.0   \
     --no-early-stopping   \
     --threshold-strategy max_f1   \
     --threshold 0.5     \
     --cancer-proportional-split \
     --split-seed 42 \
     --training-seed 1000 \
     --optimizer adamw

# late swa + adamw
python -m src.train   --dataset-root /media/rokny/DATA6/Jade/sc_training_dataset_latest/sample_h5ad/Tumor/pre \
    --output-dir /media/rokny/DATA3/Jade/HTG_Training_Results/1001_adamw_late_swa_128 \
    --encoder sage  \
    --pooling mean  \
     --readout cell  \
     --num-gnn-layers 2  \
     --gat-heads 4  \
     --hidden-dim 128  \
     --dropout 0.1  \
     --gene-strategy hvg  \
     --n-hvg 5000  \
     --sampling-mode random  \
     --num-cells 128  \
     --batch-size 2  \
     --epochs 60  \
     --lr 0.001    \
     --swa  \
     --swa-start 45  \
     --swa-freq 1   \
     --swa-lr 0.001  \
     --val-fraction 0.25   \
     --test-fraction 0.0   \
     --no-early-stopping   \
     --threshold-strategy max_f1   \
     --threshold 0.5     \
     --cancer-proportional-split \
     --split-seed 42 \
     --training-seed 1000 \
     --optimizer adamw

# no swa + adam
python -m src.train   --dataset-root /media/rokny/DATA6/Jade/sc_training_dataset_latest/sample_h5ad/Tumor/pre \
    --output-dir /media/rokny/DATA3/Jade/HTG_Training_Results/1001_adamw_no_swa_128 \
    --encoder sage  \
    --pooling mean  \
     --readout cell  \
     --num-gnn-layers 2  \
     --gat-heads 4  \
     --hidden-dim 128  \
     --dropout 0.1  \
     --gene-strategy hvg  \
     --n-hvg 5000  \
     --sampling-mode random  \
     --num-cells 128  \
     --batch-size 2  \
     --epochs 60  \
     --lr 0.001    \
     --val-fraction 0.25   \
     --test-fraction 0.0   \
     --no-early-stopping   \
     --threshold-strategy max_f1   \
     --threshold 0.5     \
     --cancer-proportional-split \
     --split-seed 42 \
     --training-seed 1000 \
     --optimizer adam

# no swa + adamw
python -m src.train   --dataset-root /media/rokny/DATA6/Jade/sc_training_dataset_latest/sample_h5ad/Tumor/pre \
    --output-dir /media/rokny/DATA3/Jade/HTG_Training_Results/1001_adamw_no_swa_128 \
    --encoder sage  \
    --pooling mean  \
     --readout cell  \
     --num-gnn-layers 2  \
     --gat-heads 4  \
     --hidden-dim 128  \
     --dropout 0.1  \
     --gene-strategy hvg  \
     --n-hvg 5000  \
     --sampling-mode random  \
     --num-cells 128  \
     --batch-size 2  \
     --epochs 60  \
     --lr 0.001    \
     --val-fraction 0.25   \
     --test-fraction 0.0   \
     --no-early-stopping   \
     --threshold-strategy max_f1   \
     --threshold 0.5     \
     --cancer-proportional-split \
     --split-seed 42 \
     --training-seed 1000 \
     --optimizer adamw