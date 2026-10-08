#! /bin/bash
cd /media/rokny/DATA3/Jade/HTG_Model_ICI_Prediction/src

python benchmark/benchmark_random_forest.py \
  --dataset-root /media/rokny/DATA6/Jade/sc_training_dataset_latest/sample_h5ad/Tumor/pre/ \
  --split-json /media/rokny/DATA3/Anish/experiments_for_performance/baseline/split.json \
  --gene-universe /media/rokny/DATA3/Anish/experiments_for_performance/baseline/gene_universe.txt \
  --output-dir /media/rokny/DATA3/Jade/HTG_Training_Results/round2_benchmark_rf

python benchmark/benchmark_linear_regression.py \
  --dataset-root /media/rokny/DATA6/Jade/sc_training_dataset_latest/sample_h5ad/Tumor/pre/ \
  --split-json /media/rokny/DATA3/Anish/experiments_for_performance/baseline/split.json \
  --gene-universe /media/rokny/DATA3/Anish/experiments_for_performance/baseline/gene_universe.txt \
  --output-dir /media/rokny/DATA3/Jade/HTG_Training_Results/round2_benchmark_linreg

python benchmark/benchmark_mlp.py \
  --dataset-root /media/rokny/DATA6/Jade/sc_training_dataset_latest/sample_h5ad/Tumor/pre/ \
  --split-json /media/rokny/DATA3/Anish/experiments_for_performance/baseline/split.json \
  --gene-universe /media/rokny/DATA3/Anish/experiments_for_performance/baseline/gene_universe.txt \
  --output-dir /media/rokny/DATA3/Jade/HTG_Training_Results/round2_benchmark_mlp

python benchmark/benchmark_gnn.py \
  --dataset-root /media/rokny/DATA6/Jade/sc_training_dataset_latest/sample_h5ad/Tumor/pre/ \
        --split-json /media/rokny/DATA3/Anish/experiments_for_performance/baseline/split.json \
  --gene-universe /media/rokny/DATA3/Anish/experiments_for_performance/baseline/gene_universe.txt \
  --output-dir /media/rokny/DATA3/Jade/HTG_Training_Results/round2_benchmark_gnn