#! /bin/bash
cd /media/rokny/DATA3/Jade/HTG_Model_ICI_Prediction/src

#python benchmark_random_forest.py \
#  --dataset-root /media/rokny/DATA6/Jade/sc_training_dataset_latest/sample_h5ad/Tumor/pre/ \
#  --split-json /media/rokny/DATA3/Anish/HTG_Model_ICI_Results/round2/optimal2/split.json \
#  --gene-universe /media/rokny/DATA3/Anish/HTG_Model_ICI_Results/round2/optimal2/gene_universe.txt \
#  --output-dir /media/rokny/DATA3/Jade/HTG_Training_Results/round2_benchmark_rf

python benchmark_linear_regression.py \
  --dataset-root /media/rokny/DATA6/Jade/sc_training_dataset_latest/sample_h5ad/Tumor/pre/ \
  --split-json /media/rokny/DATA3/Anish/HTG_Model_ICI_Results/round2/optimal2/split.json \
  --gene-universe /media/rokny/DATA3/Anish/HTG_Model_ICI_Results/round2/optimal2/gene_universe.txt \
  --output-dir /media/rokny/DATA3/Jade/HTG_Training_Results/round2_benchmark_linreg

python benchmark_mlp.py \
  --dataset-root /media/rokny/DATA6/Jade/sc_training_dataset_latest/sample_h5ad/Tumor/pre/ \
  --split-json /media/rokny/DATA3/Anish/HTG_Model_ICI_Results/round2/optimal2/split.json \
  --gene-universe /media/rokny/DATA3/Anish/HTG_Model_ICI_Results/round2/optimal2/gene_universe.txt \
  --output-dir /media/rokny/DATA3/Jade/HTG_Training_Results/round2_benchmark_mlp

python benchmark_gnn.py \
  --dataset-root /media/rokny/DATA6/Jade/sc_training_dataset_latest/sample_h5ad/Tumor/pre/ \
  --split-json /media/rokny/DATA3/Anish/HTG_Model_ICI_Results/round2/optimal2/split.json \
  --gene-universe /media/rokny/DATA3/Anish/HTG_Model_ICI_Results/round2/optimal2/gene_universe.txt \
  --output-dir /media/rokny/DATA3/Jade/HTG_Training_Results/round2_benchmark_gnn