#! /bin/bash
cd /media/rokny/DATA3/Jade/HTG_Model_ICI_Prediction/src
# 41 samples from tumor pre cohort
python benchmark_random_forest.py predict \
  --checkpoint /media/rokny/DATA3/Jade/HTG_Training_Results/round2_benchmark_rf \
  --dataset-root /media/rokny/DATA6/Jade/validation_cohort_unseen_final \
  --output-dir /media/rokny/DATA3/Jade/HTG_Training_Results/round2_benchmark_rf/predict

python benchmark_linear_regression.py predict \
  --checkpoint /media/rokny/DATA3/Jade/HTG_Training_Results/round2_benchmark_linreg \
  --dataset-root /media/rokny/DATA6/Jade/validation_cohort_unseen_final \
  --output-dir /media/rokny/DATA3/Jade/HTG_Training_Results/round2_benchmark_linreg/predict

python benchmark_mlp.py predict \
  --checkpoint /media/rokny/DATA3/Jade/HTG_Training_Results/round2_benchmark_mlp \
  --dataset-root /media/rokny/DATA6/Jade/validation_cohort_unseen_final \
  --output-dir /media/rokny/DATA3/Jade/HTG_Training_Results/round2_benchmark_mlp/predict

#python benchmark_gnn.py predict \
#  --checkpoint /media/rokny/DATA3/Jade/HTG_Training_Results/round2_benchmark_gnn \
#  --dataset-root /media/rokny/DATA6/Jade/validation_cohort_unseen_final \
#  --output-dir /media/rokny/DATA3/Jade/HTG_Training_Results/round2_benchmark_gnn/predict