#!/bin/bash

experiment_names=(simple_tasks single_source multiple_sources)

outdir=scripts_results

mkdir -p $outdir
cd ../../..
export PYTHONPATH="$(pwd)"

for experiment_name in "${experiment_names[@]}"; do
  arr=$(awk -F',' -v a="$experiment_name" '{ if ($1 == a){print $2} }' tests_real_data/meta_experiments.csv)
  datasets=($arr)

  #renaming config.py for the correct experiment
  cp tests_real_data/processing_results/funs/config_files/config_${experiment_name}.py tests_real_data/processing_results/funs/config_files/config.py
  for dataset in "${datasets[@]}"; do
    arr=$(awk -F',' -v a="$dataset" '{ if ($1 == a){print $2} }' ../../meta_datasets.csv)
    classes_of_interest=($arr)
    for class_of_interest in "${classes_of_interest[@]}"; do
      python tests_real_data/processing_results/python_scripts/results_batched_00.py ${dataset} ${class_of_interest}
      python tests_real_data/processing_results/python_scripts/results_batched_01.py ${dataset} ${class_of_interest}
    done
  done
done