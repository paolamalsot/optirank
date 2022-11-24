#!/bin/bash

experiment_names=(simple_tasks single_source multiple_sources)
comparison_test="student"

conda activate optirank
echo "$(pwd)"

outdir=scripts_results
mkdir -p $outdir
cd ../../..
export PYTHONPATH="$(pwd)"

for experiment_name in "${experiment_names[@]}"; do
  #renaming config.py for the correct experiment
  cp tests_real_data/processing_results/funs/config_files/config_${experiment_name}.py tests_real_data/processing_results/funs/config_files/config.py
  outfile=tests_real_data/processing_results/bash_scripts/scripts_results/results_processing_on_local_${experiment_name}.out

  echo "AGGLOMERATION RESULTS"
  eval python tests_real_data/processing_results/python_scripts/results_batched_agglomerate.py >> $outfile 1>&2
  echo "COMPARISON MATRICES"
  eval python tests_real_data/processing_results/python_scripts/pairwise_comparison_matrices.py >> $outfile 1>&2
  echo "RESULTS BENCHMARK"
  eval python tests_real_data/processing_results/python_scripts/results_benchmark_00.py >> $outfile 1>&2
  echo "MAIN TABLE"
  eval python tests_real_data/processing_results/python_scripts/results_benchmark_01.py >> $outfile 1>&2
  echo "TIMING"
  eval python tests_real_data/processing_results/python_scripts/results_timing.py >> $outfile 1>&2
  echo "n_genes"
  eval python tests_real_data/processing_results/investigations/gene_selection_investigation/sparsity_comparison_with_log_regr.py >> $outfile 1>&2

  #putting together the comparison_results
  eval python tests_real_data/processing_results/python_scripts/putting_together_pairwise_comparisons.py

done