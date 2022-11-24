from tests_real_data.processing_results.python_scripts.pairwise_comparison_matrices import *
import os
from utilities.small_functions import mkDir_if_not
import shutil

comparison_test = "Student"
metric = "balanced_accuracy"

output_directory_figures = mkDir_if_not("tests_real_data/figures_for_paper/comparison_matrices/{}/{}".format(comparison_test, experiment_name))

for dataset in dataset_names_paper_full_list:
    matrix_file = get_matrix_path_latex(dataset, comparison_test, None)
    shutil.copyfile(matrix_file, os.path.join(output_directory_figures, "{}.txt".format(dataset)))

output_directory_benchmark = mkDir_if_not("tests_real_data/figures_for_paper/benchmark/{}".format(comparison_test))
choosing_mode="one_standard_error_rule"
metric_specific_benchmark_dir = mkDir_if_not(os.path.join(get_benchmark_dir(experiment_name), metric, choosing_mode))
benchmark_file = os.path.join(metric_specific_benchmark_dir, "results_dataset_wise{}_with_significance.txt".format(
                    comparison_test_suffix[comparison_test]))
shutil.copyfile(benchmark_file, os.path.join(output_directory_benchmark, "{}.txt".format(experiment_name)))