import os
from utilities.small_functions import mkDir_if_not
from tests_real_data.loading_data import get_output_dir

merged_datasets_dir = mkDir_if_not("tests_real_data/merged_dataset_results")

def get_benchmark_dir(experiment_name):
    return mkDir_if_not(os.path.join(merged_datasets_dir, experiment_name, "benchmark"))

def get_investigation_dir(experiment_name):
    return mkDir_if_not(os.path.join(merged_datasets_dir, experiment_name, "investigation"))

def get_timing_dir(experiment_name):
    return mkDir_if_not(os.path.join(merged_datasets_dir, experiment_name, "timing"))

def path_results_for_dataset(dataset):
    return os.path.join(scores_for_best_hp_dir(dataset), "all.csv")

# def medium_processed_dir_per_class(dataset,class_of_interest): #TODO:remove?
#     return mkDir_if_not(os.path.join(scores_for_best_hp_dir(dataset), class_of_interest))

def path_results_for_best_hp_dataset_class_of_interest(dataset, class_of_interest):
    return os.path.join(scores_for_best_hp_dir(dataset), "results_{}.csv".format(class_of_interest))

def processed_results_dir(dataset):
    output_dir = mkDir_if_not(os.path.join("tests_real_data/processed_results", dataset + "_dataset"))
    return output_dir

def scores_aggr_dir(dataset):
    output_dir = processed_results_dir(dataset)
    medium_processed_results_dir = mkDir_if_not(os.path.join(output_dir, "scores_aggr"))
    return medium_processed_results_dir

def scores_for_best_hp_dir(dataset):
    output_dir = processed_results_dir(dataset)
    dir = mkDir_if_not(os.path.join(output_dir, "scores_for_best_hp"))
    return dir

def results_total_cv_grid_path(dataset, class_of_interest, classifier_name):
    dir = scores_aggr_dir(dataset)
    path = os.path.join(dir, class_of_interest + "_" + classifier_name + "_detailed.csv")
    return path

def get_dir_investigation(dataset):
    output_dir = get_output_dir(dataset)
    processed_results_dir = mkDir_if_not(os.path.join(output_dir, "processed_results"))
    dir_investigation = mkDir_if_not(os.path.join(processed_results_dir, "investigation_CV"))
    return dir_investigation

def results_cv_grid_path(dataset, class_of_interest, classifier):
    dir = scores_aggr_dir(dataset)
    path = os.path.join(dir, class_of_interest + "_" + classifier + ".csv")
    return path