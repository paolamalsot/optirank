import pandas as pd
from simulation_model.create_params_grid import n_folds, path_experiment_table
from simulation_model.cross_validation import path_cv_results
from utilities.small_functions import mkDir_if_not
import os
import torch
from joblib import load
from simulation_model.cross_validation import path_classifier_object
from simulation_model.create_params_grid import path_experiment_data
import numpy as np


def get_results_experiment_name_path(experiment_name):
    results_dir = mkDir_if_not("simulation_model/results")
    return os.path.join(results_dir, "{}_results_cv.pkl".format(experiment_name))


def get_results_gamma_true(experiment_name, index_param, i_fold):
    (_, _, p, _), _ = load(path_experiment_data(experiment_name, index_param, i_fold))
    gamma = p.gamma
    return gamma


def calculate_overlap(gamma_true, gamma_sol):
    overlap = torch.dot(gamma_true.to(dtype=torch.double), gamma_sol.to(dtype=torch.double)) / (
                torch.norm(gamma_true) * torch.norm(gamma_sol))
    return overlap.item()


bilinear_name = "optirank"

if __name__ == "__main__":
    # create a pandas dataframe for each experiment
    experiment_table = pd.read_pickle(path_experiment_table)

    for i_row, row in experiment_table.iterrows():
        experiment_name = row["experiment_name"]
        n_params = row["n_params"]
        params_grid = list(row["param_grid"])
        results_all = []

        for i_fold in range(n_folds):
            for index_param, param_set in enumerate(params_grid):
                results_index_param = pd.read_csv(path_cv_results(experiment_name, index_param, i_fold))  # among other
                optirank_classifier = load(path_classifier_object(experiment_name, index_param, bilinear_name, i_fold))
                gamma_true = get_results_gamma_true(experiment_name, index_param, i_fold)
                gamma_sol = optirank_classifier.classifier.sol.gamma
                index_optirank_classifier = np.where(results_index_param.classifier_name == bilinear_name)
                overlaps = np.zeros(len(results_index_param.index))
                overlaps[index_optirank_classifier] = calculate_overlap(gamma_true, gamma_sol)
                results_index_param["overlap"] = overlaps.tolist()
                results_index_param["param_name"] = list(param_set.keys())[0]
                results_index_param["param_value"] = list(param_set.values())[0]
                results_index_param["index_param"] = index_param
                results_index_param["i_fold"] = i_fold
                results_all.append(results_index_param)

        results = pd.concat(results_all)
        results.to_pickle(get_results_experiment_name_path(experiment_name))
