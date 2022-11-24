import numpy as np
import pandas as pd
from simulation_model.funs import generate_data
from joblib import dump, load
from utilities.small_functions import mkDir_if_not
from sklearn.model_selection import ParameterGrid
from utilities.small_functions import setLoglevel
import os

setLoglevel("info")

data_dir = mkDir_if_not(os.path.join("simulation_model", "data"))

default_params = {
    "d": 50,
    "n_perturbing": 40,
    "n_samples": 1000,
    "tau": 0.2,
    "sigma": 0.05}

path_experiment_table = os.path.join(data_dir, "experiments_table.pkl")
n_folds = int(np.loadtxt("simulation_model/n_folds.txt").item())  # number of times to repeat each data generation!


def path_experiment_data(experiment_name, index_param, i_fold):
    dir = mkDir_if_not(os.path.join(data_dir, experiment_name))
    path_experiment_data = os.path.join(dir, "data_param={}_{}.pkl".format(index_param, i_fold))
    return path_experiment_data


if __name__ == "__main__":

    experiments_table = pd.DataFrame(columns=["experiment_name", "n_params", "param_grid"])

    # experiment_4
    experiment_name = "robustness_to_noise"
    params_list = ParameterGrid({"sigma": [0.01, 0.05, 0.1, 0.2]})
    experiments_table = experiments_table.append({"experiment_name": experiment_name,
                                                  "n_params": len(params_list),
                                                  "param_grid": params_list}, ignore_index=True)

    # experiment 1:
    experiment_name = "different_tau"
    params_list = ParameterGrid({"tau": np.linspace(0, 0.5, 5).tolist()})
    experiments_table = experiments_table.append({"experiment_name": experiment_name,
                                                  "n_params": len(params_list),
                                                  "param_grid": params_list}, ignore_index=True)

    # experiment_2
    experiment_name = "different_n_perturbing_genes"
    params_list = ParameterGrid({"n_perturbing": [0, 10, 20, 30, 35, 40]})
    experiments_table = experiments_table.append({"experiment_name": experiment_name,
                                                  "n_params": len(params_list),
                                                  "param_grid": params_list}, ignore_index=True)

    # experiment_3
    experiment_name = "different_d"
    params_list = ParameterGrid([{"d": [50 * factor], "n_samples": [1000 * factor]} for factor in [1, 2, 3, 4]])
    experiments_table = experiments_table.append({"experiment_name": experiment_name,
                                                  "n_params": len(params_list),
                                                  "param_grid": params_list}, ignore_index=True)

    experiments_table.to_pickle(path_experiment_table)
    experiments_table[["experiment_name", "n_params"]].to_csv(os.path.join(data_dir, "experiments_table.csv"),
                                                              header=None, index=False)

    for i_row, row in experiments_table.iterrows():
        params_list = row["param_grid"]
        experiment_name = row["experiment_name"]
        for i, param in enumerate(params_list):
            for i_fold in range(n_folds):
                params = {**default_params, **param}
                data = generate_data(**params)
                dump((data, params), path_experiment_data(experiment_name, i, i_fold), compress=True)
