from simulation_model.create_params_grid import path_experiment_data
from utilities.small_functions import mkDir_if_not
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from simulation_model.cross_validation_helper_funs import *
from joblib import dump, load
import argparse
import pandas as pd
import os
import time

raw_results_dir = mkDir_if_not("simulation_model/raw_results")


def path_cv_results(experiment_name, index_param, i_fold):
    results_name = os.path.join(raw_results_dir,
                                "experiment_{}_index_param_{}_fold_{}_CV_results.csv".format(experiment_name,
                                                                                             index_param, i_fold))
    return results_name


def path_classifier_object(experiment_name, index_param, name_classifier, index_fold):
    results_name = os.path.join(raw_results_dir,
                                "experiment_{}_index_param_{}_classifier_{}_fold_{}.pkl".format(experiment_name,
                                                                                                index_param,
                                                                                                name_classifier,
                                                                                                index_fold))
    return results_name


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('index_param', type=int)
    parser.add_argument('i_fold', type=int)
    parser.add_argument('-c', action='store_true')
    args = parser.parse_args()
    argv = vars(args)
    experiment_name = argv['experiment_name']
    index_param = argv['index_param']
    i_fold = argv['i_fold']
    complete_mode = argv["c"]
    results_path = path_cv_results(experiment_name, index_param, i_fold)
    if not (os.path.exists(results_path) and complete_mode):
        path_data = path_experiment_data(experiment_name, index_param, i_fold)
        data, params = load(path_data)
        X, y, p, indices = data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        classifiers = get_list_all_classifiers(X_train, train_ratio=4 / 5, p=p)
        list_all_classifiers = classifiers  # default of sklearn

        results = pd.DataFrame(
            columns=["classifier_name", "best_params", "train balanced_accuracy", "test balanced_accuracy"])

        for classifier, name_classifier, parameter_grid in list_all_classifiers:

            print(name_classifier)

            cross_validator = GridSearchCV(classifier, parameter_grid, scoring="balanced_accuracy", verbose=3,
                                           refit=True)
            start = time.time()
            cross_validator.fit(X_train, y_train)
            stop = time.time()
            print("time:{}".format(stop - start))

            print("Best parameters:")
            print(cross_validator.best_params_)

            # prediction
            y_train_pred = cross_validator.predict(X_train)
            balanced_accuracy_train = balanced_accuracy_score(y_train, y_train_pred)
            print("train balanced_accuracy: {}".format(balanced_accuracy_train))

            y_test_pred = cross_validator.predict(X_test)
            balanced_accuracy_test = balanced_accuracy_score(y_test, y_test_pred)
            print("test balanced_accuracy: {}".format(balanced_accuracy_test))

            print("\n")

            # saving_classifier_lightweight
            if name_classifier in bilinear_ranking_classifier_names:
                classifier_to_dump = cross_validator.best_estimator_.to_lightweight()
            else:
                classifier_to_dump = cross_validator.best_estimator_
            dump(classifier_to_dump, path_classifier_object(experiment_name, index_param, name_classifier, i_fold),
                 compress=True)

            results_dict = {"classifier_name": name_classifier,
                            "best_params": str(cross_validator.best_params_),
                            "train_balanced_accuracy": balanced_accuracy_train,
                            "test_balanced_accuracy": balanced_accuracy_test}

            results = results.append(results_dict, ignore_index=True)

        results.to_csv(results_path)
    else:
        print("already exists")
