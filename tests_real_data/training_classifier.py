from utilities.small_functions import mkDir_if_not
from tests_real_data.loading_data import get_output_dir, three_datasets_names_sub_merged
from tests_real_data.classifiers import classifiers_to_dump, get_untrained_classifier, \
    logistic_regression_classifier_names, optirank_classifier_names
from tests_real_data.help_funs import calculate_metrics, averageable_metrics
from utilities.small_functions import node_to_str, percentage_zero
from utilities.optirank.classifiers.classifiers_helper import pipeline_to_lightweight, check_convergence_log_regr
from tests_real_data.preprocess_data_single_cell_net import path_data, get_untransformed_data
from tests_real_data.setup.get_mask_dataset_1_multiple_sources import get_mask_dataset_1_in_trainval_splits
from utilities.small_functions import setLoglevel
from utilities.small_functions import warn_with_traceback, warnings
from joblib import dump, load
import traceback
import os
import numpy as np
import logging
import time
import argparse

warnings.showwarning = warn_with_traceback
setLoglevel("debug")


def get_setup_three_datasets(dataset_name, CV_index, class_of_interest):
    Xy, test_run_index = load(path_data(dataset_name, CV_index, class_of_interest, "single_cell_net"))
    n_samples = Xy["train"][1].shape[0]
    X_other, y_other = Xy["train_1"]
    classifier, param_list = get_untrained_classifier("ANOVA_subset_ranking_lr", n_samples)
    classifier.named_steps["anova_subset_ranking"].X_other = X_other
    classifier.named_steps["anova_subset_ranking"].y_other = y_other

    return Xy, test_run_index, classifier, param_list


def get_setup_three_datasets_01_sub_merged(dataset_name, CV_index, class_of_interest):
    Xy, test_run_index = load(path_data(dataset_name, CV_index, class_of_interest, "single_cell_net"))
    n_samples = Xy["train"][1].shape[0]
    mask_dataset_1_train, mask_dataset_1_val = get_mask_dataset_1_in_trainval_splits(dataset_name, CV_index)
    classifier, param_list = get_untrained_classifier("ANOVA_subset_ranking_lr", n_samples)
    classifier.named_steps["anova_subset_ranking"].mask_dataset_other = mask_dataset_1_train

    return Xy, test_run_index, classifier, param_list


def run(dataset_name, class_of_interest, classifier_name, index_param, CV_index, full_grid, complete_mode):
    filter_zero = "single_cell_net"
    output_dir = get_output_dir(dataset_name)
    results_dir = os.path.join(output_dir, "raw_results")
    mkDir_if_not(results_dir)

    if classifier_name != "ANOVA_subset_ranking_lr":

        if classifier_name != "single_cell_net":
            Xy, test_run_index = load(path_data(dataset_name, CV_index, class_of_interest, "single_cell_net"))
        else:
            Xy, test_run_index = get_untransformed_data(dataset_name, CV_index, class_of_interest)

        n_samples = Xy["train"][1].shape[0]

        classifier, param_list = get_untrained_classifier(classifier_name, n_samples)
    else:
        if dataset_name not in three_datasets_names_sub_merged:
            Xy, test_run_index, classifier, param_list = get_setup_three_datasets(dataset_name, CV_index,
                                                                                  class_of_interest)
        else:
            Xy, test_run_index, classifier, param_list = get_setup_three_datasets_01_sub_merged(dataset_name, CV_index,
                                                                                                class_of_interest)

    if dataset_name in three_datasets_names_sub_merged:
        mask_dataset_1_train, mask_dataset_1_val = get_mask_dataset_1_in_trainval_splits(dataset_name, CV_index)

    if not (full_grid):
        indices_params = [index_param]
    else:
        indices_params = np.arange(len(param_list))

    class_of_interest = node_to_str(class_of_interest)

    for index_param in indices_params:
        logging.info("{} param on {}".format(index_param + 1, len(indices_params)))
        start = time.time()

        name = "__".join(
            [dataset_name, class_of_interest, filter_zero, classifier_name, str(CV_index), str(index_param)])

        results_name = name + "__metrics.pkl"
        results_classifier_name = name + "__trained_classifier.pkl"
        results_path = os.path.join(results_dir, results_name)
        classifier_path = os.path.join(results_dir, results_classifier_name)

        if complete_mode:
            classifier_saved = not (classifier_name in optirank_classifier_names) or (
                os.path.exists(classifier_path))
            results_saved = os.path.exists(results_path)
            if classifier_saved and results_saved:
                print("param {} already exists".format(index_param))
                continue

        classifier.set_params(**param_list[index_param])

        # training classifier
        start_fit = time.time()
        try:
            classifier.fit(*Xy["train"])
        except Exception as e:
            logging.error(traceback.format_exc())
            continue
        stop_fit = time.time()
        fitting_time = stop_fit - start_fit

        # testing classifier
        results = {}
        for split_name, (X, y) in Xy.items():
            y_pred = classifier.predict(X)
            y_probas = classifier.predict_proba(X)[:, 1]
            metrics = calculate_metrics(y, y_probas, y_pred)
            results["{}_metrics".format(split_name)] = metrics
            results["y_{}_probas".format(split_name)] = y_probas
            results["y_{}".format(split_name)] = y

            if split_name in ["train", "validation"] and dataset_name in three_datasets_names_sub_merged:

                if split_name == "train":
                    mask_dataset_1 = mask_dataset_1_train
                else:
                    mask_dataset_1 = mask_dataset_1_val

                for i_dataset, mask_dataset in zip([0, 1], [np.logical_not(mask_dataset_1), mask_dataset_1]):
                    split_name_sup = split_name + "_" + str(i_dataset)
                    y_pred_sup = y_pred[mask_dataset]
                    y_probas_sup = y_probas[mask_dataset]
                    y_sup = y[mask_dataset]
                    metrics_sup = calculate_metrics(y_sup, y_probas_sup, y_pred_sup)
                    results["{}_metrics".format(split_name_sup)] = metrics_sup

                # calculate the average metrics
                avg_metrics = {metric_name: np.mean([results["{}_metrics".format(split_name + "_0")][metric_name],
                                                     results["{}_metrics".format(split_name + "_1")][metric_name]],
                                                    axis=0) for metric_name in averageable_metrics}
                results["{}_metrics".format(split_name + "_avg")] = avg_metrics

                # exchange the average with the other metric!
                results["{}_metrics".format(split_name + "_not_balanced")], results["{}_metrics".format(split_name)] = \
                results["{}_metrics".format(split_name)], results["{}_metrics".format(split_name + "_avg")]
                del results["{}_metrics".format(split_name + "_avg")]

        results["test_run_index"] = test_run_index
        results["fitting_time"] = fitting_time

        # append sparsity when necessary:
        if classifier_name in optirank_classifier_names:
            results["converged"] = classifier.named_steps["optirank"].converged
            results["per_gamma_0"] = percentage_zero(
                classifier.named_steps["optirank"].classifier.sol.gamma.numpy())
            results["per_w_0"] = percentage_zero(
                classifier.named_steps["optirank"].classifier.sol.w.numpy())
            new_light_pipe = pipeline_to_lightweight(classifier, copy=True)
            dump(new_light_pipe, classifier_path, compress=True)

        if classifier_name == "ANOVA_subset_ranking_lr":
            new_light_pipe = pipeline_to_lightweight(classifier, copy=True)
            dump(new_light_pipe, classifier_path, compress=True)
            results["per_gamma_0"] = percentage_zero(classifier.named_steps["anova_subset_ranking"].gamma_.numpy())

        if classifier_name in logistic_regression_classifier_names:
            results["per_w_0"] = percentage_zero(classifier.named_steps["logisticregression"].coef_.flatten())
            results["converged"] = check_convergence_log_regr(classifier.named_steps["logisticregression"])

        if classifier_name in classifiers_to_dump:
            dump(classifier, classifier_path, compress=True)

        # saving results
        dump(results, results_path, compress=3)
        stop = time.time()
        logging.debug("Time for one iteration:{}".format(stop - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CV')
    parser.add_argument('dataset')
    parser.add_argument('classifier_name')
    parser.add_argument('CV_split', type=str)
    parser.add_argument("index_param", type=str)
    parser.add_argument("class_of_interest", type=str)
    parser.add_argument('-f', action='store_true')  # full grid
    parser.add_argument('-c', action='store_true')  # complete those that don't exist
    args = parser.parse_args()
    argv = vars(args)

    CV_index = int(argv["CV_split"])
    dataset_name = argv["dataset"]
    classifier_name = argv["classifier_name"]
    index_param = int(argv["index_param"])
    class_of_interest = argv["class_of_interest"]
    full_grid = argv["f"]
    complete_mode = argv["c"]

    run(dataset_name, class_of_interest, classifier_name, index_param, CV_index, full_grid, complete_mode)
