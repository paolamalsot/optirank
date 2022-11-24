from utilities.optirank.classifiers.Optirank import optirank_transformer_pipe
from tests_real_data.classifiers import classifiers_to_dump, get_untrained_classifier
from tests_real_data.help_funs import calculate_metrics, averageable_metrics
from tests_real_data.preprocess_data_single_cell_net import path_data
from utilities.optirank.classifiers.classifiers_helper import create_param_from_log_regr_classifier
from tests_real_data.setup.get_mask_dataset_1_multiple_sources import get_mask_dataset_1_in_trainval_splits
from tests_real_data.loading_data import three_datasets_names_sub_merged
from utilities.small_functions import str_to_node, node_to_str, percentage_zero, warn_with_traceback, mkDir_if_not
from utilities.small_functions import setLoglevel
import argparse
from joblib import dump, load
import os
from tests_real_data.loading_data import get_output_dir
import logging
import time
import numpy as np
import traceback
import warnings

warnings.showwarning = warn_with_traceback
parser = argparse.ArgumentParser(description='CV')
parser.add_argument('dataset')
parser.add_argument('CV_split', type=str)
parser.add_argument("class_of_interest", type=str)
parser.add_argument("type", type=str)  # the type is lambda_P=0 or log_regr
parser.add_argument('-c', action='store_true')  # complete those that don't exist
args = parser.parse_args()
argv = vars(args)

setLoglevel("debug")

CV_index = int(argv["CV_split"])
dataset_name = argv["dataset"]
type = argv["type"]
complete_mode = argv["c"]
if type not in ["lambda_P=0", "log_regr"]:
    raise ValueError("Wrong type")
trained_classifier_name = "optirank"
classifier_name = trained_classifier_name + "_" + type

class_of_interest = argv["class_of_interest"]

filter_zero = "single_cell_net"
output_dir = get_output_dir(dataset_name)

results_dir = os.path.join(output_dir, "raw_results")
mkDir_if_not(results_dir)

Xy, test_run_index = load(path_data(dataset_name, CV_index, class_of_interest, "single_cell_net"))

n_samples = Xy["train"][1].shape[0]
classifier, param_list = get_untrained_classifier("optirank", n_samples)

indices_params = np.arange(len(param_list))

class_of_interest = node_to_str(class_of_interest)

if dataset_name in three_datasets_names_sub_merged:
    mask_dataset_1_train, mask_dataset_1_val = get_mask_dataset_1_in_trainval_splits(dataset_name, CV_index)

for index_param in indices_params:

    lightweight_trained_classifier_core_name = "__".join(
        [dataset_name, class_of_interest, filter_zero, trained_classifier_name, str(CV_index), str(index_param)])
    lightweight_trained_classifier_name = lightweight_trained_classifier_core_name + "__trained_classifier.pkl"
    lightweight_trained_results_name = lightweight_trained_classifier_core_name + "__metrics.pkl"
    name = "__".join(
        [dataset_name, class_of_interest, filter_zero, classifier_name, str(CV_index), str(index_param)])

    try:
        lightweight_trained_classifier = load(os.path.join(results_dir, lightweight_trained_classifier_name))
    except Exception as e:
        logging.error(traceback.format_exc())
        continue

    results_path = os.path.join(results_dir, name + "__metrics.pkl")
    results_classifier_name = name + "__trained_classifier.pkl"
    classifier_path = os.path.join(results_dir, results_classifier_name)

    if complete_mode:
        results_saved = os.path.exists(results_path)
        classifier_saved = os.path.exists(classifier_path) or not (classifier_name in classifiers_to_dump)
        if results_saved and classifier_saved:
            print("param {} already exists".format(index_param))
            continue
    if type == "log_regr":
        classifier = optirank_transformer_pipe(lightweight_trained_classifier, tol=10 ** (-3), max_iter=1000)
    else:
        raise NotImplementedError  # TODO if time but not necessary for deadline

    logging.info("{} param on {}".format(index_param + 1, len(indices_params)))
    start = time.time()

    # fitting classifier
    start_fit = time.time()

    try:
        classifier.fit(*Xy["train"])
    except Exception as e:
        logging.error(traceback.format_exc())
        continue

    stop_fit = time.time()

    print(stop_fit - start)
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
                                                 results["{}_metrics".format(split_name + "_1")][metric_name]], axis=0)
                           for metric_name in averageable_metrics}
            results["{}_metrics".format(split_name + "_avg")] = avg_metrics

            # exchange the average with the other metric!
            results["{}_metrics".format(split_name + "_not_balanced")], results["{}_metrics".format(split_name)] = \
                results["{}_metrics".format(split_name)], results["{}_metrics".format(split_name + "_avg")]
            del results["{}_metrics".format(split_name + "_avg")]

    results["test_run_index"] = test_run_index
    if "fitting_time" in results.keys():
        results["fitting_time"] = results["fitting_time"] + (stop_fit - start_fit)
    # suplementary results
    if type == "lambda_P=0":
        bilinear_classifier = classifier.named_steps["bilinearrankingclassifierfromsollambdap0"].classifier
        sol = classifier.named_steps["bilinearrankingclassifierfromsollambdap0"].classifier.sol
    elif type == "log_regr":
        bilinear_classifier = classifier.named_steps["subsetrankinglogregrpipe"]
        sol = create_param_from_log_regr_classifier(classifier.named_steps["subsetrankinglogregrpipe"])
    else:
        raise ValueError("Wrong type")

    results["converged"] = bilinear_classifier.converged
    results["per_gamma_0"] = percentage_zero(sol.gamma.numpy())
    results["per_w_0"] = percentage_zero(sol.w.numpy())

    # saving results
    dump(results, results_path)
    stop = time.time()
    logging.debug("Time for one iteration:{}".format(stop - start))

    if classifier_name in classifiers_to_dump:
        dump(classifier, classifier_path, compress=True)
