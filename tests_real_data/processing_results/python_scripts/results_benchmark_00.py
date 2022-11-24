#averages test-scores on CV_splits (and on classes), aggregates the results of all datasets in the experiment
# -> tests_real_data/merged_dataset_results/[experiment_name]/benchmark/results_class_wise, results_dataset_wise

from tests_real_data.processing_results.funs.results_constants import dataset_names, choosing_modes, scoring_metrics, test_split_name, experiment_name
from tests_real_data.processing_results.funs.results_dir_organisation import path_results_for_dataset, get_benchmark_dir
import os
import itertools
import pandas as pd

#possibility 1: consider average performance of classifier in every one VS rest rest in every class for each dataset #Do TCGA, met-500 and PCAWG

if __name__ == "__main__":

    results_df_2 = pd.DataFrame(columns = ["dataset_name", "classifier_name", "class_name", "metric_name", "mean", "sem"])
    results_df_1 = pd.DataFrame(columns = ["dataset_name", "classifier_name", "metric_name", "mean", "sem"])


    for dataset in dataset_names:
        results = pd.read_csv(path_results_for_dataset(dataset))

        splits = [(test_split_name, dataset)]
        if dataset == "TCGA":
            splits = splits + [("PCAWG", "PCAWG"), ("met-500", "met-500")]


        for classifier, class_of_interest, metric, choosing_mode, (split, dataset_name) in itertools.product(pd.unique(results["classifier_name"]), pd.unique(results["class_of_interest"]), scoring_metrics, choosing_modes, splits):

            res = results.loc[(results.classifier_name == classifier) & (results.class_of_interest == class_of_interest) & (results.metric_name == metric) & (results.choosing_mode == choosing_mode), [split + "_value"]]
            average = res.mean().item()
            sem = res.sem().item()

            results_df_2 = results_df_2.append({"dataset_name":dataset_name, "classifier_name": classifier, "class_name":class_of_interest, "metric_name": metric, "choosing_mode": choosing_mode, "mean":average, "sem":sem}, ignore_index= True)

        for classifier, metric, choosing_mode, (split, dataset_name) in itertools.product(pd.unique(results["classifier_name"]), scoring_metrics, choosing_modes, splits):

            res = results.loc[(results.classifier_name == classifier) & (results.metric_name == metric) & (results.choosing_mode == choosing_mode), [split + "_value"]]
            average = res.mean().item()
            sem = res.sem().item()

            results_df_1 = results_df_1.append({"dataset_name":dataset_name, "classifier_name": classifier, "metric_name": metric, "choosing_mode": choosing_mode, "mean":average, "sem":sem}, ignore_index= True)

    results_df_2.to_csv(os.path.join(get_benchmark_dir(experiment_name), "results_class_wise.csv"))
    results_df_1.to_csv(os.path.join(get_benchmark_dir(experiment_name), "results_dataset_wise.csv"))