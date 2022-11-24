#timing different classifiers

from tests_real_data.processing_results.funs.results_constants import dataset_names, choosing_modes, scoring_metrics, experiment_name
from tests_real_data.processing_results.investigations.gene_selection_investigation.sparsity_comparison_with_log_regr import df_to_latex
from tests_real_data.processing_results.funs.results_dir_organisation import path_results_for_dataset, get_timing_dir
from tests_real_data.processing_results.funs.results_funs import add_classifier_acronym
import os
import itertools
import pandas as pd

#possibility 1: consider average performance of classifier in every one VS rest rest in every class for each dataset #Do TCGA, met-500 and PCAWG
metric = "balanced_accuracy"
choosing_mode = "one_standard_error_rule"

if __name__ == "__main__":

    results_df_2 = pd.DataFrame(columns = ["dataset_name", "classifier_name", "class_name", "mean", "sem", "fitting_time"])
    results_df_1 = pd.DataFrame(columns = ["dataset_name", "classifier_name", "mean", "sem", "fitting_time"])


    for dataset in dataset_names:

        results = pd.read_csv(path_results_for_dataset(dataset))


        for classifier, class_of_interest in itertools.product(pd.unique(results["classifier_name"]), pd.unique(results["class_of_interest"])):

            res = results.loc[(results.classifier_name == classifier) & (results.class_of_interest == class_of_interest) & (results.metric_name == metric) & (results.choosing_mode == choosing_mode), "fitting_time"]
            average = res.mean()
            sem = res.sem()

            results_df_2 = results_df_2.append({"dataset_name":dataset, "classifier_name": classifier, "class_name":class_of_interest, "mean":average, "sem":sem}, ignore_index= True)

        for classifier in pd.unique(results["classifier_name"]):

            res = results.loc[(results.classifier_name == classifier) & (results.metric_name == metric) & (results.choosing_mode == choosing_mode), "fitting_time"]
            average = res.mean()
            sem = res.sem()

            results_df_1 = results_df_1.append({"dataset_name":dataset, "classifier_name": classifier, "mean":average, "sem":sem}, ignore_index= True)

    results_df_2.to_csv(os.path.join(get_timing_dir(experiment_name), "results_class_wise.csv"))
    results_df_1.to_csv(os.path.join(get_timing_dir(experiment_name), "results_dataset_wise.csv"))

    results_df_1["timing_str"] = results_df_1["mean"].map('{:,.2f}'.format) + " ± " + results_df_1["sem"].map('{:,.2f}'.format)
    results_to_show = results_df_1.loc[results_df_1.classifier_name != "optirank_log_regr", :]
    results_to_show = add_classifier_acronym(results_to_show)
    out = results_to_show.pivot(index="dataset_name", columns="classifier_acronym", values="timing_str")
    markdown_table = out.to_markdown()

    f = open(os.path.join(get_timing_dir(experiment_name), "markdown_table"), "w")
    f.write(markdown_table)
    f.close()
    results_to_show = add_classifier_acronym(results_to_show, latex = True)
    out = results_to_show.pivot(index="dataset_name", columns="classifier_acronym", values="timing_str")
    df_to_latex(out, os.path.join(get_timing_dir(experiment_name), "latex_table"), label="", float_format="%.0f")