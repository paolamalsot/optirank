#To run after results_autorank.py and pairwise_comparison_matrices.py
#Creates a table with mean and sem across all cv-splits and classes, shows the rank of the average performance.
#In bold, there will be the best classifier, along all those that are not significantly different from it, according
#to the pairwise differences.
#Also outputs one .csv file, mentioning the number of classes where each classifier was within the bests.
#tests_real_data/merged_dataset_results/[experiment_name]/benchmark/balanced_accuracy/one_standard_error_rule/...

import os
import pandas as pd
from tests_real_data.processing_results.funs.results_constants import dataset_names_paper_full_list
from tests_real_data.processing_results.python_scripts.pairwise_comparison_matrices import *

def format_according_to_within_best(x):
    if x.within_best:
        out = "$\mathbf{" + '{:.2f}'.format(x["mean"]*100) + " \pm " + '{:.2f}'.format(x["sem"]*100)\
              + "}$"+" \textbf{(" + '{:,.0f}'.format(x["rank"]) + ")}"
    else:
        out = '${:.2f}'.format(x["mean"]*100) + " \pm " + '{:.2f}$'.format(
            x["sem"]*100) \
              + " (" + '{:,.0f}'.format(x["rank"]) + ")"
    return out

def format_according_to_within_best_markdown(x):
    if x.within_best:
        out = "**" + '{:.2f}'.format(x["mean"]*100) + " ± " + '{:.2f}'.format(x["sem"]*100)\
              +" (" + '{:.0f}'.format(x["rank"]) + ")**"
    else:
        out = '{:.2f}'.format(x["mean"]*100) + " ± " + '{:.2f}'.format(
            x["sem"]*100) \
              + " (" + '{:,.0f}'.format(x["rank"]) + ")"
    return out

def format_dataset_name_latex(string):
    return "\task{" + string.replace("_01_sub_merged", "").replace("_", "-") + "}"

if __name__ == "__main__":
    metric = "balanced_accuracy"
    choosing_mode = "one_standard_error_rule"
    metric_specific_benchmark_dir = mkDir_if_not(os.path.join(get_benchmark_dir(experiment_name), metric, choosing_mode))
    for comparison_test in ["Student", "Wilcoxon"]:
        for class_wise in [True, False]:
            if class_wise:
                df = pd.read_csv(os.path.join(get_benchmark_dir(experiment_name), "results_class_wise.csv"))
                row_indices = ["dataset_name", "class_name"]
                outfile_md = os.path.join(metric_specific_benchmark_dir,
                                       "results_class_wise{}_with_significance_markdown.md".format(
                                           comparison_test_suffix[comparison_test]))
                outfile_latex = os.path.join(metric_specific_benchmark_dir, "results_class_wise{}_with_significance.txt".format(
                    comparison_test_suffix[comparison_test]))
                outfile_n_positive_classes = os.path.join(metric_specific_benchmark_dir, "summary_n_winning_classes_per_dataset{}.csv".format(
                    comparison_test_suffix[comparison_test]))

            else:
                df = pd.read_csv(os.path.join(get_benchmark_dir(experiment_name), "results_dataset_wise.csv"))
                row_indices = ["dataset_name"]
                outfile_md = os.path.join(metric_specific_benchmark_dir,
                                       "results_dataset_wise{}_with_significance_markdown.md".format(
                                           comparison_test_suffix[comparison_test]))
                outfile_latex = os.path.join(metric_specific_benchmark_dir, "results_dataset_wise{}_with_significance.txt".format(
                    comparison_test_suffix[comparison_test]))
            df = add_classifier_acronym(df, latex=False)
            df_subset = df.loc[
                    (df.metric_name == metric) & (df.choosing_mode == choosing_mode) & np.isin(df.classifier_name,
                                                                                               classifiers_figures),]

            ascending = meta_metrics.loc[meta_metrics.name == metric, "best_orientation"].item() == "min"
            df_out = df_subset.copy()

            df_out["rank"] = df_out[row_indices + ["mean"]].groupby(row_indices).rank(ascending=ascending).copy()

            def within_best(x):
                if class_wise:
                    index_best_value = df_out.loc[(df_out.dataset_name == x.dataset_name) & (df_out.class_name == x.class_name), "rank"].idxmin()
                else:
                    index_best_value = df_out.loc[df_out.dataset_name == x.dataset_name, "rank"].idxmin()
                best_classifier = df_out.loc[index_best_value, "classifier_acronym"]
                this_classifier = x["classifier_acronym"]
                if this_classifier == best_classifier:
                    return True
                else:
                    dataset = x.dataset_name
                    if class_wise:
                        class_of_interest = x.class_name
                    else:
                        class_of_interest = None
                    df_pairwise = pd.read_csv(get_matrix_path(dataset, comparison_test=comparison_test, class_of_interest = class_of_interest), index_col=0, dtype="str")
                    return is_1_not_significantly_different_than_2(df_pairwise, this_classifier, best_classifier)

            df_out["within_best"] = df_out.apply(within_best, axis = 1)
            df_out["string_col"] = df_out.apply(format_according_to_within_best, axis = 1)
            # for index in row_indices + ["classifier_acronym"]:
            #     df_out[index] = df_out[index].str.replace("_", "-", regex=False)
            # row_indices_ = [index.replace("_", "-") for index in row_indices]
            # df_out.columns = df_out.columns.str.replace("_", "-")
            # df_out.dataset_name = [dataset_name.replace("_", "-") for dataset_name in df_out["dataset-name"]]
            df_out = add_classifier_acronym(df_out, latex=True)
            formatted_df = df_out.pivot(index=row_indices, columns="classifier_acronym", values="string_col")
            #dataset_names_paper_full_list_changed = [dataset_name.replace("_", "-") for dataset_name in dataset_names_paper_full_list]
            formatted_df = formatted_df.reindex(dataset_names_paper_full_list)
            formatted_df.index = [format_dataset_name_latex(el) for el in formatted_df.index]
            n_classifiers = len(pd.unique(df_out["classifier_name"]))
            col_format = "|l|" + "c" * (n_classifiers) + "|"
            latex_str = formatted_df.to_latex(index=True, index_names = False, escape=False, label="tab:summary-results-{}".format(experiment_name), position = "h!", column_format= col_format)

            with open(outfile_latex, "w") as f:
                print(latex_str, file=f)

            #markdown table
            df_out = df_subset.copy()
            df_out["rank"] = df_out[row_indices + ["mean"]].groupby(row_indices).rank(ascending=ascending).copy()
            df_out["within_best"] = df_out.apply(within_best, axis=1)
            df_out["string_col_md"] = df_out.apply(format_according_to_within_best_markdown, axis=1)
            formatted_df = df_out.pivot(index=row_indices, columns="classifier_acronym", values="string_col_md")
            formatted_df = formatted_df.reindex(dataset_names_paper_full_list)
            markdown_table = formatted_df.to_markdown()
            f = open(outfile_md, "w")
            f.write(markdown_table)
            f.close()

            #table that shows per dataset per classifier, in how many class it is within the best!
            if class_wise:
                df_out = df_subset.copy()
                df_out["rank"] = df_out[row_indices + ["mean"]].groupby(row_indices).rank(ascending=ascending).copy()
                df_out["within_best"] = df_out.apply(within_best, axis=1)
                df_out = df_out[["dataset_name", "classifier_acronym", "within_best"]].groupby(["dataset_name", "classifier_acronym"])["within_best"].apply(np.sum).reset_index()
                formatted_df = df_out.pivot(index=["dataset_name"], columns="classifier_acronym", values="within_best")

                #saving as csv
                formatted_df.to_csv(outfile_n_positive_classes)