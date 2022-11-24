#sparsity comparisons and evaluations
#results in merged_dataset_results/experiment_name/investigations_CV/sparsity/...
#in "results_cases_occurences_[optirank_classifier].csv", investigate the frequency of test_score_A [<;>] test_score_B & genes_A [<;>] genes_B events
#in "percentage_of_best_cases_occurences_which_are_foreseable_[optirank_classifier].csv", investigate which percentage of "best_occurences" show also a better score in validation.
#"best occurences" is when optirank is the sparsest and has the best score
#"results_n_genes_[mean_fun]_[optirank_classifier].csv" shows the mean number of genes used, and ratio of genes
#when mean_fun is mean used to average the ratio. We use the geometric mean, which has the following property:

from tests_real_data.processing_results.funs.results_constants import *
from tests_real_data.processing_results.funs.results_funs import put_all_together
from tests_real_data.processing_results.funs.results_dir_organisation import get_investigation_dir
from utilities.small_functions import mkDir_if_not
from tests_real_data.processing_results.python_scripts.results_benchmark_01 import format_dataset_name_latex
import itertools
import operator as operator
import numpy as np
import os

def df_to_latex(df, outfile, label, float_format="%.2f"):
    # format row number into strings
    formatted_df = df.reindex(dataset_names_paper_full_list)
    formatted_df.index = [format_dataset_name_latex(el) for el in formatted_df.index]
    n_columns = len(df.columns)
    col_format = "|l|" + "c" * (n_columns) + "|"
    latex_str = formatted_df.to_latex(index=True, index_names = False, escape=False, label=label, position = "h!", column_format = col_format, float_format = float_format)
    with open(outfile, "w") as f:
        print(latex_str, file=f)

def format_n_genes_columns(string):
    dict_replacement = {"ANrank-lr": "Ar", "optirank": "orlr", "lr":"lr", "optirank_wolr": "or"}
    classifier_acronym = string.replace("n_genes_", "")
    return "$\s" + dict_replacement[classifier_acronym] + "$"

def format_frequency_column(string):
    dict_replacement = {"ANrank-lr": "Ar", "optirank": "orlr", "lr": "lr", "optirank_wolr": "or"}
    classifier_acronym = string.replace(" <= n_genes_others", "").replace("n_genes_", "")
    return "$100\,\hat{P}(\s" + "{}\!\le\!\sothers)$".format(dict_replacement[classifier_acronym])

def n_genes_ratio_fun(gdf, optirank_classifier, competitor_classifiers, test_split_name_):
    res_dict = {}
    all_classifiers = [optirank_classifier] + competitor_classifiers
    for classifier_ in all_classifiers:
        acronym_ = acronyms_dict[classifier_]
        res_dict["n_genes_{}".format(acronym_)] = gdf.loc[gdf.classifier_name == classifier_, "n_genes"].item()
        res_dict["test_score_{}".format(acronym_)] = gdf.loc[gdf.classifier_name == classifier_, test_split_name_ + "_value"].item()
        res_dict["val_score_{}".format(acronym_)] = gdf.loc[gdf.classifier_name == classifier_, validation_split_name + "_value"].item()

    for classifier_A, classifier_B in list(itertools.combinations(all_classifiers, 2)):
        acronym_A = acronyms_dict[classifier_A]
        acronym_B = acronyms_dict[classifier_B]
        n_genes_A = res_dict["n_genes_{}".format(acronym_A)]
        n_genes_B = res_dict["n_genes_{}".format(acronym_B)]
        res_dict["n_genes_{} - n_genes_{}".format(acronym_A, acronym_B)] = n_genes_A - n_genes_B

    for classifier_ in all_classifiers:
        acronym_ = acronyms_dict[classifier_]
        other_classifiers_ = set(all_classifiers).difference(classifier_)
        key = "n_genes_{} <= n_genes_others".format(acronym_)
        res_dict[key] = all([res_dict["n_genes_{}".format(acronym_)] <= res_dict["n_genes_{}".format(acronyms_dict[other_classifier_])] for other_classifier_ in other_classifiers_])

    res_dict["best_score_optirank_in_validation"] = all([res_dict["val_score_{}".format(acronyms_dict[optirank_classifier])] > res_dict["val_score_{}".format(acronyms_dict[compet])] for compet in competitor_classifiers])

    return pd.Series(res_dict)

metric_name = "balanced_accuracy"

if __name__ == "__main__":

    sparsity_investigation_directory = mkDir_if_not(os.path.join(get_investigation_dir(experiment_name), "sparsity"))

    for classifier_name in [optirank_name + "_log_regr", optirank_name]:
        acronym_optirank = acronyms_dict[classifier_name]
        for choosing_mode, suffix in choosing_modes_with_suffix_list:
            res = put_all_together([classifier_name] + possibly_sparse_classifiers_other_than_optirank_derived,
                                   metric_name, choosing_mode)
            out_df_list = []
            for dataset in dataset_names:

                if dataset == "TCGA":
                    dataset_name_split_tuple_list = [(test_split_name, dataset), ("PCAWG", "PCAWG"), ("met-500", "met-500")]
                else:
                    dataset_name_split_tuple_list = [(test_split_name, dataset)]

                for split_name, dataset_name_ in dataset_name_split_tuple_list:

                    test_score = "{}_value".format(split_name)
                    validation_score = "{}_value".format(validation_split_name)
                    res_selection = res.loc[(res["dataset_name"] == dataset),].copy()
                    cols = ["dataset_name", "CV_index", "classifier_name", "class_of_interest", "n_genes",
                            validation_score, test_score, "per_w_0", "per_gamma_0"]
                    res_selection = res_selection[cols].groupby(["dataset_name", "CV_index", "class_of_interest"]).apply(
                        n_genes_ratio_fun, classifier_name, possibly_sparse_classifiers_other_than_optirank_derived, split_name).reset_index()
                    operator_sign_list = [(operator.le, "<="), (operator.gt, ">")]
                    columns = []
                    for classifier_A, classifier_B in list(itertools.combinations([classifier_name] + possibly_sparse_classifiers_other_than_optirank_derived, 2)):
                        acronym_A = acronyms_dict[classifier_A]
                        acronym_B = acronyms_dict[classifier_B]

                        for (operator_score, sign), (operator_genes, sign_genes) in itertools.product(operator_sign_list, operator_sign_list):
                            colname = "score_{} {} score_{} & n_genes_{} {} n_genes_{}".format(acronym_A, sign, acronym_B, acronym_A, sign_genes, acronym_B)
                            res_selection[colname] = \
                                operator_score(res_selection["test_score_{}".format(acronym_A)], res_selection["test_score_{}".format(acronym_B)]) & operator_genes(res_selection["n_genes_{}".format(acronym_A)], res_selection["n_genes_{}".format(acronym_B)])
                            columns.append(colname)
                    res_selection["dataset_name"] = dataset_name_

                    out_df_list.append(res_selection)

            out_df = pd.concat(out_df_list)
            out = out_df[["dataset_name"] + columns].groupby("dataset_name").apply(lambda x: np.sum(x)/len(x))
            outdir = mkDir_if_not(os.path.join(sparsity_investigation_directory,metric_name,
                choosing_mode))
            out.to_csv(os.path.join(outdir, "results_cases_occurences_{}.csv".format(acronym_optirank)))
            df_to_latex(out, os.path.join(outdir, "results_cases_occurences_{}_latex.txt".format(acronym_optirank)), "", "%.2f")

            #in the best case, optirank is the best score and the sparsest!
            best_case_col = "score_{} > score_competitors & n_genes_{} <= n_genes_competitors".format(acronym_optirank, acronym_optirank)
            out_df[best_case_col] = np.all(np.stack([out_df["score_{} > score_{} & n_genes_{} <= n_genes_{}".format(acronym_optirank, acronyms_dict[compet], acronym_optirank, acronyms_dict[compet])] for compet in possibly_sparse_classifiers_other_than_optirank_derived], axis = 1), axis = 1)
            out_previsible = out_df.loc[out_df[best_case_col], ["dataset_name", "best_score_optirank_in_validation"]].groupby(["dataset_name"]).apply(lambda x: np.sum(x)/len(x))
            out_previsible.to_csv(os.path.join(outdir, "percentage_of_best_cases_occurences_which_are_foreseable_{}.csv".format(acronym_optirank)))
            df_to_latex(out_previsible, os.path.join(outdir, "percentage_of_best_cases_occurences_which_are_foreseable_{}_latex.txt".format(acronym_optirank)), "", "%.2f")

            #different n_genes means

            delta_genes_columns = [colname for colname in out_df.columns if ("n_genes" in colname and (" - " in colname))]
            n_genes_columns = [colname for colname in out_df.columns if ("n_genes" in colname and not(" - " in colname))]
            out = out_df[["dataset_name"] + delta_genes_columns].groupby(["dataset_name"]).mean()

            out[n_genes_columns] = out_df[
                ["dataset_name"]+ n_genes_columns].groupby(["dataset_name"]).mean()
            out = out.reindex(dataset_names_paper_full_list)
            out.to_csv(os.path.join(outdir, "results_n_genes_{}.csv".format(acronym_optirank)))

            #final latex table
            n_genes_columns = [colname for colname in out_df.columns if ("n_genes" in colname and not((" - " in colname) or (" <= " in colname) or (" > " in colname)))]
            frequency_sparsest_columns = [colname for colname in out_df.columns if (("n_genes" in colname) and ("others" in colname))]
            out = out_df[["dataset_name"] + n_genes_columns + frequency_sparsest_columns].groupby(["dataset_name"]).mean()
            #format table
            out[frequency_sparsest_columns] = out[frequency_sparsest_columns] * 100
            out = out.reindex(dataset_names_paper_full_list)
            # reformat column names
            new_cols = []
            for col in out.columns:
                if col in frequency_sparsest_columns:
                    new_cols.append(format_frequency_column(col))
                elif col in n_genes_columns:
                    new_cols.append(format_n_genes_columns(col))
                else:
                    new_cols.append(col)
            out.columns = new_cols
            df_to_latex(out, os.path.join(outdir, "results_n_genes_{}_latex.txt".format(acronym_optirank)), "", "%.0f")