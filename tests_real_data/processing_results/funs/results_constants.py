import pandas as pd
import numpy as np
from tests_real_data.processing_results.funs.config_files.config import *

classifiers_df = pd.read_csv("tests_real_data/meta_classifiers.csv", header = 0, names = ["classifier_name", "n_params"], dtype = {"classifier_name":str, "n_params":int})
meta_dataset = pd.read_csv("tests_real_data/meta_datasets.csv", header = 0,
                           names=["dataset_name", "classes", "n_splits", "n_test_splits", "n_samples_train"], index_col=None)
meta_metrics = pd.read_csv("tests_real_data/meta_metrics.csv")
filter_zero_name = "single_cell_net"
choosing_modes_with_suffix_list = [("average_scoring", "_avg_scoring"), ("one_standard_error_rule", "_1-std")]
choosing_modes, suffixes = zip(*choosing_modes_with_suffix_list)
d = 1000

#can be a subset in case we do not wish all classifiers to appear on results and tables
classifiers_figures = classifiers_list.copy()
classifiers_figures.remove("optirank")
optirank_derived_classifiers_list = ["optirank_log_regr"] #lambda_P=0

all_optirank_classifiers = optirank_classifier_names + \
                           optirank_derived_classifiers_list

classifiers_with_supp_metrics = optirank_derived_classifiers_list + \
                               optirank_classifier_names + ["ANOVA_subset_ranking_lr"]

acronyms_dict = {"optirank_log_regr": "optirank",
                 "optirank": "optirank_wolr",
                 'optirank_lambda_P=0': 'optirank_lambda_P=0',
                 "single_cell_net": "SCN",
                 "logistic_regression_based_on_rankings": "rank-lr",
                 "logistic_regression": "lr",
                 "random_forest": "rf",
                 "ANOVA_subset_ranking_lr": "ANrank-lr"}

acronyms_dict_latex = {"optirank_log_regr": "\optirank",
                       "optirank": "\optirank",
                       "single_cell_net": "\SCN",
                       "logistic_regression_based_on_rankings": r"\ranklr",
                       "logistic_regression": "\lr",
                       "random_forest": r"\rf",
                       "ANOVA_subset_ranking_lr": "\ANranklr"}

scoring_metrics = ["accuracy",
                   "balanced_accuracy",
                   "cross_entropy",
                   "balanced_cross_entropy",
                   "roc_auc_score",
                   "auprc"]

def add_classifier_acronym(df, latex = False):
    """df is a dataframe with column classifier_name. Add column classifier_acronym"""
    if latex:
        fun = lambda x: acronyms_dict_latex[x]
    else:
        fun = lambda x: acronyms_dict[x]
    df["classifier_acronym"] = df["classifier_name"].apply(fun)
    return df

def get_classes(dataset):
    classes_of_interest = np.array(meta_dataset.loc[meta_dataset["dataset_name"] == dataset, "classes"].item().split(
        " "))
    return classes_of_interest

def get_split_names(dataset_name):
    split_names_ = split_names
    if dataset_name == "TCGA":
        split_names_ += ["PCAWG", "met-500"]
    return split_names_

def get_n_test_splits(dataset_name):
    return meta_dataset.loc[meta_dataset.dataset_name == dataset_name, "n_test_splits"].item()

def get_n_splits(dataset_name):
    return meta_dataset.loc[meta_dataset.dataset_name == dataset_name, "n_splits"].item()

def get_n_params_for_classifier(classifier_name):
    if classifier_name in optirank_derived_classifiers_list:
        classifier_name_for_params = "optirank"
    else:
        classifier_name_for_params = classifier_name
    return classifiers_df.loc[classifiers_df["classifier_name"] == classifier_name_for_params, "n_params"].item()
