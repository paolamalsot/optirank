#files of constants that depend on experiment-name
import numpy as np

experiment_name = "simple_tasks"
validation_split_name="validation"
test_split_name="test"
optirank_name = "optirank" #bilinear_ranking_classifier_with_sum_gamma

split_names = ["train", "validation", "test"]

classifiers_list = ["optirank",
                    "optirank_log_regr",
                    "single_cell_net",
                    "logistic_regression_based_on_rankings",
                    "logistic_regression",
                    "random_forest"
                    ]

possibly_sparse_classifiers_other_than_optirank_derived=["logistic_regression"]
optirank_classifier_names = ["optirank", "optirank_log_regr"]

dataset_names=["BRCA", "TCGA", "Baron_Segerstolpe", "MWS_TM10x", "MWS_TMfacs", "TM10x_MWS", "TM10x_TMfacs", "TMfacs_MWS", "Baron_Murano"]


dataset_names_paper_dict = {"internal-validation ": ["BRCA", "TCGA"],
                            "external-validation": ["PCAWG"],
                            "biologically-different dataset": ["met-500"],
                            "cross-platform single cell RNA-seq":["Baron_Murano", "Baron_Segerstolpe", "MWS_TM10x", "MWS_TMfacs", "TM10x_MWS", "TM10x_TMfacs", "TMfacs_MWS"]}

dataset_names_paper_full_list = np.concatenate([np.array(vals) for vals in dataset_names_paper_dict.values()])