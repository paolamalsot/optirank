#files of constants that depend on experiment-name

experiment_name = "multiple_sources"
validation_split_name="validation"
test_split_name="test"
optirank_name = "optirank"

split_names = ["train", "validation", "test", "validation_not_balanced"]

classifiers_list = ["optirank",
                    "optirank_log_regr",
                    "ANOVA_subset_ranking_lr",
                    "single_cell_net",
                    "logistic_regression_based_on_rankings",
                    "logistic_regression",
                    "random_forest"
                    ]

possibly_sparse_classifiers_other_than_optirank_derived=["ANOVA_subset_ranking_lr", "logistic_regression"]
optirank_classifier_names = ["optirank", "optirank_log_regr"]

dataset_names = ["TCGA_PCAWG_met500_01_sub_merged",
                 "Baron_Segerstolpe_Murano_01_sub_merged",
                 "MWS_TMfacs_TM10x_01_sub_merged"]

dataset_names_paper_full_list = dataset_names