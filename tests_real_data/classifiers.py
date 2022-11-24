#competing classifiers for real data

from utilities.optirank.classifiers.Optirank import Optirank
from utilities.optirank.classifiers.classifiers_helper import convert_penalties
from utilities.optirank.classifiers.default_args import default_optirank_args
from utilities.singleCellNetClassifier import singleCellNetWholePipeline
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import FunctionTransformer
from utilities.math_functions import rankdata_fun
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from utilities.ANOVA_subset_ranking import ANOVA_subset_ranking
import itertools

loglist = [0, 0.0001, 0.001, 0.01, 0.1]
percentage_sum_gamma_list = [0.01, 0.2, 0.4, 0.6, 0.8, 1.0, 0.005, 0.02, 0.05, 0.1]

solver = "saga"

classifiers_to_dump = ["optirank_log_regr", "logistic_regression_based_on_rankings"]

def get_untrained_classifier(classifier_name, n_samples):
    #n_samples is necessary to get the correspondance of the grid between optirank and scikit learn
    inverse_loglist_params = [convert_penalties(l1_w, l2_w, n_samples) for l1_w,l2_w in itertools.product(loglist, loglist)]
    if classifier_name == "optirank":
        classifier = make_pipeline(Optirank(**default_optirank_args))

        if classifier_name == "optirank":
            param_grid = list(ParameterGrid({"optirank__constraint_per_gamma_k": percentage_sum_gamma_list, "optirank__lambda_w_1": loglist, "optirank__lambda_w_2": loglist}))

    if classifier_name == "logistic_regression_based_on_rankings":
        transformer_rank = FunctionTransformer(rankdata_fun, kw_args = {"rank_type": "avg", "normalization": "d"})
        classifier = make_pipeline(transformer_rank, LogisticRegression(max_iter=10000, class_weight="balanced", tol=10**(-3), solver = solver))
        param_grid = [{"logisticregression__" + param_name:value for param_name, value in el.items()} for el in inverse_loglist_params]

    if classifier_name == "logistic_regression":
        classifier = make_pipeline(LogisticRegression(max_iter=10000, class_weight="balanced", tol=10**(-3), solver = solver))
        param_grid = [{"logisticregression__" + param_name: value for
                      param_name, value in el.items()} for el in inverse_loglist_params]

    if classifier_name == "random_forest":
        classifier = make_pipeline(RandomForestClassifier(class_weight="balanced"))
        param_grid = list(ParameterGrid({"randomforestclassifier__n_estimators":[300]}))

    if classifier_name == "single_cell_net":
        classifier = singleCellNetWholePipeline()
        param_grid = list(ParameterGrid({"nTrees":[1000]}))

    if classifier_name == "ANOVA_subset_ranking_lr":
        classifier = make_pipeline(
            ANOVA_subset_ranking(time_economy=True, perc_gamma=1, X_other=None, y_other=None),
            LogisticRegression(max_iter=10000, class_weight="balanced", tol=10 ** (-3), solver=solver))
        param_grid = []
        for perc in percentage_sum_gamma_list:
            for el in inverse_loglist_params:
                param_set = {"anova_subset_ranking__perc_gamma":perc}
                for param_name, value in el.items():
                    param_set["logisticregression__" + param_name] = value
                param_grid.append(param_set)

    return classifier, param_grid

optirank_classifier_names = ["optirank", "optirank_log_regr"] #for the moment only 1 (before no constraint sum_gamma)
logistic_regression_classifier_names = ["logistic_regression", "logistic_regression_based_on_rankings", "ANOVA_subset_ranking_lr"]