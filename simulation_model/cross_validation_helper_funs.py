from utilities.optirank.classifiers.default_args import default_optirank_args
from utilities.optirank.classifiers.Optirank import Optirank
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from utilities.math_functions import rankdata_fun
from sklearn.pipeline import make_pipeline
from utilities.optirank.classifiers.classifiers_helper import convert_penalties


bilinear_ranking_classifier_names = ["optirank"]


def get_list_all_classifiers(X_train, train_ratio, p):
    # cross_validation with sci-kit learn (sum_gamma) and the default with abs_dloss...
    loglist = [0, 0.0001, 0.001, 0.01, 0.1]
    lambda_w_2_list = {"lambda_w_2": loglist}
    n_samples = X_train.shape[0] * train_ratio
    sklearn_logistic_regression_params = [convert_penalties(0, l2_w, n_samples) for l2_w in loglist]
    constraint_per_gamma_k_list = [0.2, 0.4, 0.6, 0.8, 1]

    # optirank_classifier
    optirank_params = {**default_optirank_args, "max_relaxation_iter": 1000}
    optirank = Optirank(**optirank_params)
    parameter_dict_optirank = {"lambda_gamma_1": [0],
                               "lambda_gamma_2": [0],
                               "constraint_per_gamma_k": constraint_per_gamma_k_list}
    parameter_grid_optirank = {**parameter_dict_optirank, **lambda_w_2_list}

    # logistic_regression
    classifier_logistic_regression = LogisticRegression(max_iter=10000, tol=10 ** (-5), class_weight="balanced")
    parameter_grid_logistic_regression = [{param_name: [value] for param_name, value in el.items()} for el in
                                          sklearn_logistic_regression_params]

    # logistic regression on ranks
    transformer_rank = FunctionTransformer(rankdata_fun, kw_args={"rank_type": "avg", "normalization": "d"})
    log_regr = LogisticRegression(max_iter=10000, class_weight="balanced", tol=10 ** (-5))
    classifier_logistic_regression_on_ranks = make_pipeline(transformer_rank, log_regr)
    param_grid_logistic_regression_on_ranks = [
        {"logisticregression__" + param_name: [value] for param_name, value in el.items()} for el in
        sklearn_logistic_regression_params]

    list_classifiers = [
        (classifier_logistic_regression, "logistic_regression", parameter_grid_logistic_regression),
        (classifier_logistic_regression_on_ranks, "logistic_regression_on_ranks",
         param_grid_logistic_regression_on_ranks),
        (optirank, "optirank", parameter_grid_optirank),
    ]

    return list_classifiers
