# check what is the distance between lambda_P_0 solution and final, log_reg and final,
# check how many lambda_P iterations were needed (1 or more).
# check how this differs in the best hp or not!
from tests_real_data.processing_results.funs.results_constants import *
from tests_real_data.processing_results.funs.results_dir_organisation import *
from utilities.optirank.src.loss.params import Params
from utilities.optirank.classifiers.classifiers_helper  import create_param_from_log_regr_classifier
from sklearn.metrics.pairwise import cosine_similarity
import torch
from scipy.stats import hypergeom
import numpy as np
from joblib import load
from utilities.ANOVA_subset_ranking_helper import get_sol_ANOVA


#agglomerate_cv_taking account supplementary properties from the classifier .pkl file (and the log-regr)
def distances_between_sols(ref_sol:Params, other_sol:Params):
    """
    Returns the cosine distance between the two wTgamma, returns abs(db) (faute de mieux...)
    """
    flattened_products = [(sol.w.reshape(-1, 1) @ sol.gamma.reshape(1, -1)).flatten().numpy() for sol in [ref_sol, other_sol]]
    cos_sim = np.clip(cosine_similarity(X = np.vstack(flattened_products))[0, 1], -1, 1)
    cos_dist = 1-cos_sim
    abs_db = torch.abs(ref_sol.b - other_sol.b).item()

    return cos_dist, abs_db

def get_n_genes(sol):
    gamma_nonzero = set(np.nonzero(sol.gamma).numpy().flatten().tolist())
    w_nonzero = set(np.nonzero(sol.w).numpy().flatten().tolist())
    return len(set.union(gamma_nonzero, w_nonzero))

def get_p_val_intersection_gene_sets(sol, type):
    if type == "w_in_gamma":
        target_set = set(np.nonzero(sol.gamma).numpy().flatten().tolist())
        source_set = set(np.nonzero(sol.w).numpy().flatten().tolist())
    elif type == "gamma_in_w":
        source_set = set(np.nonzero(sol.gamma).numpy().flatten().tolist())
        target_set = set(np.nonzero(sol.w).numpy().flatten().tolist())
    else:
        raise ValueError("Wrong type")

    M = len(sol.gamma)
    n = len(target_set)
    N = len(source_set)
    k = len(set.intersection(source_set, target_set))
    rv = hypergeom(M, n, N)
    p_val = rv.cdf(k)
    return p_val

def retrieving_supp_metrics_bilinear_ranking_classifier(dataset_name, filter_zero_name, class_of_interest, classifier_name):

    if classifier_name not in classifiers_with_supp_metrics:
        raise ValueError("classifier name {} not in classifiers_with_supp_metrics list: {}".format(classifier_name, classifiers_with_supp_metrics))

    output_dir = get_output_dir(dataset_name)
    n_params_for_classifier = get_n_params_for_classifier(classifier_name)
    n_splits = meta_dataset.loc[meta_dataset.dataset_name == dataset_name, "n_splits"].item()

    output_df = []
    # and others for the metrics

    for index_param in range(n_params_for_classifier):
        for CV_index in np.arange(0, n_splits):
            results_dir = os.path.join(output_dir, "raw_results")
            name = "__".join([dataset_name, class_of_interest, filter_zero_name, classifier_name, str(CV_index),
                              str(index_param)])
            if classifier_name == "optirank":
                classifier_filename = os.path.join(results_dir, name + "__trained_classifier.pkl")
                classifier = load(classifier_filename)
                sol = classifier.named_steps["optirank"].classifier.sol

                # getting additional metrics
                n_genes = get_n_genes(sol)
                p_val_w_in_gamma = get_p_val_intersection_gene_sets(sol, "w_in_gamma")
                p_val_gamma_in_w = get_p_val_intersection_gene_sets(sol, "gamma_in_w")
                n_lambda_P_iterations = classifier.named_steps["optirank"].lambda_P_iter

                supp_dict = {
                        "n_lambda_P_iterations": n_lambda_P_iterations,
                        "n_genes": n_genes,
                        "p_val_w_in_gamma": p_val_w_in_gamma,
                        "p_val_gamma_in_w": p_val_gamma_in_w}

                #comparison with log-regr classifier
                if "optirank_log_regr" in classifiers_list:
                    log_regr_classifier_filename = classifier_filename.replace("optirank", "optirank_log_regr")
                    log_regr_classifier = load(log_regr_classifier_filename)
                    sol_log_regr = create_param_from_log_regr_classifier(log_regr_classifier.named_steps["subsetrankinglogregrpipe"])
                    dist_log_regr_coef, dist_log_regr_intercept = distances_between_sols(sol, sol_log_regr)
                    supp_dict["dist_log_regr_coef"] = dist_log_regr_coef
                    supp_dict["dist_log_regr_intercept"] = dist_log_regr_intercept

                #comparison with lambda_P=0 classifier
                sol_lambda_P_0 = classifier.named_steps["optirank"].res_lambda_P_0_
                dist_lambda_P_0_coef,  dist_lambda_P_0_intercept = distances_between_sols(sol, sol_lambda_P_0)
                supp_dict["dist_lambda_P_0_coef"] = dist_lambda_P_0_coef
                supp_dict["dist_lambda_P_0_intercept"] = dist_lambda_P_0_intercept

            elif classifier_name == "optirank_log_regr":
                classifier_filename = os.path.join(results_dir, name + "__trained_classifier.pkl")
                classifier = load(classifier_filename)
                sol = create_param_from_log_regr_classifier(classifier.named_steps["subsetrankinglogregrpipe"])

                no_log_regr_classifier_filename = classifier_filename.replace("optirank_log_regr", "optirank")
                no_log_regr_classifier = load(no_log_regr_classifier_filename)
                no_log_regr_sol = no_log_regr_classifier.named_steps["optirank"].classifier.sol
                sol_lambda_P_0 = no_log_regr_classifier.named_steps["optirank"].res_lambda_P_0_

                # getting additional metrics
                dist_lambda_P_0_coef, dist_lambda_P_0_intercept = distances_between_sols(sol, sol_lambda_P_0)
                dist_log_regr_coef, dist_log_regr_intercept = distances_between_sols(sol, no_log_regr_sol)
                n_genes = get_n_genes(sol)
                p_val_w_in_gamma = get_p_val_intersection_gene_sets(sol, "w_in_gamma")
                p_val_gamma_in_w = get_p_val_intersection_gene_sets(sol, "gamma_in_w")
                n_lambda_P_iterations = no_log_regr_classifier.named_steps["optirank"].lambda_P_iter

                supp_dict = {"dist_lambda_P_0_coef": dist_lambda_P_0_coef,
                             "dist_lambda_P_0_intercept": dist_lambda_P_0_intercept,
                             "dist_log_regr_coef": dist_log_regr_coef,
                             "dist_log_regr_intercept": dist_log_regr_intercept,
                             "n_lambda_P_iterations": n_lambda_P_iterations,
                             "n_genes": n_genes,
                             "p_val_w_in_gamma": p_val_w_in_gamma,
                             "p_val_gamma_in_w": p_val_gamma_in_w}

            elif classifier_name == "optirank_lambda_P=0":
                full_classifier_name = name.replace("_lambda_P=0", "")
                classifier_filename = os.path.join(results_dir, full_classifier_name + "__trained_classifier.pkl")
                classifier = load(classifier_filename)
                sol_lambda_P_0 = classifier.named_steps["optirank"].res_lambda_P_0_
                n_lambda_P_iterations = 1
                p_val_w_in_gamma = get_p_val_intersection_gene_sets(sol_lambda_P_0, "w_in_gamma")
                p_val_gamma_in_w = get_p_val_intersection_gene_sets(sol_lambda_P_0, "gamma_in_w")
                n_genes = get_n_genes(sol_lambda_P_0)
                supp_dict = {"n_genes": n_genes,
                             "n_lambda_P_iterations": n_lambda_P_iterations,
                             "p_val_w_in_gamma": p_val_w_in_gamma,
                             "p_val_gamma_in_w": p_val_gamma_in_w}

            elif classifier_name == "ANOVA_subset_ranking_lr":
                classifier_filename = os.path.join(results_dir, name + "__trained_classifier.pkl")
                classifier = load(classifier_filename)
                sol_ANOVA = get_sol_ANOVA(classifier)
                p_val_w_in_gamma = get_p_val_intersection_gene_sets(sol_ANOVA, "w_in_gamma")
                p_val_gamma_in_w = get_p_val_intersection_gene_sets(sol_ANOVA, "gamma_in_w")
                n_genes = get_n_genes(sol_ANOVA)
                supp_dict = {"n_genes": n_genes,
                             "p_val_w_in_gamma": p_val_w_in_gamma,
                             "p_val_gamma_in_w": p_val_gamma_in_w}
            else:
                raise NotImplementedError


            main_res_dict = {"classifier_name": classifier_name,
                        "dataset_name": dataset_name,
                        "class_of_interest": class_of_interest,
                        "index_param": index_param,
                        "CV_index": CV_index}

            res_dict = {**main_res_dict, **supp_dict}

            df_to_append = pd.DataFrame(res_dict, index=[0])
            output_df.append(df_to_append)

    output_df = pd.concat(output_df)
    return output_df