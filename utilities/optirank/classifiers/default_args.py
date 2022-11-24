#DEFAULT ARGUMENTS#

import torch
import numpy as np
from utilities.optirank.src.BCD.Create_Instructions_List import classical_architecture_separate_b_once
from utilities.small_functions import percentage_zero, percentage_ones
from utilities.optirank.src.relaxation.lambda_P_setting import delta_loss
from utilities.optirank.src.BCD.BCD_units.convergence_criterion import absolute_delta_args


default_BCD_args = {"BCD_architecture": classical_architecture_separate_b_once, "max_iter": 10000, "search_method__L_min": 10**(-10),
                 "search_method__eta": [1.5, 1.5], "search_method__init_L_method": "hessian_proj", "initializazionparameters__name": "gamma_05_w_0",
                 "search_method__n_min": -1, "search_method__L_start": "previous",
                 "search_method__search_method_name": "first_best", "search_method__n_max": np.inf}

#chosen after inspection of setting lambda_P strategy
default_setting_lambda_P_strategy_args = {"class":delta_loss, "args":{"M": 100, "delta_lambda_min": 10**(-20), "with_interpolation": False}}
default_bilinear_ranking_classifier_args = {"rounding_threshold": 0.0, "setting_lambda_P_strategy_args": default_setting_lambda_P_strategy_args, "convergence_criterion_args": absolute_delta_args, "high_tol": False, "max_relaxation_iter": 10000, "tol_dist_to_border": 10**(-10), **default_BCD_args}
default_optirank_args = {**default_bilinear_ranking_classifier_args, "R_normalization": "k"}
#default_bilinear_optirank_args_no_constraint_sum_gamma = {**default_bilinear_ranking_classifier_args, "R_normalization": "d"}

#functions for diagnostics
subgradients_funs_dict = {
        "|dsurrogate_loss/dw|min": lambda p: torch.norm(p.subgradient_minimal_norm_surrogate_loss_on_w()).item(),
        "|dsurrogate_loss/dgamma_dual|": lambda p: torch.norm(p.gradient_surrogate_loss_with_penalties_dgamma_dual()).item(),
        "|dsurrogate_loss/db|":lambda p: torch.norm(p.dlogloss_db()).item(),
        "|dsurrogate_loss/dgamma|proj": lambda p: torch.norm(p.gradient_surrogate_loss_on_gamma()).item()
}

percentages_funs_dict = {"perc_gamma_1": lambda p: percentage_ones(p.gamma.numpy()),
                         "perc_gamma_0": lambda p: percentage_zero(p.gamma.numpy())}