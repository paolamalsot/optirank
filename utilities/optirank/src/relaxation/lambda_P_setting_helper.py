"""
Helper methods to round and scale gamma with different strategies.
"""

import torch
import logging
import numpy as np
from utilities.optirank.src.loss.params import Params_With_Loss


def round_to_01_sum_gamma_constraint(gamma, k):
    """
    Rounds gamma such that there are k components to 1 and d-k to 0. The biggest k are set to 1, others to 0.
    :param gamma: torch tensor
    :param k: integer
    :return: torch tensor
    """
    biggest_k = torch.argsort(gamma, descending=True)[0:k]
    res = torch.clone(gamma)
    res[:] = 0.0
    res[biggest_k] = 1.0
    return res


def scale_gamma(gamma, k):
    """
    Scales the non-0/1 entries of gamma such that the sum gives k.
    :param gamma: numpy array
    :param k: integer
    :return: numpy array
    """
    res = torch.clone(gamma)
    non_zero_one_entries = torch.logical_not(torch.logical_or(gamma == 1, gamma == 0))
    sum_gamma_k_non_zero_one_entries = torch.sum(gamma[non_zero_one_entries])
    sum_gamma_k_i1 = torch.count_nonzero(gamma == 1)
    if sum_gamma_k_i1 > k:  # Warning: one must make sure BEFORE that the rounding is not stupid.
        raise ValueError("sum_gamma_k_i1 > 1")
    scaling_factor = (k - sum_gamma_k_i1) / sum_gamma_k_non_zero_one_entries
    if scaling_factor < 0:
        logging.warning("The scaling factor was negative")
    res[non_zero_one_entries] = gamma[non_zero_one_entries] * scaling_factor
    return res


def round_and_scale_gamma(gamma, indices, k):
    """
    Rounds gamma[indices] to their nearest integer value and scales the remaining such that sum_gamma = k.
    """
    res = torch.clone(gamma)
    res[indices] = torch.round(gamma[indices])
    res = scale_gamma(res, k)
    return res


def indices_strategy_thresh(gamma, t):
    """
    Returns indices that at distance less than t either of 0 or 1.
    """
    return torch.nonzero(torch.logical_or(gamma <= t, gamma >= 1 - t)).flatten()


def thresh_strategy_n(gamma, n):
    """
    Returns the threshold such that n gammas are within threshold distance to 0-1.
    :param gamma: numpy array
    :param n: number of gammas that we want to "trap"
    :return: float thresh
    """
    sorted, indices = torch.sort(torch.abs(gamma - torch.round(gamma)))
    thresh = sorted[n - 1].item()
    return thresh


def lim_max_threshold(gamma, k):
    """
    Returns tuple lim_tresh, max_tresh
    lim_tresh is the limit threshold such that any t over lim_tresh will yield a scaled and rounded gamma with thresh violating (sum_i gamma_i = k or gamma_i in 0,1)
    max_tresh is the maximum threshold that rounding at maximum_treshold will create the "most rounded" gamma pattern without violating the rule.
    NB: any threshold in [max_tresh, lim_thresh[ will do just fine and create the same rounding pattern as with max_thresh
    """
    sorted, indices = torch.sort(gamma)
    d = len(gamma)
    if k < d:
        lim_threshold_1 = 1 - sorted[-k - 1].item()  # otherwise too many 1
        d = len(gamma)
        lim_threshold_2 = sorted[d - k].item()  # otherwise too many 0
        lim_thresh = min(lim_threshold_1, lim_threshold_2)
        if lim_thresh > 0.5:
            lim_thresh = torch.inf
        distance_to_border = dist_to_border(gamma)
        possible_thresholds = distance_to_border[distance_to_border < lim_thresh]
        if len(possible_thresholds) > 0:
            max_thresh = torch.max(possible_thresholds).item()
        else:
            max_thresh = 0
        return lim_thresh, max_thresh
    elif k == d:
        logging.info("k = d. So max_threshold does not have a sense..")
        return 0, 0  # not sure this is the most meaningful answer
    else:
        raise ValueError("k > d")


def dist_to_border(gamma):
    return torch.abs(gamma - torch.round(gamma))


def sum_dist_to_border(gamma):
    return torch.sum(dist_to_border(gamma))


def rounding_and_scaling_thresh(gamma, k, t, cut_beyond_max_thresh=True):
    """
    Rounds gamma indices within t distance of 0/1 . Scales the remaining indices such that sum_gamma = k.
    The default cut_beyond_max_thresh option ensures that the specified threshold does not yield forbidden gamma values.
    :param gamma: torch tensor of size d
    :param k: integer
    :param t: scalar
    :param cut_beyond_max_thresh: boolean indicating whether to cut threshold with max_threshold.
    :return: gamma copy rounded and scaled.
    """
    lim_thresh, max_thresh = lim_max_threshold(gamma, k)
    if t > lim_thresh:
        logging.warning("t {}  bigger than lim_tresh, {}".format(t, lim_thresh))
    if cut_beyond_max_thresh:
        t = min(t, max_thresh)
    indices_thresh = indices_strategy_thresh(gamma, t)
    res = round_and_scale_gamma(gamma, indices_thresh, k)
    return res


def rounding_thresh(gamma, t):
    """
    Rounds all gamma entries within t distance of 0/1.
    """
    indices_thresh = indices_strategy_thresh(gamma, t)
    res = torch.clone(gamma)
    res[indices_thresh] = torch.round(gamma[indices_thresh])
    return res


def get_indices_round(sol: Params_With_Loss, per_remaining, t):
    """
    Returns indices to round such that both following conditions are met:
    - at least per_remaining of non-0/1 gammas are rounded
    - all gammas within t distance of 0/1 are rounded
    - at least one gamma is rounded.
    :param sol: Params_with_Loss object
    :param per_remaining: scalar
    :param t: scalar
    :return: torch tensor with indices
    """
    k = sol.loss_object.constraint_sum_gamma_k
    if k is None:
        lim_thresh = np.inf
        max_thresh = np.inf
    else:
        lim_thresh, max_thresh = lim_max_threshold(sol.gamma, k)
    logging.debug("lim_tresh = {}".format(lim_thresh))
    logging.debug("max_thresh = {}".format(max_thresh))
    n_0_or_1 = np.count_nonzero(torch.logical_or(sol.gamma == 1, sol.gamma == 0))
    n_i_r = sol.loss_object.d - n_0_or_1
    n_i_r_to_block = int(max(per_remaining * n_i_r, 1))  # ensures at least one gamma entry is rounded.
    logging.debug("n_i_r_to_block = {}".format(n_i_r_to_block))
    n_to_block = n_0_or_1 + n_i_r_to_block
    logging.debug("n_tot_to_block = {}".format(n_to_block))
    thresh_n = thresh_strategy_n(sol.gamma, n_to_block)
    logging.debug("thresh_n = {}".format(thresh_n))
    thresh_final = min(max_thresh, max(t, thresh_n))
    logging.debug("thresh_final = {}".format(thresh_final))
    indices_round = indices_strategy_thresh(sol.gamma, thresh_final)
    return indices_round
