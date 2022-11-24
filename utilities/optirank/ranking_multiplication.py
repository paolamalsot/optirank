import numpy as np
import torch
from scipy.stats import rankdata

R_normalization_methods = [None, "d", "d*sqrt(d)", "k", "k*sqrt(k)"]
constraint_sum_gamma_k_normalization_methods = R_normalization_methods
no_constraint_sum_gamma_k_normalization_methods = [None, "d", "d*sqrt(d)"]


def N(R_normalization_method, d, k):
    """
    :param string R_normalization_method: either "d", d*sqrt(d)", "k", and "k*sqrt(k)". The second two make more sense
      when the constraint sum(gamma) = k is active.
    :param int d: variable w (and gamma) dimension
    :param int k: k involved in the constraint sum(gamma) = k
    :return float: normalization factor
    """
    if R_normalization_method == "d":
        return d
    elif R_normalization_method == "d*sqrt(d)":
        return d * np.sqrt(d)
    elif R_normalization_method == "k":
        return k
    elif R_normalization_method == "k*sqrt(k)":
        return k * np.sqrt(k)
    elif R_normalization_method is None:
        return 1


def ranking_transformation(X, gamma, rank_type, R_normalization):
    """
    :param torch tensor X: design matrix Nxd
    :param torch tensor gamma: gamma variable (d)
    :param rank_type: ranking type ("min", "max" and "avg")
    :param string R_normalization: normalization method (either "d", d*sqrt(d)", "k", and "k*sqrt(k)")
    :return torch tensor: returns the result of the ranking transformation with respect to the ranking reference encoded
     in gamma
    """
    forward_and_backward_sort_indices_ = forward_and_backward_sort_indices(X)
    d = X.shape[1]
    k = torch.sum(gamma)
    N_ = N(R_normalization, d, k)
    return (Rgamma(gamma, *forward_and_backward_sort_indices_, type=rank_type) + offset_for_types[rank_type]) / N_


def ranking_tensor(X, type):
    """
    returns the ranking tensor R of the design matrix X. If R_sij = 1 means that X_si > X_sj
    :param numpy array X: n_samples x n_genes design matrix
    :param string type: rank-type
    :param float N: normalization_factor
    :return numpy array:binary n_samples x n_genes x n_genes ranking tensor
    """

    X_3D = (np.expand_dims(X, 2) > np.expand_dims(X, 1)).astype('ubyte')
    X_equality = (np.expand_dims(X, 2) == np.expand_dims(X, 1)).astype('ubyte')
    if X_3D.shape != (X.shape[0], X.shape[1], X.shape[1]):
        raise ValueError('The shape of the 3D tensor is not what I expected')

    if type == "min":
        X_out = X_3D
    elif type == "max":
        X_out = X_3D + X_equality
    elif type == "avg":
        X_out = X_3D + 0.5 * X_equality
    else:
        return "Wrong type"

    return X_out


def forward_and_backward_sort_indices(X):
    """
    function that calculates the forward and backward sort indices for the each sample of the design matrix X. The
    forward indices are the indices such that for a sample x (of size d), x[forward] returns a sorted orray.
    backward_min and backward max are the ranked version of X with the min and max method respectively.
    :param numpy array X: design matrix (nxd)
    :return (torch tensor, torch tensor, torch tensor): (forward, backward_min, backward_max) tuple.
    """
    forward = torch.from_numpy(np.argsort(X, axis=1)).to(dtype=torch.int64)
    backward_min = torch.from_numpy(rankdata(X, axis=1, method="min") - 1).to(dtype=torch.int64)
    backward_max = torch.from_numpy(rankdata(X, axis=1, method="max") - 1).to(dtype=torch.int64)
    return forward, backward_min, backward_max


def Rgamma(gamma, forward_sort_indices, backward_min_sort_indices, backward_max_sort_indices, type):
    """
    :param torch tensor gamma: ranking reference gamma (tensor of size d)
    :param torch tensor forward_sort_indices: forward sort indices (tensor of size d)
    :param torch tensor backward_min_sort_indices: backward min sort indices (tensor of size d)
    :param torch tensor backward_max_sort_indices: backward max sort indices (tensor of size d)
    :param string type: ranking type ("min", "max", "avg")
    :return torch tensor : result of the multiplication R gamma (nxd), with R the ranking tensor corresponding to the
      forward and backward sort indices.
    """
    forward_gamma = torch.take(gamma, forward_sort_indices)

    res = {}
    if type == "min" or type == "avg":
        result = torch.cumsum(forward_gamma, dim=1) - forward_gamma
        res["min"] = torch.take_along_dim(result, backward_min_sort_indices, dim=1)
    if type == "max" or type == "avg":
        result = torch.cumsum(forward_gamma, dim=1) - 1
        res["max"] = torch.take_along_dim(result, backward_max_sort_indices, dim=1)
    if type == "avg":
        res["avg"] = (res["min"] + res["max"]) * 0.5
    return (res[type] - offset_for_types[type])


def wTR(w, forward_sort_indices, backward_min_sort_indices, backward_max_sort_indices, type):
    """
    :param torch tensor w: regression coefficients w (tensor of size d)
    :param torch tensor forward_sort_indices: forward sort indices (tensor of size d)
    :param torch tensor backward_min_sort_indices: backward min sort indices (tensor of size d)
    :param torch tensor backward_max_sort_indices: backward max sort indices (tensor of size d)
    :param string type: ranking type ("min", "max", "avg")
    :return torch tensor : result of the multiplication w.T R (nxd), with R the ranking tensor corresponding to the
      forward and backward sort indices.
    """
    forward_w = torch.flip(w[forward_sort_indices], dims=[1])
    res = {}
    if type == "min" or type == "avg":
        int_result_min = torch.flip(torch.cumsum(forward_w, dim=1) - forward_w, dims=[1])
        res["min"] = torch.take_along_dim(int_result_min, backward_max_sort_indices, dim=1)
    if type == "max" or type == "avg":
        int_result_max = torch.flip(torch.cumsum(forward_w, dim=1), dims=[1])
        res["max"] = torch.take_along_dim(int_result_max, backward_min_sort_indices, dim=1)
    if type == "avg":
        res["avg"] = (res["min"] + res["max"]) * 0.5
    return res[type]


offset_for_types = {"min": 0,
                    "max": -1,
                    "avg": -0.5}


def zs_from_Rgamma(w, Rgamma, b, type, N):
    """
    :param torch tensor w: size d tensor with regression coefficients
    :param torch tensor Rgamma: size n x d tensor with the result of R @ gamma
    :param float b: regression offset
    :param string type: rank-type
    :param float N: scalar normalizing factor
    :return torch tensor: size n tensor with regression logistic scores
    """
    return ((Rgamma + offset_for_types[type]) / N) @ w + b


def zs_from_wTR(wTR, gamma, w, b, type, N):
    """
    :param torch tensor wTR: size n x d tensor with the resut of w.T @ R
    :param torch tensor gamma: size d tensor with ranking reference
    :param torch tensor w: size d tensor with regression coefficients
    :param float b: regression offset
    :param string type: rank-type
    :param float N: normalization factor
    :return: torch tensor: size n tensor with regression logistic scores
    """
    return wTR / N @ gamma + offset_for_types[type] / N * torch.sum(w) + b


def zs_from_w_gamma_b(X, w, gamma, b, type, N):
    """
    :param numpy array X: design matrix (nxd)
    :param torch tensor w: size d tensor with regression coefficients
    :param torch tensor gamma: size d tensor with ranking reference
    :param float b: regression offset
    :param string type: rank-type
    :param float N: normalization factor
    :return: torch tensor: size n tensor with regression logistic scores
    """
    forward, backward_min, backward_max = forward_and_backward_sort_indices(X)
    wTR_ = wTR(w, forward, backward_min, backward_max, type)
    return zs_from_wTR(wTR_, gamma, w, b, type, N)
