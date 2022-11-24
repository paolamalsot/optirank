import numpy as np
from utilities.optirank.src.BCD.prox.sum_gamma_fixed.cum_sum_until import do_cumsum


def find_intersection(x, k):
    """
    finds lambda such that psi(lambda) = k. psi(lambda) = sum_i{clip(x_i-lambda, 0, 1)}
    :param x: numpy array of size d
    :param k: integer
    :return: lambda (float), f_i_minus_1, f_i, xs_sorted
    """
    indices_sorted = np.argsort(-np.concatenate((x, x - 1)))  # because of minus sorts in the descending order
    xs_sorted = np.concatenate((x, x - 1))[indices_sorted]
    ids_from_right = np.concatenate((np.ones_like(x), - np.ones_like(x)))[indices_sorted]

    derivatives = np.cumsum(ids_from_right)  # left derivative at point xs_sorted
    dxs = np.concatenate((xs_sorted[0:-1] - xs_sorted[1:], np.array([0])))  # xs_sorted[i] - xs_sorted[i+1]
    delta_fs = dxs * derivatives  # delta between f[xs_sorted[i+1]] and f[xs_sorted[i]]
    delta_fs = np.concatenate((np.array([0]), delta_fs[0:-1]))
    size_array = len(delta_fs)
    index_cython, f_i_minus_1_cython, f_i_cython = do_cumsum(delta_fs, k)

    # little trick:
    if index_cython == size_array - 1:
        f_i_cython = len(x)
        f_i_minus_1_cython = len(x) - delta_fs[-1]

    # solve the linear problem
    # k = f_i_minus_1_cython + derivatives[index_cython - 1] * abs(lambda - xs_sorted[index_cython - 1])

    if f_i_minus_1_cython == f_i_cython:
        lambda_intersect = xs_sorted[index_cython]
    else:
        lambda_intersect = - (k - f_i_minus_1_cython) / derivatives[index_cython - 1] + xs_sorted[index_cython - 1]

    return lambda_intersect, index_cython, f_i_minus_1_cython, f_i_cython, xs_sorted


def find_intersection_bis(x, indicator_x_i0, indicator_x_i1):
    """
    Find lambda_interesect such that psi(lambda_intersect) = 0. psi(lambda) = sum_i{h_i}. h_i = clip(x_i-lambda, 0, inf) if i \in I_1.
    h_i = clip(x_i-lambda, -inf, 0) if i \in I_0, h_i = x_i - lambda
    :param x: numpy array
    :param indicator_x_io: boolean array that indicates in x_i belongs to I_0
    :param indicator_x_i1: boolean array that indicates in x_i belongs to I_1
    :return: lambda_intersect, i, psi_x_(i-1), psi_x_i
    Note that lambda_intersect is between x_(i-1) and x_i
    #verifier que ça marche qd ça tombe au mm endroit!
    """
    order = np.argsort(x)
    x_ordered = x[order]
    indicator_x_ir = np.logical_not(np.logical_or(indicator_x_i0, indicator_x_i1))
    indicator_x_i0_ordered = indicator_x_i0[order]
    indicator_x_i1_ordered = indicator_x_i1[order]

    if np.any(np.logical_and(indicator_x_i0, indicator_x_i1)):
        raise ValueError("x_i cannot belong to both I_0 and I_1")

    # dpsi_dlambda(i) = dpsi_dlambda between x_(i) and x_(i+1)
    # We sum 3 contributions: from i \in I_r, i \in I_0 and I \in I_1
    dpsi_dlambda_i_r = - np.ones_like(x) * np.count_nonzero(indicator_x_ir)
    dpsi_dlambda_i_1 = - np.flip(np.cumsum(
        np.flip(indicator_x_i1_ordered))) + indicator_x_i1_ordered  # in order to have the correct left continuity...
    dpsi_dlambda_i_0 = - np.cumsum(indicator_x_i0_ordered)
    dpsi_dlambda = dpsi_dlambda_i_r + dpsi_dlambda_i_1 + dpsi_dlambda_i_0

    delta_x = np.concatenate((x_ordered[1:] - x_ordered[0:-1], np.array([0])))  # x_i+1 - x_i
    dpsi = dpsi_dlambda * delta_x  # psi(x_(i+1)) - psi(x_(i))

    # now calculate psi at x_(0)
    psi_x_0 = np.sum(x[np.logical_or(indicator_x_i1, indicator_x_ir)] - x_ordered[0], keepdims=True)  # actually x_(0)

    # now calculate the point where psi gets below 0 <-> -psi gets over 0!
    to_cum_sum = np.concatenate((psi_x_0, dpsi))  # the result of the cum_sum of to_cum_sum at i will be psi(x_(i))
    i, minus_psi_i_minus_1, minus_psi_i = do_cumsum(-to_cum_sum, 0)
    # cum_sum = np.cumsum(-to_cum_sum)
    # i = np.argwhere(cum_sum>=0)[0].item()
    # minus_psi_i_minus_1 = cum_sum[i-1]
    # minus_psi_i = cum_sum[i]
    psi_i_minus_1 = - minus_psi_i_minus_1
    psi_i = - minus_psi_i

    if dpsi_dlambda[i - 1] == 0:
        lambda_intersect = x_ordered[i - 1]
    else:
        lambda_intersect = x_ordered[i - 1] - psi_i_minus_1 / dpsi_dlambda[i - 1]

    return lambda_intersect, i, psi_i_minus_1, psi_i, x_ordered
