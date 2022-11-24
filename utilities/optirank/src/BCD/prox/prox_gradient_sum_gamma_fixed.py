from utilities.optirank.src.BCD.prox.sum_gamma_fixed.find_intersection import find_intersection_bis
import torch
import numpy as np


def prox_subgradient(x, i_0, i_1):
    """
    Returns h such that h is argmin||x-h||^2, under the constraints that sum_i h_i=0, h_i <= 0 if i \in I_0, h_i >=0 if i \in I_1
    :param x: torch array
    :param i_0: torch boolean array
    :param i_1: torch boolean array
    :return: h, torch array
    """
    lambda_intersect, _, _, _, _ = find_intersection_bis(x.numpy(), i_0.numpy(), i_1.numpy())
    i_r = torch.logical_not(torch.logical_or(i_0, i_1))
    # indexes = np.array([np.argwhere(el).item() for el in np.stack((i_0, i_1, i_r)).transpose()])
    indexes = np.argwhere(np.stack((i_0.numpy(), i_1.numpy(), i_r.numpy())).transpose())[:, 1]
    to_clip = x.numpy() - lambda_intersect
    arrays = [np.clip(to_clip, None, 0), np.clip(to_clip, 0, None), to_clip]
    h = torch.from_numpy(np.choose(indexes, arrays))
    return h
