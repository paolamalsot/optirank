import torch
from utilities.optirank.src.BCD.prox.sum_gamma_fixed.find_intersection import find_intersection


def prox_sum_to_k(x, k):
    """
    finds z such that z is argmin (z-x)**2 with sum_i z_i = k and every z_i \in 0-1
    :param x: torch tensor size d
    :param k: integer
    :return: z, torch tensor
    """
    lambda_interest, _, _, _, _ = find_intersection(x.numpy(), k)
    return torch.clamp(x - lambda_interest, 0, 1)
