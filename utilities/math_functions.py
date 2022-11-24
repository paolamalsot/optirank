import torch
from utilities.optirank.ranking_multiplication import ranking_transformation


def rankdata_fun(x, rank_type="min", normalization=None, gamma=None):
    d = x.shape[1]

    if gamma == None:
        gamma = torch.ones(d)

    return ranking_transformation(x, gamma, rank_type, normalization)
