import torch
from utilities.optirank.src.loss.params import Params


def get_sol_ANOVA(classifier):
    w = torch.from_numpy(classifier.named_steps["logisticregression"].coef_.flatten())
    b = torch.from_numpy(classifier.named_steps["logisticregression"].intercept_)
    gamma = classifier.named_steps["anova_subset_ranking"].gamma_
    gamma_dual = torch.zeros_like(gamma)
    sol = Params(w, gamma, b, gamma_dual)
    return sol
