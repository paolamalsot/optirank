# PROXIMAL OPERATORS #

from abc import ABC, abstractmethod
from utilities.optirank.src.loss.params import Params_With_Loss
from utilities.optirank.src.BCD.prox.prox_operators import prox_w, prox_b, prox_gamma, prox_gamma_dual, \
    prox_gamma_sum_to_k
import numpy as np


class Prox_Step(ABC):
    """ Abstract class for proximal operators """

    def __init__(self):
        pass

    def step(self, p_b: Params_With_Loss, L):
        """
        :param p_b: Parameter before
        :param L: inverse stepsize
        :return: next p, dp pair
        """
        p = self.prox(p_b, L)
        dp = None  # saving time

        return p, dp

    @abstractmethod
    def prox(self, p_b: Params_With_Loss, L):
        pass


var_names = ["w", "gamma", "gamma_dual", "b"]


class Multivariate_Prox_Step(Prox_Step):
    """ Proximal operators for multiple variables (for example w and b)"""

    def __init__(self, vars):
        self.vars = vars
        if np.any([(var not in var_names) for var in vars]):
            raise ValueError("Incorrect variable name")
        super().__init__()

    def prox(self, p_b: Params_With_Loss, L):
        new_vars = {}
        for var in self.vars:
            new_vars[var] = prox_functions_dict[var]["normal"](p_b, L)
        for var in complement_vars(self.vars):
            new_vars[var] = getattr(p_b, var).detach().clone()
        new_vars["loss_object"] = p_b.loss_object
        p = Params_With_Loss(**new_vars)
        return p


def complement_vars(vars):
    """
    Returns the variables that are not changing during the prox_step w.r.t vars (creates a speedup in time)
    :param vars:
    :return:
    """
    if set(vars) == {"w", "b"}:
        return ["gamma", "gamma_dual", "Rgamma"]
    elif set(vars) == {"gamma", "b"}:
        return ["w", "gamma_dual", "wTR"]
    elif set(vars) == {"w", "gamma_dual", "b"}:
        return ["gamma", "Rgamma"]
    elif set(vars) == {"gamma"}:
        return ["w", "gamma_dual", "wTR", "b"]
    elif set(vars) == {"w"}:
        return ["gamma", "gamma_dual", "Rgamma", "b"]
    elif set(vars) == {"b"}:
        return ["Rgamma", "wTR", "gamma", "w", "gamma_dual"]
    else:
        raise NotImplementedError


def prox_gamma_(p: Params_With_Loss, L):
    if p.loss_object.constraint_sum_gamma_k is not None:
        k = p.loss_object.constraint_sum_gamma_k
        return prox_gamma_sum_to_k(p, L, k)
    else:
        return prox_gamma(p, L)


prox_gamma_dual_ = lambda p, L: prox_gamma_dual(p, L, prox_gamma_dual=True)
prox_functions_dict_w = {"normal": prox_w}
prox_functions_dict_gamma = {"normal": prox_gamma_}
prox_functions_dict_gamma_dual = {"normal": prox_gamma_dual_}
prox_functions_dict_b = {"normal": prox_b}

prox_functions_dict = {"w": prox_functions_dict_w,
                       "gamma": prox_functions_dict_gamma,
                       "gamma_dual": prox_functions_dict_gamma_dual,
                       "b": prox_functions_dict_b}
