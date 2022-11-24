import torch
from utilities.optirank.ranking_multiplication import zs_from_Rgamma, zs_from_wTR, Rgamma, \
    forward_and_backward_sort_indices
from utilities.optirank.src.loss.loss import ranking_logistic_loss
import scipy
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from utilities.optirank.src.BCD.prox.prox_gamma_sum_gamma_fixed import prox_sum_to_k


class Params():
    """ This class implements the variables in the loss that are to be optimized: w, gamma, b, gamma_dual. """

    def __init__(self, w, gamma, b, gamma_dual):
        """
        :param w: torch tensor of size d
        :param gamma: torch tensor of size d
        :param b: torch scalar
        :param gamma_dual: torch tensor of size d
        """
        self._w = w
        self._gamma = gamma
        self.b = b
        self.gamma_dual = gamma_dual

        # if torch.any(torch.isnan(self._gamma)) or torch.any(torch.isnan(self.gamma_dual)) or torch.any(torch.isnan(self._w)) or torch.any(torch.isnan(self.b)):
        #     raise ValueError("Getting nan values") #removed because was taking a lot of time!

    # dummy properties for w and gamma #look into https://www.geeksforgeeks.org/python-property-decorator-property/
    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, val):
        self._w = val

    @w.deleter
    def w(self):
        del self._w

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, val):
        self._gamma = val

    @gamma.deleter
    def gamma(self):
        del self._gamma

    @property
    def gamma_dual(self):
        return self._gamma_dual

    @gamma_dual.setter
    def gamma_dual(self, val):
        self._gamma_dual = val

    @gamma_dual.deleter
    def gamma_dual(self):
        del self._gamma_dual

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, val):
        self._b = val

    @b.deleter
    def b(self):
        del self._b

    def dtype(self):
        return self.w.dtype

    def to(self, dtype=None, copy=True):
        """
        This method is used both for dtype conversion (float32 and float64), and for cloning!
        :param dtype: float32, float64 or None for no type conversion.
        :param copy:  boolean flag to ask for a copy
        :return: self or copy of self
        """
        if dtype not in [torch.float32, torch.float64, None]:
            raise ValueError("Wrong dtype")
        if dtype is None:
            dtype = self.dtype()

        w = self.w.to(dtype=dtype)
        gamma = self.gamma.to(dtype=dtype)
        b = self.b.to(dtype=dtype)
        gamma_dual = self.gamma_dual.to(dtype=dtype)

        if copy:
            return Params(w.detach().clone(), gamma.detach().clone(), b.detach().clone(), gamma_dual.detach().clone())

        else:
            self.w = w
            self.gamma = gamma
            self.gamma_dual = gamma_dual
            self.b = b
            return self

    def __add__(self, other):
        gamma = self.gamma + other.gamma
        b = self.b + other.b
        gamma_dual = self.gamma_dual + other.gamma_dual
        w = self.w + other.w
        return Params(w, gamma, b, gamma_dual)

    def __neg__(self):
        gamma = - self.gamma
        b = - self.b
        gamma_dual = - self.gamma_dual
        w = - self.w
        return Params(w, gamma, b, gamma_dual)

    def __sub__(self, other):
        return self + (-other)

    def norm(self):
        return (torch.norm(self.gamma) ** 2 + torch.norm(self.w) ** 2 + self.b ** 2 + torch.norm(
            self.gamma_dual) ** 2) ** 0.5


class Params_With_Loss(Params):
    """ This class represents a Params object associated to a loss object. For computational reasons,
    wTR and Rgamma calculations are cached and replaced when needed. For numerical precision, the via_ attribute records whether zs
    was calculated using Rgamma or wTR."""

    def __init__(self, w, gamma, b, gamma_dual, loss_object: ranking_logistic_loss, Rgamma=None, wTR=None):
        """
        :param w: torch tensor of size d
        :param gamma: torch tensor of size d
        :param b: torch tensor of size 1
        :param gamma_dual: torch tensor of size d
        :param loss_object: Loss object
        :param Rgamma: torch tensor of size nxd or None
        :param wTR: torch tensor of size nxd or None
        """
        super().__init__(w, gamma, b, gamma_dual)
        self.loss_object = loss_object
        self._Rgamma = Rgamma
        self._wTR = wTR
        self.erase_saved_calculations()

    # not so dummy properties for w and gamma #look into https://www.geeksforgeeks.org/python-property-decorator-property/
    # The main goal of adding properties is for wTR and Rgamma, which are saved in order to save computation time.
    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, val):
        self._wTR = None
        self._w = val
        self.erase_saved_calculations()

    @w.deleter
    def w(self):
        del self._w

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, val):
        self._Rgamma = None
        self._gamma = val
        self.erase_saved_calculations()

    @gamma.deleter
    def gamma(self):
        del self._gamma

    @property
    def gamma_dual(self):
        return self._gamma_dual

    @gamma_dual.setter
    def gamma_dual(self, val):
        self._gamma_dual = val
        self.erase_saved_calculations()

    @gamma_dual.deleter
    def gamma_dual(self):
        del self._gamma_dual

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, val):
        self._b = val
        self.erase_saved_calculations()

    @b.deleter
    def b(self):
        del self._b

    @property
    def wTR(self):
        if self._wTR is None:
            self._wTR = self.loss_object.wTR(self.w)
        return self._wTR

    @property
    def Rgamma(self):
        if self._Rgamma is None:
            self._Rgamma = self.loss_object.Rgamma(self.gamma)
        return self._Rgamma

    def to(self, dtype=None, copy=True, only_params=False, copy_loss=None):
        """
         This method is used both for dtype conversion (float32 and float64), and for cloning!
        :param dtype: float32, float64 or None for no type conversion.
        :param copy: boolean flag to ask for a copy
        :param only_params: returns a Params object, instead of a Params_with_loss object
        :param copy_loss: boolean flag to indicate if the loss object must be copied. By default, is the same as copy.
        NB: copying the loss involves copying X and y data, which can be extremely expensive.
        :return: self or copy of self
        """
        if dtype is None:
            dtype = self.loss_object.dtype
        if copy_loss is None:
            copy_loss = copy

        res_ = super().to(dtype, copy)
        if only_params:
            return res_
        else:
            loss = self.loss_object.to(dtype=dtype, copy=copy_loss)
            if self._Rgamma is not None:
                Rgamma = self._Rgamma.to(dtype=dtype)
            else:
                Rgamma = None
            if self._wTR is not None:
                wTR = self._wTR.to(dtype=dtype)
            else:
                wTR = None

            if copy:
                return Params_With_Loss(res_.w, res_.gamma, res_.b, res_.gamma_dual, loss, Rgamma,
                                        wTR)  # hope it works!
            else:
                self.w = res_.w
                self.gamma = res_.gamma
                self.b = res_.b
                self.gamma_dual = res_.gamma_dual
                self.loss = loss
                self._Rgamma = Rgamma
                self._wTR = wTR

                return self

    def copy_except_loss(self):
        """ Returns a copy of params with the same loss object (not a full copy)."""
        new_p = Params_With_Loss(self.w.detach().clone(), self.gamma.detach().clone(), self.b.detach().clone(),
                                 self.gamma_dual.detach().clone(), self.loss_object)
        calculation_vars = ["_Rgamma", "_wTR"]
        for var in calculation_vars:
            res = getattr(self, var)
            if res is not None:
                setattr(new_p, var, res.detach().clone())
        return new_p

    def set_lambda_P(self, lambda_P):
        """ Modifies lambda_P of the loss of params."""
        loss_object_copy = self.loss_object.to(dtype=torch.float64, copy=True, copy_data=False)
        self.loss_object = loss_object_copy
        self.loss_object.lambda_P = lambda_P
        self.erase_saved_calculations()

    def to_lightweight(self, copy=True, dtype=torch.float32):
        """
        returns a lightweight version of self (by copying or erasing calculations), generally used for pickling.
        :param copy: boolean flag to ask for a copy
        :param dtype: float32 or float64, for dtype conversion. If None, no type conversion
        :return: a lightweight version/copy of self
        """
        loss = self.loss_object.to_lightweight(copy=copy)
        res_ = self.to(dtype=dtype, copy=copy)
        if copy:
            return Params_With_Loss(res_.w, res_.gamma, res_.b, res_.gamma_dual, loss, None, None)
        else:
            self = Params_With_Loss(res_.w, res_.gamma, res_.b, res_.gamma_dual, loss, None, None)
            return self

    def erase_wTR_Rgamma(self):
        self._Rgamma = None
        self._wTR = None

    def erase_saved_calculations(self):
        self.zs_ = None
        self.via_ = None
        self.logloss_ = None
        self.dlogloss_dgamma_ = None
        self.dlogloss_dw_ = None
        self.dlogloss_db_ = None
        self.surrogate_loss_with_penalties_ = None

    def zs_from_wTR(self):
        """calculates zs by using wTR calculation (either already cached or to be calculated)"""
        if (self.via_ != "wTR") or (self.zs_ is None):
            self.zs_ = zs_from_wTR(self._wTR, self.gamma, self.w, self.b, self.loss_object.rank_type,
                                   self.loss_object.N)
            self.via_ = "wTR"
        return self.zs_

    def zs_from_Rgamma(self):
        """calculates zs by using Rgamma calculation (either already cached or to be calculated)"""
        if (self.via_ != "Rgamma") or (self.zs_ is None):
            self.zs_ = zs_from_Rgamma(self.w, self.Rgamma, self.b, self.loss_object.rank_type, self.loss_object.N)
            self.via_ = "Rgamma"
        return self.zs_

    def zs(self, via=None):
        """calculates zs by using wTR/Rgamma calculation (either already cached or to be calculated)
        :param:via: either None, "Rgamma", or "wTR". If None do the least computationally expensive calculation
        using caches."""
        if via is None:
            if self.zs_ is None:
                if self._Rgamma is not None:
                    self.zs_from_Rgamma()
                elif self._wTR is not None:
                    self.zs_from_wTR()
                else:
                    self.zs_from_Rgamma()
            else:
                return self.zs_
        else:
            if via == "Rgamma":
                self.zs_from_Rgamma()
            elif via == "wTR":
                self.zs_from_wTR()
        return self.zs_

    # LOGISTIC LOSS CALCULATIONS (logloss)

    def logloss(self, via=None):
        if self.logloss_ is None:
            self.logloss_ = self.loss_object.logloss(self.zs(via=via))
        return self.logloss_

    def dlogloss_dgamma(self):
        if self.dlogloss_dgamma_ is None:
            self.dlogloss_dgamma_ = self.loss_object.dlogloss_dgamma(self.wTR, self.zs())
        return self.dlogloss_dgamma_

    def dlogloss_dw(self):
        if self.dlogloss_dw_ is None:
            self.dlogloss_dw_ = self.loss_object.dlogloss_dw(self.Rgamma, self.zs())
        return self.dlogloss_dw_

    def dlogloss_db(self):
        if self.dlogloss_db_ is None:
            self.dlogloss_db_ = self.loss_object.dlogloss_db(self.zs())
        return self.dlogloss_db_

    # SURROGATE_LOSS_WITH_PENALTIES: loss objective including logloss, regularization penalties, and dual variables.
    def surrogate_loss_with_penalties(self):
        if self.surrogate_loss_with_penalties_ is None:
            self.surrogate_loss_with_penalties_ = self.logloss() + self.loss_object.surrogate_penalties(self.w,
                                                                                                        self.gamma,
                                                                                                        self.gamma_dual)
        return self.surrogate_loss_with_penalties_

    def gradient_surrogate_loss_on_gamma(self):
        if self.loss_object.constraint_sum_gamma_k is not None:
            return self.loss_object.projected_gradient_surrogate_loss_with_penalties_dgamma_constraint_sum_gamma(
                self.dlogloss_dgamma(), self.gamma, self.gamma_dual)
        else:
            return self.loss_object.subgradient_surrogate_loss_with_penalties_dgamma(self.dlogloss_dgamma(), self.gamma,
                                                                                     self.gamma_dual)

    def subgradient_minimal_norm_surrogate_loss_on_w(self):
        return self.loss_object.subgradient_minimal_norm_surrogate_loss_with_penalties_dw(self.dlogloss_dw(), self.w)

    def gradient_surrogate_loss_with_penalties_dgamma_dual(self):
        return self.loss_object.gradient_surrogate_loss_with_penalties_dgamma_dual(self.gamma, self.gamma_dual)

    def __add__(self, other):
        res = super().__add__(other)
        if torch.any(other.gamma != 0) and torch.any(other.w != 0):
            Rgamma = None
            wTR = None
        else:  # only one of w or gamma is modified
            if (self._Rgamma is not None) and (other._Rgamma is not None):
                Rgamma = self.Rgamma + other.Rgamma
            else:
                Rgamma = None
            if (self._wTR is not None) and (other._wTR is not None):
                wTR = self.wTR + other.wTR
            else:
                wTR = None
        if self.loss_object is other.loss_object:
            loss_object = self.loss_object
        else:
            raise ValueError("Not the same loss object")
        return Params_With_Loss(res.w, res.gamma, res.b, res.gamma_dual, loss_object, Rgamma=Rgamma, wTR=wTR)

    def __neg__(self):
        res = super().__neg__()

        if self._Rgamma is not None:
            Rgamma = - self.Rgamma
        else:
            Rgamma = None
        if self._wTR is not None:
            wTR = - self.wTR
        else:
            wTR = None

        loss_object = self.loss_object

        return Params_With_Loss(res.w, res.gamma, res.b, res.gamma_dual, loss_object, Rgamma=Rgamma, wTR=wTR)

    def __sub__(self, other):
        return self + (-other)

    def dloss_with_penalties_dgamma_in_0_1(self):
        """
        calculates the coordinate-wise derivative of the loss if each of the gamma_i was replaced by 0 or 1
        returns a dx2 array
        """
        return self.loss_object.dloss_with_penalties_dgamma_in_0_1(self.wTR, self.zs(), self.gamma)

    def accuracy(self, balanced=None):
        """
        calculates the accuracy score of the associated logistic regression predictor.
        :param balanced: boolean flag for calculating balanced accuracy.
        If None, same as loss object balanced flag.
        """
        y_pred = self.loss_object.y_pred(self.zs())
        y = self.loss_object.y_np
        if balanced is None:
            balanced = self.loss_object.sample_weight == "balanced"  # balanced or None
        if balanced:
            return balanced_accuracy_score(y, y_pred)  # not sure how much this is correct with label binarizer etc..
        else:
            return accuracy_score(y, y_pred)


def predict_probas(param, X):
    return scipy.special.expit(
        zs_from_Rgamma(param.w, Rgamma(param.gamma, *forward_and_backward_sort_indices(X), param.loss_object.rank_type),
                       param.b, param.loss_object.rank_type, param.loss_object.N))


def generate_random_p(d, loss: ranking_logistic_loss = None):
    """
    generates a random Params/Params_with_loss object
    :param d: dimension
    :param loss: loss-object (optional)
    :return: Params or Params_with_loss object
    """
    if loss is None:
        dtype = torch.float64
    else:
        dtype = loss.dtype

    if loss.constraint_sum_gamma_k is None:
        gamma = torch.from_numpy(np.random.uniform(0, 1, d)).to(dtype=dtype)
    else:
        gamma_not_proj = torch.from_numpy(np.random.uniform(0, 1, d)).to(dtype=dtype)
        gamma = prox_sum_to_k(gamma_not_proj, loss.constraint_sum_gamma_k)

    p = Params(torch.from_numpy(np.random.randn(d)).to(dtype=dtype), gamma,
               torch.from_numpy(np.random.random(1)).to(dtype=dtype), torch.rand(d).to(dtype=dtype))
    if loss is not None:
        return params_with_loss_from_param(p, loss)
    else:
        return p


def params_with_loss_from_param(param, loss):
    return Params_With_Loss(param.w, param.gamma, param.b, param.gamma_dual, loss)


def zeros_like(param: Params_With_Loss):
    return Params_With_Loss(torch.zeros_like(param.w), torch.zeros_like(param.gamma), torch.zeros_like(param.b),
                            torch.zeros_like(param.gamma_dual), param.loss_object)


def norm_diff_params(param_1, param_2):
    return (params_with_loss_from_param(param_1, param_2.loss_object) - param_2).norm()
