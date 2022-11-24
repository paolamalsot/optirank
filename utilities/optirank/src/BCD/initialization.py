import torch
from sklearn.linear_model import LogisticRegression
import numpy as np
import logging
from utilities.optirank.src.loss.params import Params_With_Loss
from utilities.optirank.src.relaxation.lambda_P_setting_helper import scale_gamma
from utilities.optirank.src.loss.loss import ranking_logistic_loss


class InitializazionParameters():
    """
    represent an initialization method for the parameters/variables w, gamma, b
    """

    def __init__(self, name, loss_object: ranking_logistic_loss, epsilon=10 ** (-5)):
        """
        :param name: initialization method name ("Taylor", "gamma_ones_w_random", "gamma_05_w_0", "gamma_ones_w_opt")
        :param loss_object: ranking_logistic_loss object
        :param epsilon: epsilon scalar (has an effect for gamma_ones_w_opt method)
        """
        self.name = name
        self.loss_object = loss_object
        if self.name not in ["Taylor", "gamma_ones_w_random", "gamma_05_w_0", "gamma_ones_w_opt"]:
            raise ValueError("wrong init_method")
        self.epsilon = epsilon

    def initialize_parameters(self):

        d = self.loss_object.d
        normalization_factor_ = self.loss_object.N
        dtype_ = self.loss_object.dtype
        if self.name == "Taylor":
            w, gamma, b = self.loss_object.Taylor_initialize_parameters()
            if self.loss_object.constraint_sum_gamma_k is not None:
                gamma = scale_gamma(gamma, self.loss_object.constraint_sum_gamma_k)
            gamma_dual = 2 * gamma - 1
        elif self.name == "gamma_ones_w_random":
            w = torch.randn(d, dtype=dtype_) / normalization_factor_  # random w
            gamma = torch.ones_like(w, dtype=dtype_)
            if self.loss_object.constraint_sum_gamma_k is not None:
                gamma = scale_gamma(gamma, self.loss_object.constraint_sum_gamma_k)
            b = torch.Tensor([0]).to(dtype=dtype_)
            gamma_dual = 2 * gamma - 1
        elif self.name == "gamma_05_w_0":
            gamma = torch.ones(d, dtype=dtype_) * 0.5
            if self.loss_object.constraint_sum_gamma_k is not None:
                gamma = scale_gamma(gamma, self.loss_object.constraint_sum_gamma_k)
            w = torch.zeros(d, dtype=dtype_)
            b = torch.Tensor([0]).to(dtype=dtype_)
            gamma_dual = 2 * gamma - 1
        elif self.name == "gamma_ones_w_opt":
            gamma = torch.ones(self.loss_object.d, dtype=dtype_)
            if self.loss_object.constraint_sum_gamma_k is not None:
                gamma = scale_gamma(gamma, self.loss_object.constraint_sum_gamma_k)
            if self.loss_object.lambda_w_1 != 0 or self.loss_object.lambda_w_2 != 0:
                l1_ratio = self.loss_object.lambda_w_1 / (2 * self.loss_object.lambda_w_2 + self.loss_object.lambda_w_1)
                C = 1 / (2 * self.loss_object.lambda_w_2 + self.loss_object.lambda_w_1)
                classifier = LogisticRegression(penalty='elasticnet', dual=False, tol=self.epsilon, C=C, solver='saga',
                                                random_state=None, max_iter=500,
                                                multi_class='auto', verbose=0, warm_start=False, n_jobs=None,
                                                l1_ratio=l1_ratio)
            else:
                classifier = LogisticRegression(penalty='none', dual=False, tol=self.epsilon,
                                                solver='lbfgs',
                                                random_state=None, max_iter=500,
                                                multi_class='auto', verbose=0, warm_start=False, n_jobs=-1)
            classifier.fit(self.loss_object.Rgamma(gamma).cpu(), 2 * self.loss_object.y.cpu() - 1)
            w = torch.from_numpy(np.squeeze(classifier.coef_)).to(dtype=dtype_)
            b = torch.from_numpy(np.squeeze(classifier.intercept_)).to(dtype=dtype_)
            gamma_dual = (2 * gamma - 1)

        return Params_With_Loss(w, gamma, b, gamma_dual, self.loss_object)


class Initialization_L():
    """
    represents an initialization method for the stepsize used in BCD
    """

    def __init__(self, name):
        """
        :param string name: method name ("L_min", "hessian", "hessian_proj"). With "L_min", L is initialized with
          the minimal L allowed in BCD. With "hessian", L is initialized with the the double derivative along the
          gradient of the loss. With "hessian_proj", L is initialized with the double derivative along the gradient
          projected on the constraint sum_gamma = k (if the constraint is active).
        """
        self.name = name
        if not (self.name in ["L_min", "hessian", "hessian_proj"]):
            raise ValueError("Wrong name for init_L method")

    def initialize_L(self, L_min, p: Params_With_Loss, var_name):
        """
        :param float L_min: minimal inverse stepsize
        :param Params_With_Loss p: initial parameter object.
        :param string var_name: either "w", "gamma", or "b". Variable concerned in the block coordinate descent. The
          gradient and the hessian will be calculated wrt this variable.
        :return float: initialized L
        """
        # same strategies for different variables
        if var_name == "w":
            hessian = p.loss_object.hessian_w_logloss(p.Rgamma, p.zs())
            derivative = p.dlogloss_dw()
            derivative_proj = derivative
        elif var_name == "gamma":
            hessian = p.loss_object.hessian_gamma_logloss(p.wTR, p.zs())
            derivative = p.dlogloss_dgamma() + p.loss_object.dLP2_dgamma(p.gamma, p.gamma_dual)
            derivative_proj = p.gradient_surrogate_loss_on_gamma()
        elif var_name == "b":
            hessian = p.loss_object.d2logloss_db2(p.zs()).reshape(1, -1)
            derivative = p.dlogloss_db().reshape(1, -1)
            derivative_proj = derivative
        else:
            raise ValueError("Wrong var_name")

        if self.name == "L_min":
            L = L_min
        elif self.name == "hessian":
            if torch.all(derivative == 0):
                logging.debug("derivative for {} is 0. L_{} initialized to L_min".format(var_name, var_name))
                L = L_min
            else:
                L = (derivative.T @ hessian @ derivative) / torch.norm(derivative) ** 2
                L = max(L.item(), L_min)
        elif self.name == "hessian_proj":
            if torch.all(derivative_proj == 0):
                logging.debug("derivative for {} is 0. L_{} initialized to L_min".format(var_name, var_name))
                L = L_min
            else:
                L = (derivative_proj.T @ hessian @ derivative_proj) / torch.norm(derivative_proj) ** 2
                L = max(L.item(), L_min)
        else:
            raise ValueError("Wrong initialize_L name")

        if np.isnan(L):
            logging.debug("L_{} was initialized to nan -> replaced by L_min".format(var_name))
            L = L_min

        return L
