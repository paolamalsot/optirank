import torch
import numpy as np
from utilities.optirank.ranking_multiplication import Rgamma, wTR, zs_from_Rgamma, forward_and_backward_sort_indices, \
    ranking_tensor, offset_for_types, R_normalization_methods, N, no_constraint_sum_gamma_k_normalization_methods, \
    constraint_sum_gamma_k_normalization_methods
from utilities.optirank.src.BCD.prox.prox_gradient_sum_gamma_fixed import prox_subgradient


class ranking_logistic_loss:
    """
    This class implements the loss to minimize
    """

    def __init__(self, lambda_gamma_1, lambda_gamma_2, lambda_w_1, lambda_w_2, lambda_P, X, y_np, dtype=torch.float64,
                 sample_weight="balanced", R_normalization=None, rank_type="min", constraint_sum_gamma_k=None,
                 constraint_per_gamma_k=None):
        """
        :param lambda_gamma_1:scalar (regularization parameter)
        :param lambda_gamma_2:scalar (regularization parameter)
        :param lambda_w_1: scalar (regularization parameter)
        :param lambda_w_2: scalar (regularization parameter)
        :param lambda_P: scalar (regularization parameter for push-penalty)
        :param X: numpy array matrix of expressions! nxd
        :param y_np: numpy array class 0-1 hot vector nxd
        :param dtype: usually torch.float32 or torch.float64 (more precise but more expensive (in time and memory))
        :param sample_weight: None or "balanced". In balanced mode, sample weights are reweighted by a class-dependent factor,
        such that each class bears equal weight in the loss.
        :param R_normalization: "d", "sqrt(d)", "k", "sqrt(k)" or None
        :param rank_type: "min", "max", "avg"
        :param constraint_sum_gamma_k: integer/None for settting constraint: sum_gamma = constraint_sum_gamma_k
        :param constraint_per_gamma_k: scalar/None for setting constraint sum_gamma/d = constraint_per_gamma_k
        """

        self.dtype = dtype
        self.X = X
        self.y_np = y_np
        self.y = torch.from_numpy(y_np).to(dtype=dtype)
        self.n = np.size(y_np)
        self.forward_sort_indices, self.backward_min_sort_indices, self.backward_max_sort_indices = forward_and_backward_sort_indices(
            self.X)
        self.d = self.X.shape[1]
        self.lambda_gamma_1 = lambda_gamma_1
        self.lambda_gamma_2 = lambda_gamma_2
        self.lambda_w_1 = lambda_w_1
        self.lambda_w_2 = lambda_w_2
        self.lambda_P = lambda_P
        self.rank_type = rank_type
        possible_rank_types = ["min", "max", "avg"]
        if self.rank_type not in possible_rank_types:
            raise ValueError("Wrong rank type!")
        if sample_weight is not None and sample_weight != "balanced":
            raise ValueError("Wrong value for sample weight")
        self.sample_weight = sample_weight
        if self.sample_weight is None:
            self.sample_weights = torch.ones_like(self.y, dtype=self.dtype) / self.n
        elif self.sample_weight == "balanced":
            n_1 = torch.count_nonzero(self.y == 1).to(dtype=self.dtype)
            n_0 = torch.count_nonzero(self.y == 0).to(dtype=self.dtype)
            self.sample_weights = torch.empty_like(self.y, dtype=self.dtype)
            self.sample_weights[self.y == 1.0] = (1.0 / (2 * n_1))
            self.sample_weights[self.y == 0.0] = (1.0 / (2 * n_0))

        # constraint_sum_gamma_k
        if (constraint_sum_gamma_k is not None):
            if (self.lambda_gamma_1 != 0) or (self.lambda_gamma_2 != 0):
                raise ValueError("with constraint_sum_gamma_k, reg. on gamma will have no effect!")
            if not (isinstance(constraint_sum_gamma_k, int)):
                raise ValueError("constraint_sum_gamma_k if not None, must be an int")
            if (constraint_sum_gamma_k < 0) or (constraint_sum_gamma_k > self.d):
                raise ValueError("constraint_sum_gamma_k if not None, must be an int comprised between 0 and d")
        self.constraint_sum_gamma_k_ = constraint_sum_gamma_k
        self.constraint_per_gamma_k = constraint_per_gamma_k
        self.R_normalization = R_normalization
        if self.R_normalization not in R_normalization_methods:
            raise ValueError("Wrong R_normalization method")
        elif self.constraint_sum_gamma_k is None and self.R_normalization not in no_constraint_sum_gamma_k_normalization_methods:
            raise ValueError("Normalization method {} must be accompagnied with a constraint_sum_gamma_k.".format(
                self.R_normalization))
        elif self.constraint_sum_gamma_k is not None and self.R_normalization not in constraint_sum_gamma_k_normalization_methods:
            raise ValueError("Normalization method {} cannot be accompagnied with a constraint_sum_gamma_k.".format(
                self.R_normalization))
        self.N = N(self.R_normalization, self.d, self.constraint_sum_gamma_k)

    @property
    def constraint_sum_gamma_k(self):
        if self.constraint_sum_gamma_k_ is None and self.constraint_per_gamma_k is None:
            return None
        else:
            if self.constraint_sum_gamma_k_ is not None and self.constraint_per_gamma_k is None:
                return self.constraint_sum_gamma_k_
            elif self.constraint_sum_gamma_k_ is None and self.constraint_per_gamma_k is not None:
                return int(self.constraint_per_gamma_k * self.d)
            else:
                if int(self.constraint_per_gamma_k * self.d) != self.constraint_sum_gamma_k_:
                    raise ValueError("Not compatible constraints sum_gamma_k and per_gamma_k")

    def to(self, dtype, copy=True, copy_data=True):
        """returns a copy and/or data-type conversed version of the loss object
        :param dtype: float32, float64 for dtype conversion
        :param: copy: boolean flag for returning a copy
        :param: copy_data: boolean flag for copying associated X and y tensors
        """
        if dtype not in [torch.float64, torch.float32]:
            raise ValueError("Wrong dtype")
        if copy:
            if copy_data:
                return ranking_logistic_loss(self.lambda_gamma_1, self.lambda_gamma_2, self.lambda_w_1, self.lambda_w_2,
                                             self.lambda_P, np.copy(self.X), np.copy(self.y_np), dtype,
                                             self.sample_weight, self.R_normalization, self.rank_type,
                                             self.constraint_sum_gamma_k)
            else:
                return ranking_logistic_loss(self.lambda_gamma_1, self.lambda_gamma_2, self.lambda_w_1, self.lambda_w_2,
                                             self.lambda_P, self.X, self.y_np, dtype,
                                             self.sample_weight, self.R_normalization, self.rank_type,
                                             self.constraint_sum_gamma_k)
        else:
            self.y = self.y.to(dtype=dtype)
            return self

    def Taylor_initialize_parameters(self):
        """
        initialize w, gamma, b as the optimal values of the objective function (wo loss, for the first order taylor expansion of the loss around w, gamma = 0, under the constraint w dot gamma = 1.
        We expand under b=0 and set b=0)
        :return: initialized tensors (w (size d), gamma (size d), b (scalar))
        """
        X_av = - np.sum(
            np.expand_dims(self.sample_weights.numpy() * (self.y.numpy() - 0.5), axis=(1, 2)) * ranking_tensor(self.X,
                                                                                                               self.rank_type),
            axis=0)
        u, s, vh = np.linalg.svd(X_av, full_matrices=True)
        w = - torch.from_numpy(u[:, 0]).to(dtype=self.dtype)
        gamma = torch.from_numpy(np.clip(vh[0, :], 0, 1)).to(dtype=self.dtype)
        b = torch.Tensor([0]).to(dtype=self.dtype)
        return w, gamma, b

    def loss_with_penalties(self, w, gamma, zs):
        return self.logloss(zs) + self.lambda_gamma_1 * torch.norm(gamma, 1) + self.lambda_gamma_2 * torch.norm(gamma,
                                                                                                                2) ** 2 + self.lambda_w_1 * torch.norm(
            w, 1) + self.lambda_w_2 * torch.norm(w, 2) ** 2 + self.lambda_P * torch.sum(gamma * (1 - gamma))

    def d2loss_with_penalties_dgamma2(self, w, zs):  # attention not surrogate loss!
        return self.d2logloss_dgamma2(w, zs) + 2 * (self.lambda_gamma_2 - self.lambda_P)

    # push-penalty and decomposition into 2 components LP1 and LP2

    def LP2(self, gamma, gamma_dual):
        return - self.lambda_P * torch.sum(gamma * gamma_dual)

    def LP1(self, gamma, gamma_dual):
        return self.lambda_P * torch.sum(((1 + gamma_dual) / 2) ** 2)

    def LP(self, gamma, gamma_dual):
        return self.LP1(gamma, gamma_dual) + self.LP2(gamma, gamma_dual)

    def surrogate_loss_with_penalties(self, w, gamma, zs, gamma_dual):
        return self.logloss(zs) + self.surrogate_penalties(w, gamma, gamma_dual)

    def surrogate_penalties(self, w, gamma, gamma_dual):
        return self.lambda_gamma_1 * torch.norm(gamma, 1) + self.lambda_gamma_2 * torch.norm(gamma,
                                                                                             2) ** 2 + self.lambda_w_1 * torch.norm(
            w, 1) + self.lambda_w_2 * torch.norm(w, 2) ** 2 + self.LP(gamma, gamma_dual)

    def surrogate_loss_prox_g(self, w, gamma, zs, gamma_dual):  # used for tests
        return self.logloss(zs) + self.LP2(gamma, gamma_dual)

    def surrogate_loss_prox_h(self, w, gamma, zs, gamma_dual):  # used for tests
        return self.lambda_gamma_1 * torch.norm(gamma, 1) + self.lambda_w_1 * torch.norm(w,
                                                                                         1) + self.lambda_gamma_2 * torch.norm(
            gamma, 2) ** 2 + self.lambda_w_2 * torch.norm(w, 2) ** 2 + self.LP1(gamma, gamma_dual)

    def Rgamma(self, gamma):
        return Rgamma(gamma, self.forward_sort_indices, self.backward_min_sort_indices, self.backward_max_sort_indices,
                      type=self.rank_type)

    def wTR(self, w):
        return wTR(w, self.forward_sort_indices, self.backward_min_sort_indices, self.backward_max_sort_indices,
                   type=self.rank_type)

    def zs(self, w, gamma, b):
        return zs_from_Rgamma(w, self.Rgamma(gamma), b, type=self.rank_type, N=self.N)

    def logloss(self, zs):
        """
        :param zs: torch array of size n with logistic regression scores
        :return: logistic loss
        """
        return - torch.sum((self.y * zs + torch.nn.functional.logsigmoid(-zs)) * self.sample_weights)  # scalar

    # derivatives and double derivatives
    def dlogloss_dz(self, zs):
        return (- (self.y - torch.sigmoid(zs)) * self.sample_weights)

    def d2logloss_dz2(self, zs):
        return ((torch.sigmoid(zs) * torch.sigmoid(-zs) * self.sample_weights))

    def d2logloss_db2(self, zs):
        return torch.sum(self.d2logloss_dz2(zs))

    def dzs_dw(self, Rgamma):
        return (Rgamma.T + offset_for_types[self.rank_type]) / self.N

    def dzs_dgamma(self, wTR):
        return (wTR).T / self.N

    def dlogloss_dw(self, Rgamma, zs):
        """
        :param Rgamma: size n x d
        :param zs: numpy size n vector with scores (for speed we don't re-calculate it!)
        :return: evalutation of dlogloss/dw (numpy size d vector)
        """

        return self.dzs_dw(Rgamma) @ self.dlogloss_dz(zs)  # d

    def dlogloss_dgamma(self, wTR, zs):
        """
        :param w: numpy size d vector
        :param zs: numpy size n vector with scores (for speed we don't re-calculate it!)
        :return: evalutation of dlogloss/dgamma (numpy size d vector)
        """
        return self.dzs_dgamma(wTR) @ self.dlogloss_dz(zs)  # d

    def dlogloss_db(self, zs):
        """
        :param zs: numpy size n vector with scores (for speed we don't re-calculate it!)
        :return: evalutation of dlogloss/db (numpy size d vector)
        """
        return torch.sum(self.dlogloss_dz(zs))  # 1

    # double derivatives (useful for initialization of stepsize)
    def d2logloss_dgamma2(self, w, zs):
        return (self.dzs_dgamma(self.wTR(w))) ** 2 @ self.d2logloss_dz2(zs)

    def d2logloss_dw2(self, Rgamma, zs):
        return (self.dzs_dw(Rgamma)) ** 2 @ self.d2logloss_dz2(zs)

    def hessian_w_logloss(self, Rgamma, zs):  # note it is quite expensive in RAM
        # return torch.sum((torch.sigmoid(zs)*torch.sigmoid(-zs) * self.sample_weights).view(-1,1,1) * torch.einsum('si,sj->sij', Rgamma, Rgamma), dim = 0)
        return torch.einsum('s,si,sj->ij', self.d2logloss_dz2(zs), self.dzs_dw(Rgamma).T, self.dzs_dw(Rgamma).T)

    def hessian_gamma_logloss(self, wTR, zs):
        return torch.einsum('s,si,sj->ij', self.d2logloss_dz2(zs), self.dzs_dgamma(wTR).T, self.dzs_dgamma(wTR).T)
        # return torch.sum((torch.sigmoid(zs)*torch.sigmoid(-zs) * self.sample_weights).view(-1,1,1) * torch.einsum('si,sj->sij', wTR, wTR), dim = 0)

    def dloss_dgamma_dual2(self, gamma_dual):
        return torch.ones_like(gamma_dual) * self.lambda_P

    def dLP2_dgamma_dual(self, gamma, gamma_dual):
        return - self.lambda_P * gamma  # d

    def dLP2_dgamma(self, gamma, gamma_dual):
        return - self.lambda_P * gamma_dual  # d

    def dLP1_dgamma_dual(self, gamma, gamma_dual):
        return self.lambda_P * ((1 + gamma_dual) / 2)

    # all the subgradient-functions return the left and right-sided limit of the subderivatives (https://en.wikipedia.org/wiki/Subderivative)
    # In the first dimension: coordinate
    # In the second min-max subdifferential

    def subgradient_minimal_norm_surrogate_loss_with_penalties_dw(self, dlogloss_dw, w):
        gradient = dlogloss_dw + 2 * self.lambda_w_2 * w + torch.sign(
            w) * self.lambda_w_1  # except in zero where torch.sign is zero!
        indices_0 = torch.where(w == 0)[0]
        gradient[indices_0] = gradient[indices_0] - torch.clamp(self.lambda_w_1 * torch.ones_like(gradient[indices_0]),
                                                                max=torch.abs(gradient[indices_0])) * \
                              torch.sign(gradient)[indices_0]
        return gradient

    def subgradient_surrogate_loss_with_penalties_dgamma(self, dlogloss_dgamma, gamma, gamma_dual):
        gradient_except_sum_gamma = dlogloss_dgamma + 2 * self.lambda_gamma_2 * gamma + self.dLP2_dgamma(gamma,
                                                                                                         gamma_dual) + self.lambda_gamma_1 * torch.sign(
            gamma)  # I put the l1 penalty, is it okay?
        return gradient_except_sum_gamma

    def subgradient_surrogate_loss_without_penalties_dgamma(self, dlogloss_dgamma, gamma):
        gradient_except_sum_gamma = dlogloss_dgamma + 2 * self.lambda_gamma_2 * gamma + self.lambda_gamma_1 * torch.sign(
            gamma)  # I put the l1 penalty, is it okay?
        return gradient_except_sum_gamma

    def projected_gradient_surrogate_loss_with_penalties_dgamma_constraint_sum_gamma(self, dlogloss_dgamma, gamma,
                                                                                     gamma_dual):
        gradient_except_sum_gamma = dlogloss_dgamma + 2 * self.lambda_gamma_2 * gamma + self.dLP2_dgamma(gamma,
                                                                                                         gamma_dual) + self.lambda_gamma_1  # I put the l1 penalty, is it okay?
        projected_gradient = prox_subgradient(gradient_except_sum_gamma, gamma == 0, gamma == 1)
        return projected_gradient

    def gradient_surrogate_loss_with_penalties_dgamma_dual(self, gamma, gamma_dual):
        gradient = self.dLP2_dgamma_dual(gamma, gamma_dual) + self.dLP1_dgamma_dual(gamma, gamma_dual)
        return gradient

    def y_pred(self, zs):
        return torch.sigmoid(zs) > 0.5

    def dloss_with_penalties_dgamma_in_0_1(self, wTR, zs, gamma):
        # What is the loss if all w, gamma except gamma_i that goes to 0 or 1 ?
        zs_in_0 = zs.view(-1, 1) - wTR * gamma / self.N
        dloss_in_0 = -1 * torch.sum(
            (self.y.view(-1, 1) - torch.sigmoid(zs_in_0)) * self.sample_weights.view(-1, 1) * wTR / self.N,
            dim=0) + self.lambda_P
        zs_in_1 = zs.view(-1, 1) + wTR * (1 - gamma) / self.N
        dloss_in_1 = -1 * torch.sum(
            (self.y.view(-1, 1) - torch.sigmoid(zs_in_1)) * self.sample_weights.view(-1, 1) * wTR / self.N,
            dim=0) + self.lambda_gamma_1 + 2 * self.lambda_gamma_2 - self.lambda_P
        return torch.cat([dloss_in_0.view(-1, 1), dloss_in_1.view(-1, 1)], 1)  # d x 2 tensor

    def to_lightweight(self, copy=True):
        if copy:
            return ranking_logistic_loss_lightweight(self.R_normalization, self.rank_type, self.constraint_sum_gamma_k,
                                                     self.N, self.lambda_w_1, self.lambda_w_2, self.lambda_gamma_1,
                                                     self.lambda_gamma_2, self.sample_weight)
        else:
            self = ranking_logistic_loss_lightweight(self.R_normalization, self.rank_type, self.constraint_sum_gamma_k,
                                                     self.N, self.lambda_w_1, self.lambda_w_2, self.lambda_gamma_1,
                                                     self.lambda_gamma_2,
                                                     self.sample_weight)  # dunno if it is allowed because we change the type! #TODO: remove because cannot work!!
            return self


class ranking_logistic_loss_lightweight():

    def __init__(self, normalization, rank_type, constraint_sum_gamma, N, lambda_w_1, lambda_w_2, lambda_gamma_1,
                 lambda_gamma_2, sample_weight):
        self.R_normalization = normalization
        self.rank_type = rank_type
        self.constraint_sum_gamma = constraint_sum_gamma
        self.N = N
        self.lambda_w_1 = lambda_w_1
        self.lambda_w_2 = lambda_w_2
        self.lambda_gamma_1 = lambda_gamma_1
        self.lambda_gamma_2 = lambda_gamma_2
        self.sample_weight = sample_weight


def loss_object_with_different_data(loss: ranking_logistic_loss, X, y):
    """
    returns a new loss object with different data (but otherwise same parameters)
    :param loss: loss_object
    :param X: nxd numpy array
    :param y: n numpy array
    :return:
    """
    return ranking_logistic_loss(loss.lambda_gamma_1, loss.lambda_gamma_2, loss.lambda_w_1, loss.lambda_w_2,
                                 loss.lambda_P, X, y, dtype=loss.dtype, sample_weight=loss.sample_weight,
                                 R_normalization=loss.R_normalization, rank_type=loss.rank_type,
                                 constraint_sum_gamma_k=loss.constraint_sum_gamma_k)


def minimal_norm_subgradient(subgradient_range):
    min = subgradient_range[:, 0]
    max = subgradient_range[:, 1]
    # returns the minimal norm of possible subderivatives
    return torch.norm(torch.minimum(torch.maximum(min, torch.zeros_like(min)), max))


def push_penalty_per_lambda_P(gamma):
    return torch.sum(gamma * (1 - gamma))
