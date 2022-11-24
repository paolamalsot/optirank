# Different strategies to determine successive lambda_P values to use during relaxation.

from abc import ABC, abstractmethod
from utilities.optirank.classifiers.classifiers_helper import optimize_w_b_only
from utilities.optirank.src.BCD.prox.prox_operators import gamma_dual_opt
from utilities.optirank.src.loss.loss import push_penalty_per_lambda_P
from utilities.optirank.src.relaxation.lambda_P_setting_helper import *

boundary_selection_methods = ["closest_boundary", "smallest_lambda_P"]


def lambda_P_additional_partial_derivative_on_boundaries_coordinatewise_hyp_sol(gamma, dloss_dgamma_01,
                                                                                boundary_selection_method):
    """
    For each gamma entry:
    - Looks at the corresponding 0 or 1 value, which we call trapping value.
    In closest boundary, the trapping value is the nearest 0/1 integer.
    In smallest lambda_P, this correponds to the integer needing the smallest additional lambda_P(see later).
    - Calculates the additional lambda_P value needed such that if gamma_i was set to the trapping value (with others values remaining the same), it would be a local minimum (by combination of gradient and domain boundaries).
    :param gamma: values for gamma torch array
    :param dloss_dgamma_01: results for dloss_dgamma_01 on p
    :param boundary_selection_method: either "closest_boundary" or "smallest lambda_P".
    :return: torch tensor of same size as gamma with values for lambda_P required to put the corresponding coordinate to zero or one. 
    When negative, it means that a more negative lambda_P would have sufficed.
    """

    # Note that it could be possible to consider the constraint sum_gamma = k, but might not be realistic because we don't consider a point on the hyper-simplex.
    if boundary_selection_method not in boundary_selection_methods:
        raise ValueError("Wrong boundary selection method:{}. Possibilities: {}".format(boundary_selection_method,
                                                                                        boundary_selection_methods))

    lambda_Ps = torch.zeros_like(gamma)
    already_set = torch.logical_or(gamma == 0, gamma == 1)
    lambda_Ps[already_set] = 0

    if boundary_selection_method == "closest_boundary":
        indices_prone_for_zero = torch.nonzero(gamma < 0.5).flatten()
        lambda_Ps[indices_prone_for_zero] = - dloss_dgamma_01[indices_prone_for_zero, 0]
        indices_prone_for_one = torch.nonzero(gamma >= 0.5).flatten()
        lambda_Ps[indices_prone_for_one] = dloss_dgamma_01[indices_prone_for_one, 1]
        coordinate_wise_hypothetical_gamma = torch.zeros_like(lambda_Ps)
        coordinate_wise_hypothetical_gamma[indices_prone_for_one] = 1
    elif boundary_selection_method == "smallest_lambda_P":
        possible_lambda_Ps = torch.hstack((- dloss_dgamma_01[:, 0].view(-1, 1), dloss_dgamma_01[:, 1].view(-1, 1)))
        lambda_Ps, coordinate_wise_hypothetical_gamma = torch.min(possible_lambda_Ps,
                                                                  dim=1)  # note this can be negative!
    else:
        raise ValueError("Wrong boundary selection method")

    return lambda_Ps, coordinate_wise_hypothetical_gamma


def lambda_P_additional_delta_loss(sol: Params_With_Loss, M, delta_lambda_min, epsilon):
    """
    returns the maximum delta_lambda_P such that the F_{lambda_P}(sol) -  F_{lambda_P + delta_lambda_P}(sol) is less than M * epsilon.
    :param sol: solution param_with_loss
    :param M: ratio value
    :param delta_lambda_min: minimal delta_lambda
    :param epsilon: should correspond to BCD stopping criterion.
    :return: delta_lambda
    """

    loss_per_lambda_P = push_penalty_per_lambda_P(sol.gamma).item()
    if loss_per_lambda_P > 0:
        delta_lambda = (M * epsilon) / loss_per_lambda_P
    else:
        delta_lambda = delta_lambda_min
    delta_lambda = max(delta_lambda_min, delta_lambda)

    return delta_lambda


def lambda_P_additional_partial_derivative_on_boundaries_blockwise_hyp_sol(sol: Params_With_Loss, t=0.1, q=0.1,
                                                                           w_optimization=False,
                                                                           projected_partial_derivative=True):
    """
    Variant on lambda_P_additional_partial_derivative_on_boundaries_coordinatewise_hyp_sol:
    - First a hypothetical solution is calculating by rounding certain gammas to 0/1:
        - At least per_remaining of non 0/1 entries are rounded
        - All entries within t distance of the nearest 0/1 are rounded.
    - If w_optimization flag is on, w of hypothetical solution is optimized.
    - Calculate lambda_P additional needed to "trap" the rounded value.
    :param sol: solution param_with_loss
    :param t: all gammas falling inside [0;t[ U ]1-t;t] are blocked (in theory)
    :param q: per_remaining of gammas falling inside ]0;1[ are blocked (in theory)
    :param w_optimization: if true, w is optimized before projected gradient calculation with gamma blocked rounded to their nearest value.
    :param projected_partial_derivative if true, the partial derivative is calculated taking into account the sum-gamma constraint, if needed.
    :return: lambda_P additional (in theory)
    """

    logging.debug("Lambda_P additional details")

    sol_round = sol.to(dtype=sol.loss_object.dtype, copy=True)
    indices_round = get_indices_round(sol, q, t)
    constraint_sum_gamma = sol_round.loss_object.constraint_sum_gamma_k is not None

    if constraint_sum_gamma:
        sol_round.gamma = round_and_scale_gamma(sol_round.gamma, indices_round,
                                                sol_round.loss_object.constraint_sum_gamma_k)
    else:  # if no constraint, no need to scale
        sol_round.gamma[indices_round] = torch.round(sol_round.gamma[indices_round])

    sol_round.gamma_dual = gamma_dual_opt(sol_round)

    if w_optimization:
        logging.debug("w optimization")
        sol_round, converged = optimize_w_b_only(sol_round)

    corresponding_border = torch.round(sol_round.gamma).to(dtype=torch.int64)  # 1 if 1 0 otherwise!

    if projected_partial_derivative and constraint_sum_gamma:
        k = sol.loss_object.constraint_sum_gamma_k
        d = sol.loss_object.d
        dloss_dgamma = sol_round.loss_object.subgradient_surrogate_loss_without_penalties_dgamma(
            sol_round.dlogloss_dgamma(), sol_round.gamma)
        ones_unit = torch.ones(d, dtype=torch.double) / (d ** 0.5)
        dloss_dgamma_proj = (dloss_dgamma - torch.dot(dloss_dgamma, ones_unit) * ones_unit)
        lambda_Ps_I0 = - dloss_dgamma_proj / (2 * k / d)
        lambda_Ps_I1 = dloss_dgamma_proj / (2 * (1 - k / d))
        lambda_Ps = torch.from_numpy(np.choose(corresponding_border.numpy(),
                                               [lambda_Ps_I0.numpy(), lambda_Ps_I1.numpy()])) - sol.loss_object.lambda_P
        # in order to really have lambda_P_additional
    else:
        dloss_dgamma = sol_round.loss_object.subgradient_surrogate_loss_with_penalties_dgamma(
            sol_round.dlogloss_dgamma(), sol_round.gamma, sol_round.gamma_dual)

        sign_reversal_op = torch.ones_like(sol_round.gamma)
        sign_reversal_op[corresponding_border == 0] = -1
        lambda_Ps = dloss_dgamma * sign_reversal_op

    lambda_P_sup = max(lambda_Ps[indices_round]).item()

    block_wise_hypothetical_sol = sol_round

    return lambda_P_sup, block_wise_hypothetical_sol, indices_round


def lambda_P_additional_displacing_quadratic_minimum(sol: Params_With_Loss):
    """
    Based on a coordinate-wise quadratic approximation of the loss, calculates the additional lambda_P needed to
    displace the quadratic minimum (by hypothesis) to the closest boundary.
    :param sol: solution param_with_loss
    :return: torch array (of size d) with lambda_P additional (in theory) needed to displace the minimum!
    """
    h = sol.loss_object.d2loss_with_penalties_dgamma2(sol.w, sol.zs())
    lambda_P_possibilities = h * dist_to_border(
        sol.gamma).flatten()  # no absolute value because it does not make sense if the double derivative is negative
    return lambda_P_possibilities


class lambda_P_setting_strategy(ABC):
    """
    Abstract class for implementing a setting lambda_P strategy.
    """

    def __init__(self, with_interpolation=False, n_interpolation_steps=5):
        """
        :param with_interpolation: Boolean flag indication wether to test interpolating lambda_P values between previous lambda_P and subsequent lambda_P.
        :param n_interpolation_steps: integer with number of interpolating steps.
        """
        self.with_interpolation = with_interpolation
        self.n_interpolation_steps = n_interpolation_steps
        self.iteration = 0

    # from https://stackoverflow.com/questions/44576167/force-child-class-to-override-parents-methods
    @abstractmethod
    def lambda_P_sup(self, sol, lambda_P_old, iteration, epsilon_max):
        pass

    def next_lambda_P(self, sol, lambda_P_old, epsilon):
        """
        Calculates the next lambda_P.
        :param sol: previous solution (Params_With_Loss object)
        :param lambda_P_old: previous lambda_P
        :param epsilon: scalar (should correspond to the epsilon of the stopping criterion of BCD)
        :return: next lambda_P, flag for negativity of lambda_P
        """
        negative_lambda_P_flag = False
        if (not self.with_interpolation) or (
                self.with_interpolation and self.iteration % self.n_interpolation_steps == 0):
            lambda_P_sup = self.lambda_P_sup(sol, lambda_P_old, self.iteration, epsilon)
            if lambda_P_sup <= 0:  # arbitrary rule
                logging.info("lambda_P_additional was <0")
                lambda_P_sup = 2 * (lambda_P_old + 10 ** (-12))
                negative_lambda_P_flag = True
            if self.with_interpolation:
                # returns n_interpolation_steps lambda_Ps between lambda_P_old (not included) to lambda_P_old + lambda_P_sup (included)
                self.lambda_Ps_interpolation = np.linspace(lambda_P_old, lambda_P_old + lambda_P_sup,
                                                           self.n_interpolation_steps + 1)[1:]
                lambda_P = self.lambda_Ps_interpolation[0]
            else:
                lambda_P = lambda_P_old + lambda_P_sup

        else:  # with_interpolation and iteration%self.n_interpolation_steps != 0
            lambda_P = self.lambda_Ps_interpolation[self.iteration % self.n_interpolation_steps]

        self.iteration += 1
        return lambda_P, negative_lambda_P_flag

    def initialize(self, lambda_P_init):
        if lambda_P_init is None:
            lambda_P = 0
        else:
            lambda_P = lambda_P_init
        return lambda_P


class partial_derivative_on_boundaries_blockwise_hyp_sol(lambda_P_setting_strategy):
    """
    Calculates lambda_P needed to block block-wise hypothetical solution.
    (The advantage of the blockwise solution is that it respects the constraint sum_gamma, so it is more "realistic".)
    (see lambda_P_additional_partial_derivative_on_boundaries_blockwise_hyp_sol documentation)
    """

    def __init__(self, t=0.1, q=0.05, projected_partial_derivative=True, w_optimization=False, with_interpolation=False,
                 n_interpolation_steps=5):
        self.t = t
        self.q = q
        self.projected_partial_derivative = projected_partial_derivative
        self.w_optimization = w_optimization
        super().__init__(with_interpolation=with_interpolation, n_interpolation_steps=n_interpolation_steps)

    def lambda_P_sup(self, sol: Params_With_Loss, lambda_P_old, iteration, _):
        lambda_P_sup, _, _ = lambda_P_additional_partial_derivative_on_boundaries_blockwise_hyp_sol(sol, t=self.t,
                                                                                                    q=self.q,
                                                                                                    projected_partial_derivative=self.projected_partial_derivative,
                                                                                                    w_optimization=self.w_optimization)
        return lambda_P_sup


class delta_loss(lambda_P_setting_strategy):
    """
    Calculates lambda_P needed to increase the loss by a predifined epsilon value.
    (see its documentation of lambda_P_additional_delta_loss)
    """

    def __init__(self, M=1, delta_lambda_min=10 ** (-5), with_interpolation=False):
        if M < 1:
            raise ValueError("M-parameter must be bigger than 1.")
        self.M = M
        self.delta_lambda_min = delta_lambda_min
        if with_interpolation:
            raise ValueError("makes no sense")
        super().__init__(with_interpolation=with_interpolation, n_interpolation_steps=0)

    def lambda_P_sup(self, sol: Params_With_Loss, lambda_P_old, iteration, epsilon):
        lambda_P_sup = lambda_P_additional_delta_loss(sol, self.M, self.delta_lambda_min, epsilon)
        return lambda_P_sup


class partial_derivative_on_boundaries_coordinatewise_hyp_sol(lambda_P_setting_strategy):
    """
    For each coordinate, calculates the lambda_P needed to block its rounded version.
    (see documentation of lambda_P_additional_partial_derivative_on_boundaries_coordinatewise_hyp_sol).
    """

    def __init__(self, q=1.0, boundary_selection_method="closest_border", with_interpolation=False,
                 n_interpolation_steps=5):
        self.q = q
        self.boundary_selection_method = boundary_selection_method
        super().__init__(with_interpolation=with_interpolation, n_interpolation_steps=n_interpolation_steps)

    def lambda_P_sup(self, sol, lambda_P_old, iteration, _):
        lambda_Ps, _ = lambda_P_additional_partial_derivative_on_boundaries_coordinatewise_hyp_sol(sol.gamma,
                                                                                                   sol.dloss_with_penalties_dgamma_in_0_1(),
                                                                                                   boundary_selection_method=self.boundary_selection_method)
        lambda_P_sup = get_smallest_lambda_P_to_block_q_i_r(lambda_Ps, sol, self.q)
        return lambda_P_sup


class displace_quadratic_minimum(lambda_P_setting_strategy):
    """
    Strategy with double derivative.
    Starting with a quadratic approximation of the loss, the idea is to calculate the lambda_P additional needed
    to displace the local minima falling in ]0,1[ to the closest boundary.
    (see documentation of lambda_P_additional_displacing_quadratic_minimum)"""

    def __init__(self, q=0.05, with_interpolation=False, n_interpolation_steps=5):
        self.q = q
        super().__init__(with_interpolation=with_interpolation, n_interpolation_steps=n_interpolation_steps)

    def lambda_P_sup(self, sol: Params_With_Loss, lambda_P_old, iteration, _):
        lambda_Ps = lambda_P_additional_displacing_quadratic_minimum(sol)
        lambda_P_sup = get_smallest_lambda_P_to_block_q_i_r(lambda_Ps, sol, self.q)
        return lambda_P_sup


def get_smallest_lambda_P_to_block_q_i_r(lambda_Ps, sol: Params_With_Loss, q):
    """ Returns the smallest lambda_P_sup such that q percent of I_r are blocked.
    :params: lambda_Ps: torch array of length d with the lambda_P needed to block the corresponding gamma.
    :params: solution
    :params: q percentage"""
    indices_remaining = torch.nonzero(torch.logical_and(sol.gamma != 0, sol.gamma != 1))
    n_block = int(max(1, q * len(indices_remaining)))
    sorted, indices = torch.sort(lambda_Ps[indices_remaining])
    lambda_P_sup = sorted[n_block - 1].item()
    return lambda_P_sup


def get_smallest_lambda_P_to_block_q_tot(lambda_Ps, q):
    """
    Returns the smallest lambda_P_sup such that q percent of gammas are blocked (irrespective if they belong to I_r..).
    :params: lambda_Ps: torch array of length d with the lambda_P needed to block the corresponding gamma.
    :params: q percentage
    """
    return np.quantile(lambda_Ps, q, interpolation='nearest')
