# implements different search methods for inverse stepsizes in prox-step.

from utilities.optirank.src.BCD.initialization import Initialization_L
from utilities.optirank.src.BCD.BCD_units.Prox_Steps import Prox_Step
from utilities.optirank.src.loss.params import Params_With_Loss
from utilities.optirank.src.BCD.prox.prox_operators import update_L
import logging

class Search_Method():
    """Implements "linesearch" method to find the appropriate inverse stepsize (L) in the update step.
     Ls are searched in the range L_start * eta^n with varying n. The search method in n_space and acceptance criterion
     depends on the "search method". Once L has been found, the next param p is calculated."""

    def __init__(self, search_method_name, L_min, n_min, L_start, n_max, eta, init_L_var_name, init_L_method, with_diagnostics = False):
        """
        :param search_method_name: either "first_best", "bidirectional", "bidirectional_delta_loss", "bidirectional_delta_loss_wo_argcrit"
        :param L_min: minimum inverse stepsize (defines maximum stepsize)
        :param n_min: minimum n (if -1, a larger step than previous one will be searched)
        :param L_start: "previous" or "min". If previous, the search for L starts from previous_value, else from L_min
        :param n_max: maximum n at which the linesearch terminates.
        :param eta: scalar
        :param init_L_var_name: either "w", "gamma", or "b"
        :param init_L_method: either "L_min", "hessian", "hessian_proj"
        :param with_diagnostics: boolean flag for recording diagnostics.
        """
        self.search_method_name = search_method_name
        self.L_min = L_min
        self.n_min = n_min
        self.L_start = L_start
        self.n_max = n_max
        self.eta = eta
        self.init_L_var_name = init_L_var_name
        self.init_L_method = init_L_method
        self.L = None
        self.line_search_method = search_methods_dict[search_method_name]
        self.with_diagnostics = with_diagnostics
        self.initialize_diagnostics()

    def initialize_diagnostics(self):
        self.diagnostics = {"L": [], "n": []}

    def get_L_start(self, p):
        """initializes the inverse stepsize"""
        if self.L_start == "min":
            L = self.L_min
        elif self.L_start == "previous":
            if self.L is None:
                init_L_method = Initialization_L(self.init_L_method)
                L = init_L_method.initialize_L(self.L_min, p, self.init_L_var_name)
            else:
                L =  self.L
        else:
            raise ValueError("Wrong L_start name. Either previous or min")
        return L

    def run(self, prox_step: Prox_Step, p: Params_With_Loss, arg_criterion):

        L_start = self.get_L_start(p)
        self.L, (p, dp), self.n = self.line_search_method(prox_step, p, arg_criterion, self.n_min, self.n_max, self.L_min, L_start, self.eta)
        if self.with_diagnostics:
            self.append_to_diagnostics()
        return p, dp

    def reset(self):
        self.L = None

    def append_to_diagnostics(self):
        self.diagnostics["L"].append(self.L)
        self.diagnostics["n"].append(self.n)

### DIFFERENT SEARCH METHODS ###

def search_first_best(prox_step: Prox_Step, p_input: Params_With_Loss, arg_criterion, n_min, n_max, L_min, L_start, eta):
    """Starting from L_start * eta **n_min, increases n until arg_crit<=0 is met."""
    n = n_min
    while True:
        try:
            L = update_L(L_min, L_start, eta, n)
            # updating parameters
            p, dp = prox_step.step(p_input,L)
            if arg_criterion(p_input, p, dp, L) <= 0:
                break
            n += 1
            if n > n_max:
                raise ValueError("N has is bigger than the limit:{}".format(n_max))
        except OverflowError: #TODO: works only in the case of not delta
            logging.info("Catching overflow error: L_before: {}, |dp|_before: {}".format(L, (p-p_input).norm()))
            p = p_input
            dp = None
            L = L_start
            break
    return L, (p, dp), n


def increment_n(n_start, direction, n_max):
    n =  n_start + direction
    if n > n_max:
        raise ValueError("N has is bigger than the limit:{}".format(n_max))
    return n

def update_for_n(prox_step: Prox_Step, p_input: Params_With_Loss, arg_criterion, n_new, L_min, L_start, eta, with_loss = False):
    L_new = update_L(L_min, L_start, eta, n_new)
    (p_new, dp_new) = prox_step.step(p_input, L_new)
    arg_crit_new = arg_criterion(p_input, p_new, dp_new, L_new)
    if with_loss:
        loss_new = p_new.surrogate_loss_with_penalties()
    else:
        loss_new = None
    return L_new, (p_new, dp_new), arg_crit_new, loss_new


def bidirectional_delta_loss(prox_step: Prox_Step, p_input: Params_With_Loss, arg_criterion, n_min, n_max, L_min, L_start, eta):
    """
        The idea is to do a sufficiently small step such that argcrit <= 0
        Once we have found this step_size, try to make it bigger/smaller such that the delta_loss increases (while arg_crit remains <0)
    """

    L, (p, dp), n = search_first_best(prox_step, p_input, arg_criterion, 0, n_max, L_min, L_start, eta)
    loss = p.surrogate_loss_with_penalties()

    direction_set = False
    if n >= 1:
        direction_set = True
        direction = 1 #we will only search in the +1 direction
    else:
        direction = -1

    while(True):
        n_new = increment_n(n, direction, n_max)
        L_new, (p_new, dp_new), arg_crit_new, loss_new = update_for_n(prox_step, p_input, arg_criterion, n_new, L_min, L_start, eta, with_loss=True)

        if (loss_new > loss) or (arg_crit_new >= 0):
            if direction_set:
                break #we keep the old L, p, dp, n
            else:
                direction = - direction
                direction_set = True
                continue #we don't update L, p, dp, n

        L, p, dp, n, loss = L_new, p_new, dp_new, n_new, loss_new

    return L, (p, dp), n


def search_bidirectional(prox_step: Prox_Step, p_input: Params_With_Loss, arg_criterion, n_min, n_max, L_min, L_start, eta):

    """ Wierd, but seems to work in practice...
        The idea is to select a step_size such that arg_crit < 0 in the following way:
        If the step size had to be decreased from L_start (note that n_min has no effect...) to fulfill arg_crit < 0,
        then we take the biggest stepsize, i.e the first one that meets the criterion.
        On the other hand, if the step size could be increased from L_start while meeting the criterion arg_crit < 0,
        then augment the stepsize until the argcriterion start to increase...
        :param L_start = "previous"
        :param n_min = "-1"
    """

    direction_set = False
    direction = -1
    n = 0

    while True:

        n_new = increment_n(n, direction, n_max)

        L_new, (p_new, dp_new), arg_crit_new, _ = update_for_n(prox_step, p_input, arg_criterion, n_new, L_min,
                                                                      L_start, eta, with_loss=False)

        if (direction == +1) and direction_set and (arg_crit_new < 0):
            L, p, dp, n, arg_crit = L_new, p_new, dp_new, n_new, arg_crit_new
            break #finally met the criterion
        elif (direction == -1) and direction_set and (arg_crit_new >= arg_crit):
            break
        elif (direction == -1) and not(direction_set) and (arg_crit_new >= 0):
            direction = 1
            direction_set = True
        elif (direction == -1) and not(direction_set) and (arg_crit_new < 0):
            direction = -1
            direction_set = True
        elif ((direction == +1) and not(direction_set)):
            raise ValueError("Line Search should not end up here..")

        L, p, dp, n, arg_crit = L_new, p_new, dp_new, n_new, arg_crit_new

    return L, (p, dp), n



def bidirectional_delta_loss_wo_argcrit(prox_step: Prox_Step, p_input: Params_With_Loss, arg_criterion, n_min, n_max, L_min, L_start, eta):
    """
        The idea is to find the stepsize with the greatest decrease in loss. To achieve this:
        1. First we determine the direction (n < 0 or n>0) with the greatest decrease in loss.
        2. Then we increment n in this direction until the "decrease in loss" starts to increase.
        NB: In this scheme, the criterion arg_crit < 0 is not necessarily fulfilled.

        To achieve 1., we simply compare the decrease in loss between -1 and 0
    """

    L_0, (p_0, dp_0), arg_crit_0, loss_0 = update_for_n(prox_step, p_input, arg_criterion, 0, L_min,
                                                           L_start, eta, with_loss=True)
    L_minus_1, (p_minus_1, dp_minus_1), arg_crit_minus_1, loss_minus_1 = update_for_n(prox_step, p_input, arg_criterion, -1, L_min,
                                                        L_start, eta, with_loss=True)
    direction = 1 if loss_0 < loss_minus_1 else -1
    if direction == -1:
        L, (p, dp), arg_crit, loss = L_minus_1, (p_minus_1, dp_minus_1), arg_crit_minus_1, loss_minus_1
        n = -1
    else:
        L, (p, dp), arg_crit, loss = L_0, (p_0, dp_0), arg_crit_0, loss_0
        n = 0

    while True:
        n_new = increment_n(n, direction, n_max)
        L_new, (p_new, dp_new), arg_crit_new, loss_new = update_for_n(prox_step, p_input, arg_criterion, n_new, L_min,
                                                               L_start, eta, with_loss=True)
        if loss_new > loss:
            break
        L, (p, dp), arg_crit, loss, n = L_new, (p_new, dp_new), arg_crit_new, loss_new, n_new
    return L, (p, dp), n



search_methods_dict = {"first_best": search_first_best,
                       "bidirectional": search_bidirectional,
                       "bidirectional_delta_loss": bidirectional_delta_loss,
                       "bidirectional_delta_loss_wo_argcrit": bidirectional_delta_loss_wo_argcrit}