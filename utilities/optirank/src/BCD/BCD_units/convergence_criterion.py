#different variants of convergence criteria

from utilities.optirank.src.loss.params import Params_With_Loss
from utilities.optirank.sparse_bilinear_BCD_funs import differences_in_logloss, diff_abs_norm_delta, difference_L_P
from abc import ABC, abstractmethod
import copy
import torch

def differences_in_surrogate_loss_with_penalties(dp:Params_With_Loss, p_f:Params_With_Loss, p_b:Params_With_Loss): #TODO: remove
    return differences_in_logloss(dp, p_f, p_b) + p_b.loss_object.lambda_w_1 * diff_abs_norm_delta(p_b.w, dp.w) + p_b.loss_object.lambda_gamma_1 * diff_abs_norm_delta(p_b.gamma, dp.gamma) + p_b.loss_object.lambda_w_2 * torch.dot(dp.w, p_b.w + p_f.w) + p_b.loss_object.lambda_gamma_2 * torch.dot(dp.gamma, p_b.gamma + p_f.gamma) + difference_L_P(p_b, p_f, dp)

all_variables_set = {"w", "gamma", "b", "gamma_dual"}
norm_gradient_for_variables_set = {"w": lambda p: torch.norm(p.subgradient_minimal_norm_surrogate_loss_on_w()),
                              "gamma": lambda p: torch.norm(p.gradient_surrogate_loss_on_gamma()),
                              "b": lambda p: torch.norm(p.dlogloss_db()),
                              "gamma_dual": lambda p: torch.norm(p.gradient_surrogate_loss_with_penalties_dgamma_dual())}

def make_convergence_criterion(**convergence_criterion_args):
    class_name = convergence_criterion_args["class_name"]
    convergence_criterion_args_copy = copy.deepcopy(convergence_criterion_args)
    convergence_criterion_args_copy.pop("class_name")
    klass = globals()[class_name]
    instance = klass(**convergence_criterion_args_copy)
    return instance

class Convergence_criterion(ABC):
    def __init__(self):
        self.it_call = 0
        self.initialized = False
        self.needs_to_be_initialized = False

    def initialization_steps(self, p:Params_With_Loss):
        if not self.needs_to_be_initialized:
            return True
        else:
            continue_flag = True
            if not self.initialized:
                self.it_call += 1
                if self.it_call < 3:
                    continue_flag = False
                elif self.it_call == 3:
                    self.initialize(p)
            return continue_flag

    @abstractmethod
    def initialize(self, p: Params_With_Loss):
        pass

    @abstractmethod
    def setting_lambda_P_epsilon(self):
        pass

    @abstractmethod
    def evaluate(self, p_b: Params_With_Loss, p:Params_With_Loss):
        pass


class Convergence_criterion_Relative_Delta_Loss(Convergence_criterion):
    """
    Convergence criterion for BCD. Convergence is achieved when both params and loss value have converged.
    Convergence types must be one of "delta_q_t/q_t-1", "delta_q_t/q_0", "delta_q_t", depending on if we want
    convergence relative to previous value, to initial value, or absolute convergence.
    """

    def __init__(self, params_convergence_type, loss_convergence_type, tol_param, tol_loss, pseudo_one_denominator = None, variables = None):
        """
        :param string params_convergence_type: either "delta_q_t/q_t-1", "delta_q_t/q_0", or "delta_q_t"
        :param string loss_convergence_type: either "delta_q_t/q_t-1", "delta_q_t/q_0", or "delta_q_t"
        :param mode_params: "sum" or "each". In sum, (w, gamma, b, gamma_dual) is considered as a stacked vector, in each, every separate component must obey the convergence criterion.
        :param float tol: Fills-in tol_param and tol_loss
        :param float tol_param: tolerance for convergence in parameters
        :param float tol_loss:tolerance for convergence in loss
        :param float pseudo_one_denominator: Small epsilon that avoids division by zero
        :param variables: list of variables in params checked for convergence for example ["w", "gamma", "b"]
        """
        super().__init__()
        possible_types = ["delta_q_t/q_t-1", "delta_q_t/q_0", "delta_q_t"]
        if (params_convergence_type not in possible_types) or (loss_convergence_type not in possible_types):
            raise ValueError("type must be in {}".format(possible_types))

        self.params_convergence_type = params_convergence_type
        self.loss_convergence_type = loss_convergence_type

        if (params_convergence_type == "delta_q_t/q_0") or (loss_convergence_type == "delta_q_t/q_0"):
            self.needs_to_be_initialized = True
        else:
            self.needs_to_be_initialized = False

        self.tol_param = tol_param
        self.tol_loss = tol_loss
        if variables is None:
            variables = all_variables_set
        self.variables = variables
        self.pseudo_one_denominator = pseudo_one_denominator
        self.param_norm_0 = None
        self.F_0 = None

    def initialize(self, param_0: Params_With_Loss):
        self.param_norm_0 = param_0.norm()
        self.param_0 = param_0
        self.F_0 = torch.abs(param_0.surrogate_loss_with_penalties().item())

    def to_lightweight(self, copy = True):
        if copy:
            res = Convergence_criterion_Relative_Delta_Loss(self.params_convergence_type, self.loss_convergence_type, self.tol_param, self.tol_loss, self.pseudo_one_denominator, self.variables)
        else:
            res = self
            self.param_norm_0 = None
            self.param_0 = None
            self.F_0 = None
        return res

    def evaluate(self, p_b: Params_With_Loss, p: Params_With_Loss):
        continue_flag = self.initialization_steps(p)
        if not(continue_flag):
            return False

        #convergence of loss value
        F_b = p_b.surrogate_loss_with_penalties()
        if self.loss_convergence_type == "delta_q_t/q_t-1":
            self.denominator_F = self.pseudo_one_denominator + F_b
        elif self.loss_convergence_type == "delta_q_t/q_0":
            self.denominator_F = self.pseudo_one_denominator + self.F_0
        elif self.loss_convergence_type == "delta_q_t":
            self.denominator_F = 1

        F = p.surrogate_loss_with_penalties()
        diff_F = F - F_b
        diff_parameter = (p - p_b)
        diff_parameter_norm = diff_parameter.norm()

        converged_F = (diff_F / self.denominator_F) <= self.tol_loss

        if self.params_convergence_type == "delta_q_t/q_t-1":
            parameter_before_norm = p_b.norm()
            denominator_params = self.pseudo_one_denominator + parameter_before_norm
        elif self.params_convergence_type == "delta_q_t/q_0":
            denominator_params = self.pseudo_one_denominator + self.param_norm_0
        elif self.params_convergence_type == "delta_q_t":
            denominator_params = 1

        converged_params = diff_parameter_norm/denominator_params <= self.tol_param

        return converged_F and converged_params

    def setting_lambda_P_epsilon(self):
        return self.tol_loss * self.denominator_F


class Convergence_criterion_Gradient_Based(Convergence_criterion):
    """
    Convergence criterion based on the value of the gradient of the loss.
    NB: if parameters do not change between iterations, convergence flag is activated in order not to loose
    computational time.
    """

    def __init__(self, epsilon_gradient = 10**(-5), epsilon_delta_param = 10**(-10), pseudo_one_denominator = None, variables = None):
        """
        :param float epsilon_gradient: epsilon for gradient convergence
        :param float epsilon_delta_param: epsilon for parameters "convergence"
        :param pseudo_one_denominator: epsilon to avoid division by zero
        :param variables: list of variables in params checked for convergence for example ["w", "gamma", "b"]
        """
        super().__init__()
        self.epsilon_gradient = epsilon_gradient
        self.epsilon_delta_param = epsilon_delta_param
        if variables is None:
            variables = all_variables_set
        self.variables = variables
        self.pseudo_one_denominator = pseudo_one_denominator
        self.needs_to_be_initialized = True

    def initialize(self, p: Params_With_Loss):
        self.loss_0 = p.surrogate_loss_with_penalties().item()
        self.initialized = True

    def evaluate(self, p_b: Params_With_Loss, p: Params_With_Loss):
        continue_flag = self.initialization_steps(p)
        if not(continue_flag):
            return False

        converged_per_variables = {}
        for variable_name in self.variables: #TODO: we could make it faster by checking first b and then more expensive ones!
            converged_per_variables[variable_name] = self.evaluate_per_variable(p_b, p, variable_name)

        return torch.all(torch.BoolTensor(list(converged_per_variables.values())))

    def evaluate_per_variable(self, p_b: Params_With_Loss, p:Params_With_Loss, variable_name):
        """evaluate convergence for a variable which is a torch tensor!"""
        v_b: torch.FloatTensor = getattr(p_b, variable_name)
        v: torch.FloatTensor = getattr(p, variable_name)
        gradient = norm_gradient_for_variables_set[variable_name](p).item()
        converged_gradient = gradient/(self.loss_0 + self.pseudo_one_denominator) <= self.epsilon_gradient
        converged_delta = (torch.norm(v - v_b).item() <= self.epsilon_delta_param) #TODO: do we need also a relative scale?
        converged = converged_delta or converged_gradient
        return converged

    def setting_lambda_P_epsilon(self):
        return None


class Convergence_criterion_Delta_Loss_Based(Convergence_criterion):
    """
    Convergence criterion based on the change of loss between subsequent iterations.
    """
    possible_denominator_methods = ["loss_initial", "1"]

    def __init__(self, epsilon_loss=10 **(-5), epsilon_delta_param=10 ** (-10), pseudo_one_denominator=None,
                 variables=None, denominator_method = "loss_initial"):
        """
        :param float epsilon_loss: tolerance for the change in the loss
        :param float epsilon_delta_param: tolerance for the change in params
        :param float pseudo_one_denominator: epsilon to avoid division by zero
        :param variables: list of variables (subset of w, gamma, b) which are checked for convergence.
        :param denominator_method: either "loss_initial" or "1". With "loss_initial", we divide the loss difference by
        the loss initial value during BCD before the comparison with epsilon.
        """
        super().__init__()
        self.epsilon_loss = epsilon_loss
        self.epsilon_delta_param = epsilon_delta_param
        if variables is None:
            variables = all_variables_set
        self.variables = variables
        self.pseudo_one_denominator = pseudo_one_denominator
        self.needs_to_be_initialized = True
        self.denominator_method = denominator_method
        if denominator_method not in Convergence_criterion_Delta_Loss_Based.possible_denominator_methods:
            raise ValueError(
                "denominator has invalid value {}.\nValid values are {}".format(denominator_method, Convergence_criterion_Delta_Loss_Based.possible_denominator_methods))

    def initialize(self, p: Params_With_Loss):
        self.loss_0 = p.surrogate_loss_with_penalties().item()
        if self.denominator_method == "loss_initial":
            self.denominator = self.loss_0
        else:
            self.denominator = 1

        self.initialized = True

    def converged_loss(self, p_b: Params_With_Loss, p: Params_With_Loss):
        dloss = torch.abs(p.surrogate_loss_with_penalties() - p_b.surrogate_loss_with_penalties())
        return dloss/self.denominator <= self.epsilon_loss

    def converged_delta_variables(self, p_b: Params_With_Loss, p: Params_With_Loss):
        converged_per_variables = {}
        for variable_name in self.variables:  # TODO: we could make it faster by checking first b and then more expensive ones!
            converged_per_variables[variable_name] = self.evaluate_per_variable(p_b, p, variable_name)
        torch.all(torch.BoolTensor(list(converged_per_variables.values())))

    def evaluate(self, p_b: Params_With_Loss, p: Params_With_Loss):
        continue_flag = self.initialization_steps(p)
        if not (continue_flag):
            return False

        converged_loss = self.converged_loss(p_b, p)
        if converged_loss:
            return True
        else:
            converged_delta_variables = self.converged_delta_variables(p_b, p)
            return converged_delta_variables

    def evaluate_per_variable(self, p_b: Params_With_Loss, p: Params_With_Loss, variable_name):
        """evaluate convergence for a variable which is a torch tensor!"""
        v_b: torch.FloatTensor = getattr(p_b, variable_name)
        v: torch.FloatTensor = getattr(p, variable_name)
        converged_delta = (torch.norm(v - v_b).item() <= self.epsilon_delta_param)  # TODO: do we need also a relative scale?
        return converged_delta

    def setting_lambda_P_epsilon(self):
        """
        :return: float with the loss difference needed to achieve convergence.
        """
        return self.epsilon_loss * self.denominator

def set_convergence_args(convergence_criterion: Convergence_criterion, **convergence_criterion_args):
    """
    convenience function to modify attributes of a convergence criterion object. The idea is to modify attributes like
    epsilon and to keep initialized attributes such as loss_0.
    :param Convergence_criterion convergence_criterion: convergence criterion object
    :param convergence_criterion_args: dictionary with attributes to modify. NB: must include the "class_name": class name
    :return: None
    """
    convergence_criterion_args_copy = copy.deepcopy(convergence_criterion_args)
    class_name = convergence_criterion_args_copy.pop("class_name")
    if class_name != convergence_criterion.__class__.__name__:
        raise ValueError("Wrong Class Name")
    for key, val in convergence_criterion_args_copy.items():
        setattr(convergence_criterion, key, val)

default_args = {"class_name": "Convergence_criterion_Relative_Delta_Loss", "params_convergence_type": "delta_q_t/q_t-1", "loss_convergence_type": "delta_q_t/q_t-1","tol_param": 10**(-5), "tol_loss": 10**(-5), "pseudo_one_denominator": 10**(-20)}
default_args_chill = {"class_name": "Convergence_criterion_Relative_Delta_Loss", "params_convergence_type": "delta_q_t/q_t-1", "loss_convergence_type": "delta_q_t/q_t-1", "tol_param": 10**(-3), "tol_loss": 10**(-3), "pseudo_one_denominator": 10**(-20)}
gradient_args = {"class_name": "Convergence_criterion_Gradient_Based", "epsilon_gradient": 10**(-5), "epsilon_delta_param": 10**(-10), "pseudo_one_denominator": 10**(-20)}
gradient_args_chill = {"class_name": "Convergence_criterion_Gradient_Based", "epsilon_gradient": 10**(-3), "epsilon_delta_param": 10**(-10), "pseudo_one_denominator": 10**(-20)}
absolute_delta_args = {"class_name": "Convergence_criterion_Delta_Loss_Based", "epsilon_loss": 10**(-5), "epsilon_delta_param": 10**(-10), "pseudo_one_denominator": 10**(-20)}
absolute_delta_args_chill = {"class_name": "Convergence_criterion_Delta_Loss_Based", "epsilon_loss": 10**(-3), "epsilon_delta_param": 10**(-10), "pseudo_one_denominator": 10**(-20)}

absolute_delta_args_wo_loss0 = {"denominator_method": "1", "class_name": "Convergence_criterion_Delta_Loss_Based", "epsilon_loss": 10**(-6), "epsilon_delta_param": 10**(-10), "pseudo_one_denominator": 10**(-20)}

#relaxation chill-strict
default_relaxation_args = {"convergence_criterion_args":default_args_chill, "convergence_criterion_args_last":default_args}
gradient_relaxation_args = {"convergence_criterion_args": gradient_args_chill, "convergence_criterion_args_last": gradient_args}
absolute_delta_relaxation_args = {"convergence_criterion_args": absolute_delta_args_chill, "convergence_criterion_args_last":absolute_delta_args}