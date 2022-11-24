#BCD units which compose BCD instructions list.

from utilities.optirank.src.BCD.BCD_units.Prox_Steps import *
from utilities.optirank.src.BCD.BCD_units.Search_Method import Search_Method
from utilities.optirank.src.BCD.prox.prox_operators import arg_criterion_L_b, arg_criterion_L_gamma, arg_criterion_L_w, gamma_dual_opt, delta_gamma_dual_opt
import torch
from utilities.optirank.src.loss.params import zeros_like

class BCD_Unit(ABC):

    def __init__(self, with_diagnostics = False):
        self.diagnostics = {}
        self.with_diagnostics = with_diagnostics
        self.initialize_diagnostics()

    @abstractmethod
    def initialize_diagnostics(self):
        pass

    @abstractmethod
    def append_to_diagnostics(self):
        pass

    def gather_diagnostics(self):
        return self.diagnostics


class Update_Block(BCD_Unit):
    def __init__(self, vars, with_diagnostics):
        self.p = None
        self.dp = None
        self.vars = vars
        super().__init__(with_diagnostics)

    @abstractmethod
    def update(self, p: Params_With_Loss):
        pass

    def initialize_diagnostics(self):
        for var in self.vars:
            self.diagnostics["d"+var] = []
        self.diagnostics["dsurrogate_loss_with_penalties"] = []

    def save(self, p, dp, p_input):
        self.p_b = self.p #this is useful for convergence checks (if there is a convergence check afterwards!)
        self.p = p
        self.p_input = p_input
        self.dp = dp
        if self.with_diagnostics:
            self.append_to_diagnostics()

    def append_to_diagnostics(self):
        for var in self.vars:
            delta_var = torch.norm(getattr(self.dp, var)).item()
            self.diagnostics["d" + var].append(delta_var)
        self.diagnostics["dsurrogate_loss_with_penalties"].append(self.p.surrogate_loss_with_penalties().item() - self.p_input.surrogate_loss_with_penalties().item())

    def reset(self):
        self.dp = None
        self.p = None
        self.p_b = None
        self.p_input = None

class Update_Block_With_Prox(Update_Block):
    """ Implements a BCD unit which updates either b, gamma, gamma_dual, or a combination of those."""

    def __init__(self, vars, search_method_args = None, with_diagnostics = False):
        """
        :param vars: list with vars to update ("w", "gamma", "b", "gamma_dual")
        :param search_method_args: dictionary with keywords arguments for search_method.
        :param with_diagnostics: boolean flags for recording diagnostics.
        """
        if search_method_args is None:
            search_method_args = dict()
        self.search_method_args = search_method_args
        self.search_method_args["init_L_var_name"] = get_init_L_var_name_from_vars_in_prox(vars)
        self.search_method_args["with_diagnostics"] = with_diagnostics
        self.search_method = Search_Method(**self.search_method_args)
        self.prox_step = Multivariate_Prox_Step(vars)
        self.arg_criterion = get_arg_criterion(vars)
        super().__init__(vars, with_diagnostics)


    def update(self, p_input: Params_With_Loss):
        p, dp = self.search_method.run(self.prox_step, p_input, self.arg_criterion)
        self.save(p, dp, p_input)
        return p

    def reset(self):
        self.search_method.reset()
        super().reset()

    def initialize_diagnostics(self):
        self.search_method.initialize_diagnostics()
        super().initialize_diagnostics()

    def save(self, p, dp, p_input):
        if self.with_diagnostics:
            dp = p - p_input
        super().save(p, dp, p_input)

    def gather_diagnostics(self):
        return {**self.diagnostics, **self.search_method.diagnostics}

def get_arg_criterion(vars):
    """
    returns the appropriate arg_criterion method for the search_method.
    NB: the arg_criterion will check the change in vars in p, potentially more variables.
    :params: vars: list with names of variables to be updated.
    """
    i = get_index_of_smallest_set_containing(vars, [el[0] for el in arg_criterion_dicts])
    full_arg_crit = arg_criterion_dicts[i][1]
    arg_crit = lambda p_b, p, dp, L: full_arg_crit(p_b, p, L)
    return arg_crit

def get_index_of_smallest_set_containing(list, list_of_sets):
    """gets the index in list_of_sets of the smallest set containing all elements in list."""
    inside = np.array([np.all([el in set for el in list]) for set in list_of_sets])
    lengths = np.array([len(set) for set in list_of_sets])
    indices_inside = np.nonzero(inside)[0]
    return indices_inside[np.argmin(lengths[inside])]

def get_init_L_var_name_from_vars_in_prox(vars):
    """
    Based on the variables to be updated in the prox_step, gives the variable pertinent to L (stepsize) initialization
    in the linesearch method.
    :param vars: List of variables
    :return: pertinent variable for init_L initialization
    """
    varname: str
    for varname in ["w", "gamma", "b"]: #ordered by priority: For example, if both w and b are updated, L takes into account only w.
        if varname in vars:
            return varname
    raise NotImplementedError


#super wierd thing to make the correspondance between smt coded badly and smt well coded!
arg_criterion_dicts = [(("w", "b", "gamma_dual"), lambda p_b, p, L: arg_criterion_L_w(p_b, p, L, prox_gamma_dual=True)),
                        (("w", "b"), lambda p_b, p, L: arg_criterion_L_w(p_b, p, L, prox_gamma_dual=False)),
                       (("b"), lambda p_b, p, L: arg_criterion_L_b(p_b, p, L)),
                     (("gamma", "b"), lambda p_b, p, L: arg_criterion_L_gamma(p_b, p, L))
]

class Update_Gamma_Dual_No_Prox(Update_Block):
    """block to update gamma dual (without prox.step)"""

    def __init__(self, with_diagnostics):
        super().__init__(["gamma_dual"], with_diagnostics)

    def update(self, p: Params_With_Loss):
        gamma_dual = gamma_dual_opt(p)
        dgamma_dual = delta_gamma_dual_opt(p)
        p_next = p.to(copy = True, copy_loss=False)
        p_next.gamma_dual = gamma_dual
        dp = zeros_like(p)
        dp.gamma_dual = dgamma_dual
        self.save(p_next, dp, p)
        return p_next


def create_update_block(descriptor, search_method_args, with_diagnostics):
    """
    :param descriptor: string describing the block needed
    :param search_method_args: additional search_method args
    :return: initialized BCD Block
    """

    if descriptor == "gamma_dual_no_prox":
        block = Update_Gamma_Dual_No_Prox(with_diagnostics)
    else: #proximal_blocks
        variables = descriptor.split(",")
        block = Update_Block_With_Prox(vars = variables, search_method_args = search_method_args, with_diagnostics=with_diagnostics)

    return block

def variables_from_block_list(block_list):
    return set([var for block in block_list for var in block.vars])