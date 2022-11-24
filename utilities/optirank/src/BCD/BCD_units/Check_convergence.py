import itertools
from utilities.optirank.src.BCD.BCD_units.BCD_unit import BCD_Unit, variables_from_block_list
from utilities.optirank.src.BCD.BCD_units.convergence_criterion import make_convergence_criterion
import torch
varnames = ["w", "gamma", "b", "gamma_dual"]


class Check_Convergence_Step(BCD_Unit):
    """
    This class implements the BCD unit which checks BCD convergence.
    """
    def __init__(self, blocks_list, index_step, convergence_criterion_args, with_diagnostics):
        """
        :param blocks_list: list of BCD unit blocks without Check_Convergence_Step objects (otherwise recursivity)
        :param index_step: index of the step preceding check_convergence in the block_list
        :param convergence_criterion_args: dictionary with class and named arguments for convergence_criterion object
        :param with_diagnostics: boolean flag for recording diagnostics
        """
        self.blocks_list = blocks_list
        self.index_step = index_step
        self.variables = variables_from_block_list(self.blocks_list)
        self.convergence_criterion = make_convergence_criterion(**convergence_criterion_args, variables = self.variables)
        super().__init__(with_diagnostics=with_diagnostics)


    def check(self):

        #p_b and p
        self.p_b = get_p_b(self.blocks_list, self.index_step)
        self.p = get_p(self.blocks_list, self.index_step)

        if self.with_diagnostics:
            self.append_to_diagnostics()

        if (self.p_b is None) or (self.p is None):
            return None
        else:
            return self.convergence_criterion.evaluate(self.p_b, self.p)

    def reset(self):
        self.p = None
        self.p_b = None

    def initialize(self):
        p = get_p(self.blocks_list, self.index_step)
        if p is None:
            raise ValueError("p is None")
        else:
            self.convergence_criterion.set_initial(p)

    def initialize_diagnostics(self):
        var_diagnostics = {"|"+prefix + var+"|":[] for var, prefix in itertools.product(varnames, ["", "d"])}
        surrogate_loss_diagnostics = {"surrogate_loss_with_penalties":[],
                            "dsurrogate_loss_with_penalties": []}
        self.diagnostics = {**var_diagnostics, **surrogate_loss_diagnostics}

    def append_to_diagnostics(self):
        for var in varnames:
            self.diagnostics["|" + var +"|"].append(torch.norm(getattr(self.p, var)).item())
            if self.p_b is not None:
                self.diagnostics["|d" + var +"|"].append(torch.norm(getattr(self.p, var) - getattr(self.p_b, var)).item())
            else:
                self.diagnostics["|d" + var + "|"].append(None)
        if self.p_b is not None:
            self.diagnostics["dsurrogate_loss_with_penalties"].append(
                (self.p.surrogate_loss_with_penalties() - self.p_b.surrogate_loss_with_penalties()).item())
        else:
            self.diagnostics["dsurrogate_loss_with_penalties"].append(None)
        self.diagnostics["surrogate_loss_with_penalties"].append(self.p.surrogate_loss_with_penalties().item())

def get_p(blocks_list, index_step):
    return blocks_list[index_step - 1].p

def get_p_b(blocks_list, index_step):
    return blocks_list[index_step - 1].p_b