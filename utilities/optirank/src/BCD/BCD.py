from utilities.optirank.src.BCD.Create_Instructions_List import create_instructions_list, gather_diagnostics_of_units
from utilities.optirank.src.BCD.initialization import InitializazionParameters
from utilities.optirank.src.BCD.BCD_units.convergence_criterion import Convergence_criterion, set_convergence_args
from utilities.optirank.src.loss.loss import ranking_logistic_loss
from utilities.optirank.src.loss.params import Params_With_Loss, Params
from utilities.optirank.src.BCD.BCD_units.BCD_unit import Update_Block
from utilities.optirank.src.BCD.BCD_units.Check_convergence import Check_Convergence_Step
from utilities.optirank.src.BCD.BCD_units.Search_Method import Search_Method
from utilities.small_functions import dispatch_arguments_for_classes
import logging

meta_args_classes = [Search_Method, InitializazionParameters, Convergence_criterion] #classes whose arguments can be written normally in BCD init signature #TODO: what means normally??
meta_args_classes = [Search_Method, InitializazionParameters, Convergence_criterion] #classes whose arguments can be written normally in BCD init signature #TODO: what means normally??

class BCD():
    """ class implementing the block-coordinate descent algorithm """
    def __init__(self, BCD_architecture, max_iter, **kwargs):
        """
        :param BCD_architecture: list of strings describing the architecture of BCD (see Create_Instructions_List for more details)
        :param init_method: name of init_method (see class InitializazionParameters)
        :param max_iter: maximum_iteration allowed
        :param kwargs: contains additional arguments meant for meta_args_classes (for example search_method__eta, convergence_criterion__class...)
        """

        self.BCD_architecture = BCD_architecture
        self.max_iter = max_iter
        dispatched_arguments = dispatch_arguments_for_classes(meta_args_classes, kwargs)
        self.convergence_criterion_args = {**dispatched_arguments["Convergence_criterion"]}
        self.search_method_args = {**dispatched_arguments["Search_Method"]}
        self.init_method_args = dispatched_arguments["InitializazionParameters"]

    def set_convergence_criterion_args(self, convergence_criterion_args):
        """
        Sets for every convergence step in the instruction list the specified convergence criterion arguments (can include the class_name!).
        NB: it keeps values saved by the convergence criterion object.
        :param convergence_criterion_args: dictionary with class_name and other keyword arguments
        """
        self.convergence_criterion_args = convergence_criterion_args
        for descriptor, block in self.list_instructions:
            if isinstance(block, Check_Convergence_Step):
                set_convergence_args(block.convergence_criterion, **convergence_criterion_args)

    def initialize_p(self, p_init: Params, loss_object: ranking_logistic_loss):
        """
        Initializes the parameter values to p_init if any, or with initialization method.
        :param p_init: Params object or None
        :param loss_object: ranking_logistic_loss object
        """
        if p_init is not None:
            self.p = p_init
            if loss_object is not None:
                if isinstance(self.p, Params_With_Loss):
                    assert self.p.loss_object == loss_object
                else:
                    self.p.loss_object = loss_object
        else:
            if loss_object is None:
                raise ValueError("no loss object has been provided (either through p_init or through loss_object argument!)")

            init_method = InitializazionParameters(**self.init_method_args, loss_object = loss_object)
            self.p = init_method.initialize_parameters()

    def initialize_diagnostics(self, with_diagnostics, diagnostics_funs_dict):
        """ initializes diagnostics at the BCD-level. Each BCD_unit will record its own diagnostics."""
        self.diagnostics = {"iter":[], "accuracy":[], "log-loss":[]}
        self.diagnostics_funs_dict = diagnostics_funs_dict
        self.with_diagnostics = with_diagnostics
        if self.diagnostics_funs_dict is not None:
            self.diagnostics_funs = {key:[] for key in self.diagnostics_funs_dict.keys()}
        else:
            self.diagnostics_funs = {}


    def append_diagnostics(self):
        """ updates diagnostics at the BCD level. """
        self.diagnostics["iter"].append(self.iter)
        self.diagnostics["accuracy"].append(self.p.accuracy())
        self.diagnostics["log-loss"].append(self.p.logloss().item())

        if self.diagnostics_funs_dict is not None:
            for key, fun in self.diagnostics_funs_dict.items():
                self.diagnostics_funs[key].append(fun(self.p))

    def run(self, loss_object = None, p_init: Params_With_Loss = None, with_diagnostics = False, diagnostics_funs_dict = None, warm_start = False):

        self.initialize_diagnostics(with_diagnostics, diagnostics_funs_dict)
        self.initialize_p(p_init, loss_object)

        if not warm_start:
            self.list_instructions = create_instructions_list(self.BCD_architecture, self.search_method_args, self.convergence_criterion_args, self.with_diagnostics)
        else:
            for descriptor, block in self.list_instructions:
                block.initialize_diagnostics()
                block.reset()

        #running the instructions list!
        self.converged = False
        self.iter = 0

        while not(self.converged) and self.iter < self.max_iter:
            if self.with_diagnostics:
                self.append_diagnostics()
            for index_step, (descriptor, BCD_block) in enumerate(self.list_instructions):
                if isinstance(BCD_block, Update_Block):
                    self.p = BCD_block.update(self.p)
                elif isinstance(BCD_block, Check_Convergence_Step):
                    self.converged = BCD_block.check()
                    if self.converged:
                        break

            self.iter +=1

        self.gather_diagnostics()
        if not(self.converged):
            logging.info("BCD has not converged in the given iterations")


    def gather_diagnostics(self):
        self.diagnostics = {**self.diagnostics, ** self.diagnostics_funs, **gather_diagnostics_of_units(self.list_instructions)}


def get_last_convergence_step(BCD_algo: BCD):
    list_instructions = BCD_algo.list_instructions
    index_conv_step = None
    for index_step, (descriptor, BCD_block) in enumerate(list_instructions):
        if isinstance(BCD_block, Check_Convergence_Step):
            index_conv_step = index_step
    if index_conv_step is None:
        raise ValueError("No convergence step")
    return list_instructions[index_conv_step][1]

def setting_lambda_P_epsilon_from_BCD(BCD_algo: BCD): #useful for delta_loss setting lambda_P strategy.
    convergence_block = get_last_convergence_step(BCD_algo)
    conv_criterion = convergence_block.convergence_criterion
    return conv_criterion.setting_lambda_P_epsilon()