from utilities.optirank.src.BCD.BCD_units.Check_convergence import Check_Convergence_Step
from utilities.optirank.src.BCD.BCD_units.BCD_unit import create_update_block

def create_instructions_list(architecture_description, search_method_args, convergence_criterion_args, with_diagnostics):
    """
    Method that creates instructions_list (and block_list) for an architecture description, and some more arguments
    :param architecture_description: list of "block-descriptors" describing the architecture of the BCD algorithms. See
            the classical_architecture for an example
    :param search_method_args: dictionary with arguments to pass to the search method
    :param convergence_criterion_args: convergence_criterion_args
    :param with_diagnostics: boolean flag for recording diagnostics
    :return: a list of (descriptor, BCD unit) tuples to run sequencially!
    """
    blocks_list = []
    indices_convergence_checks = []

    #create block list (wo convergence checks)
    for index_el, descriptor in enumerate(architecture_description):
        if descriptor == "convergence_check":
            indices_convergence_checks.append(len(blocks_list))
        else:
            new_block = create_update_block(descriptor, search_method_args, with_diagnostics = with_diagnostics)
            blocks_list.append(new_block)

    #create_list_instructions thanks to list_blocks and indices_convergence_checks
    list_convergence_checks = [Check_Convergence_Step(blocks_list, index_step, convergence_criterion_args, with_diagnostics=with_diagnostics) for index_step in indices_convergence_checks]
    list_instructions = []
    iterator_block_list = iter(blocks_list)
    for index_el, descriptor in enumerate(architecture_description):
         if descriptor == "convergence_check":
             list_instructions.append(("convergence_check", list_convergence_checks.pop(0)))
         else:
             list_instructions.append((descriptor, next(iterator_block_list)))

    if (len(list_convergence_checks) > 0) or (iterator_block_list.__length_hint__() > 0):
        raise ValueError("Definitely a problem in creating instruction list")

    return list_instructions


def gather_diagnostics_of_units(list_instructions):
    diagnostics = {}
    for index, (descriptor, BCD_unit) in enumerate(list_instructions):
        diagnostics_of_units = BCD_unit.gather_diagnostics()
        #append index of step and descriptor to keys of BCD_unit to avoid overlapping column titles
        diagnostics_out = {str(index)+"_"+descriptor+"__"+key :value for key, value in diagnostics_of_units.items()}
        diagnostics = {**diagnostics, **diagnostics_out}
    return diagnostics


classical_architecture_2_convergence_checks = ["w,b", "convergence_check", "gamma,b", "gamma_dual_no_prox", "convergence_check"]
classical_architecture = ["w,b","gamma,b", "gamma_dual_no_prox", "convergence_check"]
classical_architecture_separate_b = ["w", "b", "gamma", "b", "gamma_dual_no_prox", "convergence_check"]
classical_architecture_separate_b_once = ["w", "b", "gamma", "gamma_dual_no_prox", "convergence_check"]
paper_theorical_architecture = ["w,gamma_dual,b", "gamma,b", "convergence_check"]
optimize_w_b_only_architecture = ["w,b", "convergence_check"]