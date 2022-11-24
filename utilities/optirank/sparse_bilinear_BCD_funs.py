import torch
from utilities.optirank.ranking_multiplication import zs_from_Rgamma, zs_from_wTR
from utilities.optirank.src.loss.params import Params, Params_With_Loss


def minimal_gradient(subgradient):
    """
    :param subgradient: dx2 (For every variable, min and max derivative) (implying that every derivative between min and max belongs to subgradient)
    :return: For every variable, derivative closest 0 (in absolute value)
    """
    temp = torch.gather(subgradient, 1, torch.argmin(torch.abs(subgradient), dim=1,
                                                     keepdim=True))  # take the subgradient with mimimal absolute value in each coordinate
    temp[torch.logical_and(subgradient[:, 0] <= 0, subgradient[:, 1] >= 0)] = 0.0
    return temp


def rel_diff(param_before, param_after, pseudo_one_denominator=10 ** (-20), param_0=None, type="delta_q_t/delta_q_t-1"):
    if type not in ["delta_q_t/delta_q_t-1", "delta_q_t/delta_q_0"]:
        return ValueError("type not in possible types")
    """returns for param-before, param-after, the relative the norm of the change in respect to the norm of the initial parameter.
    To avoid division by zero, we compare to the latter parameter if the first has zero norm."""
    if (param_0 is not None) and (type == "delta_q_t/delta_q_0"):
        if torch.norm(param_0) + pseudo_one_denominator > 0:
            return (torch.norm(param_after - param_before) / (torch.norm(param_0) + pseudo_one_denominator)).item()
        else:
            raise ValueError("torch.norm(param_0) + pseudo_one_denominator = 0")
    elif (param_0 is None) and (type == "delta_q_t/delta_q_t-1"):
        if torch.norm(param_before) + pseudo_one_denominator > 0:
            return (torch.norm(param_after - param_before) / (torch.norm(param_before) + pseudo_one_denominator)).item()
        else:
            raise ValueError("torch.norm(param_before)) + pseudo_one_denominator = 0")
    else:
        raise ValueError("Wrong usage of relative difference")


def difference_in_LP2(dp, p_f, p_b):
    return p_b.loss_object.lambda_P * torch.sum((- dp.gamma * p_f.gamma_dual - dp.gamma_dual * p_b.gamma))


def differences_in_zs(dp: Params_With_Loss, p_f: Params_With_Loss, p_b: Params_With_Loss):
    # TODO: vraiment pas rigoureux, modified just for the purpose of testing...
    type = p_b.loss_object.rank_type
    if hasattr(p_f, "_Rgamma"):
        return zs_from_Rgamma(dp.w, p_b.Rgamma(), 0, type) + zs_from_wTR(p_f.wTR(), dp.gamma, dp.w, 0, type) + dp.b
    else:
        return zs_from_wTR(p_b.wTR(), dp.gamma, dp.w, 0, type) + zs_from_Rgamma(dp.w, p_f.Rgamma(), 0, type) + dp.b


def differences_in_minus_logsigmoid(dp: Params, p_final: Params, p_before: Params):
    if hasattr(p_final, "_Rgamma"):
        via = "Rgamma"
    else:
        via = "wTR"
    return torch.nn.functional.logsigmoid(-(p_final.zs(via))) - torch.nn.functional.logsigmoid(-(p_before.zs(via)))


def differences_in_logloss(dp: Params, p_final: Params, p_before: Params):
    # returns logloss_after - logloss_before
    w = p_final.loss_object.sample_weights
    argdiff_0 = p_final.loss_object.y * differences_in_zs(dp, p_final, p_before)
    argdiff_1 = differences_in_minus_logsigmoid(dp, p_final, p_before)
    argdiff = (argdiff_0 + argdiff_1) * p_final.loss_object.sample_weights
    diff = - torch.sum(argdiff)
    return diff


def diff_abs_norm_delta(x_b, dx):
    # returns abs(x) - abs(x_b) with x = x_b + dx
    return torch.sum(torch.minimum(torch.maximum(dx, -dx - 2 * x_b), torch.maximum(dx + 2 * x_b, -dx)))


def difference_L_P(p_b, p_f, dp):
    return p_b.loss_object.lambda_P * torch.sum(dp.gamma_dual / 2 * (
                1 + (p_f.gamma_dual + p_b.gamma_dual) / 2) - dp.gamma * p_f.gamma_dual - dp.gamma_dual * p_b.gamma)


def differences_in_surrogate_loss_with_penalties(dp: Params, p_f: Params, p_b: Params):
    return differences_in_logloss(dp, p_f, p_b) + p_b.loss_object.lambda_w_1 * diff_abs_norm_delta(p_b.w,
                                                                                                   dp.w) + p_b.loss_object.lambda_gamma_1 * diff_abs_norm_delta(
        p_b.gamma, dp.gamma) + p_b.loss_object.lambda_w_2 * torch.dot(dp.w,
                                                                      p_b.w + p_f.w) + p_b.loss_object.lambda_gamma_2 * torch.dot(
        dp.gamma, p_b.gamma + p_f.gamma) + difference_L_P(p_b, p_f, dp)
