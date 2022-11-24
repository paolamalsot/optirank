"""PROXIMAL OPERATOR FUNCTIONS AND CRITERIA FOR ACCEPTING STEPSIZES (argcriterion).
All these functions implement proximal operators. They depend on the previous parameter and on the inverse stepsize."""

from utilities.optirank.src.loss.params import Params, Params_With_Loss
import torch
from utilities.optirank.src.BCD.prox.prox_gamma_sum_gamma_fixed import prox_sum_to_k


def prox_w(p: Params_With_Loss, L_w):
    b = - L_w * p.w + p.dlogloss_dw()
    denominator = L_w + 2 * p.loss_object.lambda_w_2

    sol_1 = - (b + p.loss_object.lambda_w_1) / denominator
    sol_2 = - (b - p.loss_object.lambda_w_1) / denominator
    sol_3 = torch.zeros_like(b)
    sol = sol_3
    sol[b > p.loss_object.lambda_w_1] = sol_2[b > p.loss_object.lambda_w_1]
    sol[b < - p.loss_object.lambda_w_1] = sol_1[b < - p.loss_object.lambda_w_1]

    return sol


def prox_gamma_dual(p: Params_With_Loss, L_w, prox_gamma_dual):
    if prox_gamma_dual:
        a = p.loss_object.lambda_P / (L_w * 4) + 1 / 2
        b = p.loss_object.lambda_P / (2 * L_w) - p.gamma * p.loss_object.lambda_P / L_w - p.gamma_dual
        return (-b / (2 * a))
    else:
        return gamma_dual_opt(p)


def delta_gamma_dual_opt(p: Params):
    return 2 * p.gamma - 1 - p.gamma_dual


def gamma_dual_opt(p: Params):
    return 2 * p.gamma - 1


def prox_b(p: Params_With_Loss, L):
    return p.b - 1 / L * p.dlogloss_db()


def arg_criterion_L_w(p_b: Params_With_Loss, p_hat: Params_With_Loss,
                      L_w, prox_gamma_dual):
    p_b.zs(via="Rgamma")
    p_hat.zs(via="Rgamma")
    if prox_gamma_dual:
        return p_hat.logloss() + p_b.loss_object.LP2(p_hat.gamma, p_hat.gamma_dual) - \
               (p_b.logloss() + p_b.loss_object.LP2(p_b.gamma, p_b.gamma_dual) + torch.dot(p_b.dlogloss_dw(),
                                                                                           p_hat.w - p_b.w) +
                p_b.dlogloss_db() * (p_hat.b - p_b.b) + torch.dot(
                           p_b.loss_object.dLP2_dgamma_dual(p_b.gamma, p_b.gamma_dual),
                           (p_hat.gamma_dual - p_b.gamma_dual)) +
                L_w / 2 * torch.norm(p_hat.w - p_b.w) ** 2 + L_w / 2 * (p_hat.b - p_b.b) ** 2 + L_w / 2 * torch.norm(
                           p_hat.gamma_dual - p_b.gamma_dual) ** 2)
    else:
        return p_hat.logloss() - (p_b.logloss() + torch.dot(p_b.dlogloss_dw(), p_hat.w - p_b.w) + p_b.dlogloss_db() * (
                    p_hat.b - p_b.b) + L_w / 2 * torch.norm(p_hat.w - p_b.w) ** 2 +
                                  L_w / 2 * (p_hat.b - p_b.b) ** 2)


def arg_criterion_L_b(p_b: Params_With_Loss, p_hat: Params_With_Loss, L_b):
    p_hat.zs(via="wTR")
    p_b.zs(via="wTR")
    return p_hat.logloss() - (p_b.logloss() +
                              p_b.dlogloss_db() * (p_hat.b - p_b.b) + L_b / 2 * (p_hat.b - p_b.b) ** 2)


def update_L(L_min, L_baseline, eta, n):
    if n > 0:
        eta_ = eta[0]
    else:
        eta_ = eta[1]
    return max(L_min, L_baseline * eta_ ** n)


def prox_gamma(p: Params_With_Loss, L_gamma):
    a = L_gamma / 2 + p.loss_object.lambda_gamma_2
    b = - p.gamma * L_gamma + p.dlogloss_dgamma() + p.loss_object.dLP2_dgamma(p.gamma, p.gamma_dual)

    x_star = - (b + p.loss_object.lambda_gamma_1) / (2 * a)
    return torch.clamp(x_star, 0, 1)


def prox_gamma_sum_to_k(p: Params_With_Loss, L_gamma, k):
    x = 1 / L_gamma * (L_gamma * p.gamma - p.dlogloss_dgamma() - p.loss_object.dLP2_dgamma(p.gamma, p.gamma_dual))
    return prox_sum_to_k(x, k)


def arg_criterion_L_gamma(p_hat: Params_With_Loss, p: Params_With_Loss, L_gamma):
    p.zs(via="wTR")
    p_hat.zs(via="wTR")
    return p.logloss() + p_hat.loss_object.LP2(p.gamma, p.gamma_dual) - (
                p_hat.logloss() + p_hat.loss_object.LP2(p_hat.gamma, p_hat.gamma_dual) +
                torch.dot(p_hat.dlogloss_dgamma(), p.gamma - p_hat.gamma) + p_hat.dlogloss_db() * (p.b - p_hat.b) +
                torch.dot(p_hat.loss_object.dLP2_dgamma(p_hat.gamma, p_hat.gamma_dual), (p.gamma - p_hat.gamma)) +
                L_gamma / 2 * (torch.norm(p.gamma - p_hat.gamma) ** 2 + (p.b - p_hat.b) ** 2))
