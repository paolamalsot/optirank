from utilities.intrinsic_noise import adjust_factor
from utilities.optirank.ranking_multiplication import zs_from_w_gamma_b
from utilities.optirank.src.loss.params import Params, params_with_loss_from_param
from utilities.optirank.src.loss.loss import ranking_logistic_loss
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import truncnorm
from scipy.optimize import fsolve
from utilities.small_functions import percentage_ones
import scipy.stats as stats
import scipy.optimize
import itertools
import numpy as np
import torch
import logging

max_std_trunc_norm = 1000


def b_to_balance_zs(zs):
    """
    From zs, finds b such that, on average the generated data has 50% positive and 50% negative classes.
    :param zs: 1D torch array
    :return: scalar
    """
    n_samples = len(zs)

    def prop_positive(b):
        return torch.sum(torch.sigmoid(zs + b)).item() / n_samples - 0.5

    root = fsolve(prop_positive, x0=-torch.mean(zs))
    return root


def attribute_class(X, indices_non_perturbing, card_w=None, card_gamma=None, distribution_w="gaussian",
                    percentage_instrinsic_noise=0.02):
    """
    Generate class y according to model explained in the paper.
    :param X: gene expression matrix.
    :param indices_non_perturbing: indices of non_perturbing genes. These are useful as w and gamma are non-zero only for those indices.
    :param card_w: number of non-zero indices in w.
    :param card_gamma: number of non-zero indices in gamma.
    :param distribution_w: either geometric or gaussian.
    :param percentage_instrinsic_noise: percentages of samples for which the attributes class is different from the most probable class according to the model.
    :return: attributed class, Params object with w, b, gamma...
    """

    d = X.shape[1]
    d_s = len(indices_non_perturbing)
    if card_w is None:
        card_w = d_s
    else:
        if card_w > d_s:
            raise ValueError("card_w cannot be greater than d_s")
    if card_gamma is None:
        card_gamma = d_s
    else:
        if card_gamma > d_s:
            raise ValueError("card_gamma cannot be greater than d_s")

    # generating gamma
    gamma_s = np.concatenate((np.ones(card_gamma), np.zeros(d_s - card_gamma)))
    gamma = np.zeros(d)
    gamma[indices_non_perturbing] = gamma_s
    gamma = torch.from_numpy(gamma)

    # generating w
    if distribution_w == "gaussian":
        w_s = np.random.choice([-1, 1], card_w) * (np.abs(np.random.randn(card_w)) + 1)
    elif distribution_w == "geometric":
        w_s = np.random.choice([-1, 1], card_w) * np.random.geometric(0.5, card_w)
    else:
        raise NotImplementedError

    w = np.zeros(d)
    w[indices_non_perturbing] = w_s
    w = torch.from_numpy(w)

    b = torch.Tensor([0])

    # prediction prior to adjustement of b
    zs = zs_from_w_gamma_b(X, w, gamma, b, "avg", d)
    # adjusting b to have a balanced dataset
    b = b_to_balance_zs(zs)
    zs = zs_from_w_gamma_b(X, w, gamma, b, "avg", d)
    # ajustement to have a certain percentage of intrinsic noise
    factor = adjust_factor(zs.numpy(), percentage_instrinsic_noise)
    w = w * factor
    b = b * factor
    zs = zs_from_w_gamma_b(X, w, gamma, b, "avg", d)

    p = Params(w=w, gamma=gamma, b=b, gamma_dual=torch.zeros_like(gamma))
    probas = torch.sigmoid(zs)
    y = np.random.binomial(1, probas)
    logging.info("percentage_intrinsic_noise is {}".format(1 - balanced_accuracy_score(y, probas > 0.5)))
    logging.info("percentage_positive_classes is {}".format(percentage_ones(y)))
    loss = ranking_logistic_loss(0, 0, 0, 0, 0, X=X, y_np=y,
                                 R_normalization="d")  # TODO: where do I use this loss, why not just use params ?
    p = params_with_loss_from_param(p, loss)

    return y, p


def phi(x):
    return scipy.stats.norm.pdf(x)


def mean_std_truncated(mu_sigma, a, b):
    # returns a function that for mu, sigma returns mean and variance
    mu = mu_sigma[0]
    sigma = mu_sigma[1]
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    Z = scipy.stats.norm.cdf(beta) - scipy.stats.norm.cdf(alpha)

    mean = mu + (phi(alpha) - phi(beta)) * sigma / Z
    var = sigma ** 2 * (1 + (alpha * phi(alpha) - beta * phi(beta)) / Z - ((phi(alpha) - phi(beta)) / Z) ** 2)
    std = var ** 0.5
    return [mean, std]


def generate_truncnorm(mu_sigma, a, b, n_samples):
    mu = mu_sigma[0]
    sigma = mu_sigma[1]
    if sigma > 0:
        n = stats.truncnorm(a=(a - mu) / sigma, b=(b - mu) / sigma, loc=mu, scale=sigma)
    elif sigma == 0:
        n = stats.truncnorm(a=(a - mu), b=(b - mu), loc=mu, scale=sigma)  # devrait donner toujours la mm chose..
    else:
        raise NotImplementedError
    return n.rvs(size=n_samples)


def generate_non_perturbing_genes(sigma_non_perturbing, n_non_perturbing, n_samples, truncated_gaussians=True):
    """
    Generates observation matrix for non-perturbing genes.
    : returns numpy array of size n_samples x n_non-perturbing
    """
    all_genes = []
    for i in range(n_non_perturbing):
        mu_i = np.random.uniform(0, 1, 1)
        if truncated_gaussians:
            rvs = generate_truncnorm([mu_i, sigma_non_perturbing], 0, 1, n_samples)
        else:
            rvs = np.random.randn(n_samples) * sigma_non_perturbing + mu_i
        all_genes.append(rvs)
    X = np.stack(all_genes, axis=1)
    return X


def generate_perturbing_genes(n_perturbing, n_samples, tau, sigma):
    """
    Generates observation matrix for perturbing genes.
    : returns numpy array of size n_samples x n_perturbing
    """

    mus = np.random.uniform(0, 1, n_perturbing)  # the big difference!
    deltas_samples = np.random.randn(n_samples) * tau
    X = np.zeros((n_samples, n_perturbing))
    for index_sample, index_gene in itertools.product(range(n_samples), range(n_perturbing)):
        mean = mus[index_gene] + deltas_samples[index_sample]

        rvs = np.random.randn(1) * sigma + mean
        X[index_sample, index_gene] = rvs
    return X


def generate_data(d, n_perturbing, n_samples, tau, sigma, sigma_non_perturbing=None, card_w=None, card_gamma=None):
    """
    Generate data following the model explained in the paper.
    :param d: dimension of gene expression profile.
    :param n_perturbing: number of perturbing genes.
    :param n_samples: number of samples.
    :param tau: typical shifting distance of the gene packet.
    :param sigma: background noise on perturbing genes.
    :param sigma_non_perturbing: background noise on non-perturbing genes.
    :param card_w: number of w_i different than 0.
    :param card_gamma: number of gamma_i equal to 1.
    :return: generated data
    """

    if sigma_non_perturbing is None:
        # The model detailed in the paper makes no distinction between sigma_non_perturbing and sigma.
        sigma_non_perturbing = sigma

    X_all = []

    if n_perturbing > 0:
        X_perturbing = generate_perturbing_genes(n_perturbing, n_samples, tau, sigma)
        X_all.append(X_perturbing)

    n_non_perturbing = d - n_perturbing

    X_non_perturbing = generate_non_perturbing_genes(sigma_non_perturbing, n_non_perturbing, n_samples)
    X_all.append(X_non_perturbing)
    indices_non_perturbing = np.arange(n_perturbing, n_perturbing + n_non_perturbing)

    X_all = np.hstack(X_all)
    y_all, p = attribute_class(X_all, indices_non_perturbing, card_w=card_w, card_gamma=card_gamma,
                               distribution_w="gaussian")

    return X_all, y_all, p, indices_non_perturbing
