from scipy.special import expit
from scipy.optimize import bisect
from utilities.small_functions import percentage_zero
import numpy as np


def adjust_factor(zs, percentage_intrinsic_noise):
    """
    Determines the factor needed to achieve a certain percentage of "intrinsic noise" in the data.
    More precisely, if w and b are multiplied by the factor, on average the percentage of "intrinsic noise" will match
    the specified level.
    :param torch tensor zs: 1D torch array generated with a certain observation matrix R and gamma and unscaled versions
    of w and b: zs = w.T R gamma + b
    :param float percentage_intrinsic_noise: percentage of intrinsic noise
    :return: factor (float)
    """

    expected_per_misclassified_as_fun_factor = lambda fac: expected_per_misclassified(
        expit((fac * zs))) - percentage_intrinsic_noise
    # we adjust the inverse temperature to have the expected percentage misclassified!! (similat to entropy!)

    trial = 0
    max_inverse_temp = 100
    succeed = False
    n_trial_max = 100
    while trial <= n_trial_max and not (succeed):
        try:
            fac = bisect(expected_per_misclassified_as_fun_factor, a=0, b=max_inverse_temp, xtol=10 ** (-10),
                         rtol=0.0001)
            succeed = True

        except ValueError:
            succeed = False
            max_inverse_temp = max_inverse_temp * 10

        trial = trial + 1
        if trial > n_trial_max:
            raise ValueError("Max iteration temp")

    return fac


def expected_per_misclassified(y_p):
    """
    Returns the expected percentage of "misclassified" samples. By misclassified sample,
    we mean a sample for which y_p > 0.5 != y. Note that y are drawn with a Bernouilli distribution of parameter y_p.
    :param y_p: 1-D torch array
    :return: float with expected percentage of misclassified samples.
    """
    return np.mean((1 - y_p) * (y_p > 0.5).astype("int") + y_p * (y_p <= 0.5).astype("int"))
