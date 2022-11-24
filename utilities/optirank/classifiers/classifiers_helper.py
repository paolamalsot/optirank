import torch
from sklearn.base import BaseEstimator, TransformerMixin
from utilities.optirank.ranking_multiplication import ranking_transformation
from copy import deepcopy
from utilities.optirank.src.loss.params import Params_With_Loss, Params
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn import base
import numpy as np
import warnings


def create_param_from_log_regr_classifier(subsetrankinglogregrpipe_classifier):
    """
    Extracts solution from a subsetrankinglogregrpipe_classifier
    :param SubsetRankingLogRegrPipe subsetrankinglogregrpipe_classifier: subsetrankinglogregrpipe classifier
    :return: Params object with solution
    """
    w, b = subsetrankinglogregrpipe_classifier.get_wb()
    w = torch.from_numpy(w)
    b = torch.from_numpy(b)
    gamma = subsetrankinglogregrpipe_classifier.sol.gamma
    gamma_dual = subsetrankinglogregrpipe_classifier.sol.gamma_dual
    param = Params(w, gamma, b, gamma_dual)

    return param


class SubsetRankingLogRegrPipe(base.BaseEstimator):
    """
    Optirank-based estimator with gamma values prefitted with optirank,
    and w,b to be learned with a classic logistic regression implemented with scikit-learn.
    """

    def __init__(self, sol: Params_With_Loss, label_binarizer, tol=10 ** (-4), max_iter=1000, warm_start=True):
        """
        :param Params_With_Loss sol: Params_With_Loss object specifying the gamma values
        :param label_binarizer label_binarizer: label_binarizer (should be the same one as optirank)
        :param tol: tolerance for logistic regression
        :param max_iter: max_iter for logistic regression
        :param warm_start: if True, w and b will be warm-started from sol.
        """
        self.sol = sol
        self.tol = tol
        self.max_iter = max_iter
        self.label_binarizer = label_binarizer
        self.warm_start = warm_start

    def fit(self, X, y):
        n_samples = y.shape[0]
        subset_ranking_step = OptimalRankingSubsetTransformFromSol(self.sol)
        log_params = convert_penalties(self.sol.loss_object.lambda_w_1,
                                       self.sol.loss_object.lambda_w_2, n_samples)
        logistic_regression_classifier = LogisticRegression(class_weight=self.sol.loss_object.sample_weight,
                                                            tol=self.tol, solver="saga", max_iter=self.max_iter)

        if self.warm_start:
            logistic_regression_classifier.warm_start = True
            logistic_regression_classifier.coef_ = self.sol.w.numpy().reshape(1, -1)
            logistic_regression_classifier.intercept_ = self.sol.b.numpy().reshape(1)

        # we use saga in order to have elastic net penalty support
        logistic_regression_classifier.set_params(**log_params)
        self.pipe_ = make_pipeline(subset_ranking_step, logistic_regression_classifier)

        warnings.filterwarnings("ignore")
        self.pipe_.fit(X, y)

        self.converged = check_convergence_log_regr(self.pipe_.named_steps["logisticregression"])

        warnings.filterwarnings("default")

        return self

    def predict_proba(self, X):
        return self.pipe_.predict_proba(X)

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.squeeze(self.label_binarizer.inverse_transform(probas, 0.5))

    def get_wb(self):
        """
        :return (w, b): (w, b) tuple with 1-D numpy arrays
        """
        check_is_fitted(self)
        return self.pipe_.named_steps["logisticregression"].coef_.flatten(), self.pipe_.named_steps[
            "logisticregression"].intercept_.flatten()


class OptimalRankingSubsetTransformFromSol(BaseEstimator, TransformerMixin):
    """
    This scikit-learn compatible transformer converts a design matrix X into ranked data with respect to a reference
    subset.
    """

    def __init__(self, sol: Params_With_Loss):
        """
        :param Params_With_Loss sol: Params_With_loss object from which the reference subset will be extracted.
        """
        self.sol = sol

    def fit(self, X, y):
        return self

    def transform(self, X):
        rank_type = self.sol.loss_object.rank_type
        R_normalization = self.sol.loss_object.R_normalization
        gamma = self.sol.gamma
        return ranking_transformation(X, gamma, rank_type, R_normalization).numpy()


def pipeline_to_lightweight(pipeline, copy=True):
    """
    From a scikit-learn pipeline containing optirank-related elements, returns the corresponding pipeline with
    lightweight optirank-related elements
    :param pipeline: sklearn pipeline
    :param bool copy: flag to indicate whether the pipeline elements should be a copy of the original ones, or a
      reference.
    :return: scikit-learn pipeline
    """
    if copy:
        steps = []
        for key, pipe_element in pipeline.named_steps.items():
            if hasattr(pipe_element, "to_lightweight"):
                new_step = pipe_element.to_lightweight(copy=copy)
            else:
                new_step = deepcopy(pipe_element)
            steps.append(new_step)

        new_pipe = make_pipeline(*steps)
        return new_pipe

    else:
        for key, pipe_element in pipeline.named_steps.items():
            if hasattr(pipe_element, "to_lightweight"):
                pipe_element.to_lightweight(copy=copy)

        return pipeline


def convert_penalties(l1_w, l2_w, n_samples):
    """
    Converts optirank penalties on w into their equivalent for scikit learn logistic regression compatibility.
    :param float l1_w: l1 regularization
    :param float l2_w: l2 regularization
    :param int n_samples: number of samples to fit
    :return: dictionary with scikit-learn logistic-regression penalty arguments
    """
    if l1_w != 0 and l2_w != 0:
        penalty = "elasticnet"
    else:
        if l2_w == 0 and l1_w != 0:
            penalty = "l1"
        elif l1_w == 0 and l2_w != 0:
            penalty = "l2"
        else:
            penalty = "none"
    if penalty != "none":
        l1_ratio = l1_w / (2 * l2_w + l1_w)
        C = 1 / n_samples * (1 / (2 * l2_w + l1_w))
        # to avoid UserWarning: l1_ratio parameter is only used when penalty is 'elasticnet'. Got (penalty=l2)
        if penalty == "l2":
            if l1_ratio != 0:
                raise ValueError("l2 penalty is equivalent to l1_ratio 0")
            return {"penalty": penalty, "C": C, "l1_ratio": None}
        elif penalty == "l1":
            if l1_ratio != 1:
                raise ValueError("l1 penalty is equivalent to l1_ratio 1")
            return {"penalty": penalty, "C": C, "l1_ratio": None}
        elif penalty == "elasticnet":
            return {"penalty": penalty, "l1_ratio": l1_ratio, "C": C}
        else:
            raise ValueError("unknown penalty")
    else:
        return {"penalty": penalty, "C": 1.0, "l1_ratio": None}  # default value for C, should be ignored


def convert_penalties_inverse(l1_ratio, C, n_samples):
    """
    converts scikit learn logistic regression penalty arguments into their optirank equivalent
    :param float l1_ratio: l1_ratio (see scikit learn documentation)
    :param float C: C (see scikit learn documentation)
    :param int n_samples: number of samples to fit
    :return: dictionary with optirank regularization arguments
    """
    lambda_1_w = l1_ratio / (n_samples * C)
    lambda_2_w = (1 - l1_ratio) / (2 * n_samples * C)
    return {"lambda_1_w": lambda_1_w, "lambda_2_w": lambda_2_w}


def optimize_w_b_only(sol: Params_With_Loss, tol=10 ** (-4), max_iter=1000):
    """
    Takes a (w, gamma, b) solution and optimizes w,b for a fixed gamma with scikit-learn logistic regression.
    :param Params_With_Loss sol: Params_With_Loss solution
    :param float tol: tolerance for scikit-learn logistic regression
    :return: (Params_With_Loss, converged): new Params_With_Loss object with w optimized, converged flag for
      scikit-learn logistic regression.
    """

    # create a subset ranking log-regr pipe
    pipe = SubsetRankingLogRegrPipe(sol, tol, max_iter)

    # fit
    pipe.fit(sol.loss_object.X, sol.loss_object.y_np)
    w, b = [torch.from_numpy(el).to(dtype=sol.loss_object.dtype) for el in list(pipe.get_wb())]

    # create new loss param
    new_sol = sol.to(copy=True)
    new_sol.w = w
    new_sol.b = b

    converged = pipe.converged

    return new_sol, converged


def check_convergence_log_regr(log_regr: LogisticRegression):
    """
    This function checks whether a scikit learn logistic regression has converged within specified tolerance.
    :param log_regr: scikit-learn logistic regression object
    :return: boolean flag indicating convergence
    """
    if log_regr.n_iter_ == log_regr.max_iter:
        return False
    else:
        return True
