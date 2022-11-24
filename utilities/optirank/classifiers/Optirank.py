from utilities.optirank.src.loss.loss import ranking_logistic_loss
from utilities.optirank.src.loss.params import Params, Params_With_Loss, params_with_loss_from_param
from utilities.optirank.src.relaxation.lambda_P_setting import rounding_and_scaling_thresh, rounding_thresh
from utilities.optirank.src.relaxation.lambda_P_setting_helper import sum_dist_to_border
from utilities.optirank.classifiers.classifiers_helper import SubsetRankingLogRegrPipe
from utilities.optirank.src.BCD.BCD import BCD, setting_lambda_P_epsilon_from_BCD
from utilities.optirank.src.BCD.BCD_units.convergence_criterion import Convergence_criterion
from utilities.optirank.classifiers.default_args import default_setting_lambda_P_strategy_args
import utilities.optirank.src.loss.params as params
from utilities.small_functions import dispatch_arguments_for_classes
from utilities.small_functions import zip_with_convention
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator
from abc import ABC
import numpy as np
import logging
from copy import deepcopy
import inspect
import torch


class BilinearRankingClassifier(BaseEstimator):
    """
    Classifier with a fixed value of lambda P.
    Fitting this classifier solves the minimization problem with a *fixed* value of lambda_P.
    """

    def __init__(self, lambda_gamma_1=0, lambda_gamma_2=0, lambda_w_1=0, lambda_w_2=0, lambda_P=0,
                 sample_weight="balanced", constraint_sum_gamma_k=None, constraint_per_gamma_k=None, rank_type="avg",
                 R_normalization=None, **kwargs):
        """
        :param lambda_gamma_1: l1 regularization parameter on gamma (scalar)
        :param lambda_gamma_2: l2 regularization parameter on gamma (scalar)
        :param lambda_w_1: regularization parameter on w (scalar)
        :param lambda_w_2: regularization parameter on w (scalar)
        :param lambda_P: regularization parameter for the push-penalty (scalar)
        :param sample_weight: either "balanced" or None (None: every sample bears equal weight, "balanced": every class has same weight.)
        :param constraint_sum_gamma_k: either None or integer for specifying |gamma|_1 constraint
        :param constraint_per_gamma_k: either None or scalar for specifying |gamma|_1/d constraint
        :param rank_type: either "avg", "min", or "max"
        :param R_normalization: either None, "d", "sqrt(d)", "k" or "sqrt(k)"
        :param kwargs: keyword-arguments destined to BCD (see usages in BCD class)
        """
        self.lambda_gamma_1 = lambda_gamma_1
        self.lambda_gamma_2 = lambda_gamma_2
        self.lambda_w_1 = lambda_w_1
        self.lambda_w_2 = lambda_w_2
        self.lambda_P = lambda_P
        self.sample_weight = sample_weight
        self.constraint_sum_gamma_k = constraint_sum_gamma_k
        self.constraint_per_gamma_k = constraint_per_gamma_k
        self.rank_type = rank_type
        self.R_normalization = R_normalization
        self.bcd_args = kwargs

    def get_params(self, deep=True):
        return {**BaseEstimator.get_params(self, deep=True), **self.bcd_args}

    def set_loss_args(self):
        """ sets the regularization and other parameters to the loss to minimize. """
        self.loss_args = {"constraint_sum_gamma_k": self.constraint_sum_gamma_k,
                          "constraint_per_gamma_k": self.constraint_per_gamma_k,
                          "lambda_w_1": self.lambda_w_1,
                          "lambda_w_2": self.lambda_w_2,
                          "lambda_gamma_1": self.lambda_gamma_1,
                          "lambda_gamma_2": self.lambda_gamma_2,
                          "lambda_P": self.lambda_P,
                          "sample_weight": self.sample_weight,
                          "rank_type": self.rank_type,
                          "R_normalization": self.R_normalization}

    def not_converged_msg(self):
        logging.debug("Bilinear Ranking Classifier with lambda_P={} did not converge".format(self.lambda_P))

    def fit(self, X, y, p_init: Params = None, with_diagnostics=False, diagnostics_funs_dict=None):
        """
        Solves the minimization problem with BCD algorithm.
        Saves the optimal regression parameters in "sol" attribute
        "converged" attribute serves as a flag for successful convergence.
        :param X: numpy array nxd
        :param y: numpy array n
        :param p_init: parameter for warm start (class Param)
        :param with_diagnostics: Boolean indicating whether to record diagnostics (warning: takes time and memory)
        :param diagnostics_funs_dict: dictionary with name: fun entries. Each fun takes as argument a Params object.
        """

        classes = np.unique(y)
        if np.size(classes) > 2:
            raise ValueError("Optirank is a binary classifier and cannot accept more than 2 classes in y")

        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        self.label_binarizer_.fit(classes)
        self.classes_ = self.label_binarizer_.classes_
        y_hot = np.squeeze(self.label_binarizer_.transform(y).toarray().astype('int'))
        # create loss object and call BCD
        self.set_loss_args()
        loss_object = ranking_logistic_loss(X=X, y_np=y_hot, **self.loss_args)
        self.bcd = BCD(**self.bcd_args)
        if p_init is not None:
            p_init = params_with_loss_from_param(p_init, loss_object)
        self.bcd.run(loss_object=loss_object, p_init=p_init, with_diagnostics=with_diagnostics,
                     diagnostics_funs_dict=diagnostics_funs_dict)
        self.sol, self.converged, self.diagnostics = self.bcd.p, self.bcd.converged, self.bcd.diagnostics

        if not self.converged:
            self.not_converged_msg()

    def fit_next_relaxation_step(self, p_init: Params_With_Loss, with_diagnostics, diagnostics_funs_dict):
        """
        Takes the bcd in memory, changes the lambda_P of the p_init given to the self.lambda_P, runs the BCD in warm_start mode.
        Note that between subsequent iterations, convergence criterion arguments can be changed but other bcd arguments are not updated (!).
        Time is gained wrt to the fit method because BCD is not reset from scratch.
        :param p_init: parameter for warm start (class Param_With_Loss)
        :param with_diagnostics: Boolean indicating whether to record diagnostics (warning: takes time and memory)
        :param diagnostics_funs_dict: dictionary with name: fun entries. Each fun takes as argument a Params object.
        """
        p_init.set_lambda_P(self.lambda_P)
        # setting convergence criterion args: needed when high_tol=True
        convergence_criterion_args = dispatch_arguments_for_classes([Convergence_criterion], self.bcd_args)[
            "Convergence_criterion"]
        self.bcd.set_convergence_criterion_args(convergence_criterion_args)
        self.bcd.run(loss_object=None, p_init=p_init, with_diagnostics=with_diagnostics,
                     diagnostics_funs_dict=diagnostics_funs_dict, warm_start=True)
        self.sol, self.converged, self.diagnostics = self.bcd.p, self.bcd.converged, self.bcd.diagnostics
        if not self.converged:
            self.not_converged_msg()

    def predict_proba(self, X):
        """
        :param X: numpy array nxd
        :return: numpy array nx2 array with probabilities for negative and positive class respectively (for sci-kit learn compatibility)
        """
        probas = params.predict_probas(self.sol, X)
        return np.stack((1 - probas, probas), axis=1)

    def predict(self, X):
        """
        :param X: numpy array nxd
        :return: numpy array n with predicted classes for each sample.
        """
        probas = self.predict_proba(X)
        return np.squeeze(self.label_binarizer_.inverse_transform(probas, 0.5))

    def to_lightweight(self, copy=True):
        """function to call prior to saving in pkl format in order to save space"""
        if copy:
            args = list(inspect.signature(BilinearRankingClassifier).parameters.keys())
            args.remove("kwargs")
            res = BilinearRankingClassifier(**{el: getattr(self, el) for el in args})
        else:
            res = self

        # what is to remove, what is to keep!
        res.bcd = None
        res.sol = self.sol.to_lightweight(copy=copy)
        res.diagnostics = None
        res.converged = self.converged
        res.label_binarizer_ = self.label_binarizer_
        res.classes_ = self.classes_
        return res


class OptirankBasedClassifier(BaseEstimator, ABC):
    """
    Abstract class for classifiers based on BilinearRankingClassifier.
    """

    def __init__(self, lambda_gamma_1=0, lambda_gamma_2=0, lambda_w_1=0, lambda_w_2=0, sample_weight="balanced",
                 constraint_sum_gamma_k=None, constraint_per_gamma_k=None, rank_type="avg", R_normalization=None,
                 **kwargs):
        self.lambda_gamma_1 = lambda_gamma_1
        self.lambda_gamma_2 = lambda_gamma_2
        self.lambda_w_1 = lambda_w_1
        self.lambda_w_2 = lambda_w_2
        self.constraint_sum_gamma_k = constraint_sum_gamma_k
        self.constraint_per_gamma_k = constraint_per_gamma_k
        self.sample_weight = sample_weight
        self.bcd_args = kwargs
        self.rank_type = rank_type
        self.R_normalization = R_normalization
        self.classifier = None

    def get_params(self, deep=True):
        return {**BaseEstimator.get_params(self, deep=True), **self.bcd_args}

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def predict(self, X):
        return self.classifier.predict(X)

    def has_constraint_sum_gamma_k(self):
        return (not (self.constraint_per_gamma_k is None) or not (self.constraint_sum_gamma_k is None))

    def set_convergence_criterion_args(self, args):
        self.classifier.bcd_args = {**self.bcd_args, **zip_with_convention(args,
                                                                           Convergence_criterion)}


class Optirank(OptirankBasedClassifier):
    """Optirank classifier"""

    def __init__(self, lambda_gamma_1=0, lambda_gamma_2=0, lambda_w_1=0, lambda_w_2=0, sample_weight="balanced",
                 constraint_sum_gamma_k=None, constraint_per_gamma_k=None, rank_type="avg", R_normalization=None,
                 high_tol=True, setting_lambda_P_strategy_args=default_setting_lambda_P_strategy_args,
                 convergence_criterion_args=None, convergence_criterion_args_last=None, rounding_threshold=0,
                 max_relaxation_iter=20, tol_dist_to_border=10 ** (-10), diff_tolerance=10 ** (-15), **kwargs):
        """
        :param lambda_gamma_1: l1 regularization parameter on gamma (scalar)
        :param lambda_gamma_2: l2 regularization parameter on gamma (scalar)
        :param lambda_w_1: regularization parameter on w (scalar)
        :param lambda_w_2: regularization parameter on w (scalar)
        :param sample_weight: either "balanced" or None
        :param constraint_sum_gamma_k: either None or integer for specifying |gamma|_1 constraint
        :param constraint_per_gamma_k: either None or scalar for specifying |gamma|_1/d constraint
        :param rank_type: either "avg", "min", or "max"
        :param R_normalization: either None, "d", "sqrt(d)", "k" or "sqrt(k)"
        :param high_tol: boolean flag for setting high_tol strategy.
        In high_tol strategy, the convergence criterion of the BCD algorithm is regined at the last relaxation iteration.
        :param setting_lambda_P_strategy_args: {"class": class_name, "args": keywords-args dictionary}} for setting lambda_P_strategy
        :param convergence_criterion_args: arguments setting convergence crition (see for example absolute_delta_args in Convergence_criterion class)
        :param convergence_criterion_args_last: arguments for convergence crition at last iteration. Effective if high_tol = True.
        :param rounding_threshold: scalar. After each BCD run, gamma entries within rounding_threshold are rounded to their nearest integer values.
        :param max_relaxation_iter: integer for maximum number of relaxation iteration.
        :param tol_dist_to_border: scalar. When the l1 distance between gamma and it's 0/1 equivalent is less than tol_dist_to_border, gamma is rounded and the relaxation procedure terminates.
        :param diff_tolerance: scalar. When the l2 norm of the difference between subsequent gammas is less than diff_tolerance, the relaxation procedure terminates.
        :param kwargs: keyword-arguments destined to BCD (see usages in default_BCD_args in default_args.py)
        """

        self.convergence_criterion_args = convergence_criterion_args
        self.convergence_criterion_args_last = convergence_criterion_args_last
        self.max_relaxation_iter = max_relaxation_iter
        self.high_tol = high_tol
        if self.high_tol:
            if self.convergence_criterion_args_last is None:
                raise ValueError(
                    "convergence_criterion strict must be provided in high tol relaxation strategy")

        self.setting_lambda_P_strategy_args = setting_lambda_P_strategy_args
        self.rounding_threshold = rounding_threshold
        self.tol_dist_to_border = tol_dist_to_border
        self.diff_tolerance = diff_tolerance
        super().__init__(lambda_gamma_1, lambda_gamma_2, lambda_w_1, lambda_w_2, sample_weight, constraint_sum_gamma_k,
                         constraint_per_gamma_k, rank_type, R_normalization, **kwargs)

    def fit(self, X, y, p_init: Params = None, lambda_P_init=None, with_diagnostics=False, diagnostics_funs_dict=None):
        """
            :param X: numpy array nxd
            :param y: numpy array n
            :param p_init: parameters for warm-start (class Param)
            :param lambda_P_init: scalar representing initial lambda_P
            :param with_diagnostics: boolean flag for recording diagnostics.
            :param diagnostics_funs_dict: dictionary with name: fun entries. Each fun takes as argument a Params object.
            :return:
        """
        self.classifier = BilinearRankingClassifier(self.lambda_gamma_1, self.lambda_gamma_2, self.lambda_w_1,
                                                    self.lambda_w_2, sample_weight=self.sample_weight,
                                                    constraint_per_gamma_k=self.constraint_per_gamma_k,
                                                    constraint_sum_gamma_k=self.constraint_sum_gamma_k,
                                                    rank_type=self.rank_type, R_normalization=self.R_normalization,
                                                    **self.bcd_args)
        self.setting_lambda_P_strategy = self.setting_lambda_P_strategy_args["class"](
            **self.setting_lambda_P_strategy_args["args"])
        self.classifier.lambda_P = self.setting_lambda_P_strategy.initialize(lambda_P_init)
        self.diagnostics = []  # list with intermediate diagnostics
        self.intermediate_solutions_ = []  # list with intermediate solutions (Params_With_Loss)
        self.intermediate_lambda_Ps_ = []  # list with intermediate lambda_Ps
        self.negative_lambda_P_flags = []  # list with booleans indicating whether lambda_Ps are negative!
        self.BCD_converged_flags = []  # list with booleans indicating whether BCD has converged
        self.converged = False  # boolean indicating whether optirank has converged (True only if with less than max_relaxation_iter iterations, full 0/1 solutions with last iteration having converged!)
        self.lambda_P_iter = 0  # relaxation iteration
        self.iter = 0  # cumulative number of BCD iterations in successive relaxation iterations
        self.set_convergence_criterion_args(self.convergence_criterion_args)

        high_tol_round = False

        while (not (self.converged)):
            logging.debug("lambda_P:{}, high_tol_round:{}".format(self.classifier.lambda_P, high_tol_round))
            if self.lambda_P_iter == 0:
                initialize = p_init
                self.classifier.fit(X, y, p_init=initialize, with_diagnostics=with_diagnostics,
                                    diagnostics_funs_dict=diagnostics_funs_dict)
            else:
                initialize = self.classifier.sol
                self.classifier.fit_next_relaxation_step(p_init=initialize, with_diagnostics=with_diagnostics,
                                                         diagnostics_funs_dict=diagnostics_funs_dict)

            self.BCD_converged_flags.append(self.classifier.converged)
            self.iter += self.classifier.bcd.iter

            if with_diagnostics:
                self.diagnostics.append(deepcopy(
                    self.classifier.diagnostics))  # actually there is no need for a deepcopy because in BCD the dictionary is reset!
                self.intermediate_solutions_.append(self.classifier.sol.to_lightweight(copy=True))
                self.intermediate_lambda_Ps_.append(self.classifier.lambda_P)

            logging.debug("sum_distance_to_border: {}".format(sum_dist_to_border(self.classifier.sol.gamma)))
            logging.debug("sum_gamma: {}".format(torch.sum(self.classifier.sol.gamma)))

            # rounding/scale the solution
            if self.has_constraint_sum_gamma_k():
                self.classifier.sol.gamma = rounding_and_scaling_thresh(self.classifier.sol.gamma,
                                                                        self.classifier.sol.loss_object.constraint_sum_gamma_k,
                                                                        self.rounding_threshold)
            else:
                self.classifier.sol.gamma = rounding_thresh(self.classifier.sol.gamma, self.rounding_threshold)

            if self.classifier.lambda_P == 0:
                self.res_lambda_P_0_ = self.classifier.sol

            if self.lambda_P_iter > 0 and self.has_constraint_sum_gamma_k():
                norm_diff_sol = (self.classifier.sol - self.sol_b).gamma.norm()
                if norm_diff_sol < self.diff_tolerance:
                    logging.info(
                        "stopped iterating on lambda_P : the norm of the difference between the two last solutions ({}) is less than the tolerance ({})".format(
                            norm_diff_sol, self.diff_tolerance))
                    self.converged = False
                    break

            if self.has_constraint_sum_gamma_k():
                self.sol_b = self.classifier.sol

            if sum_dist_to_border(self.classifier.sol.gamma) < self.tol_dist_to_border:

                self.classifier.sol.gamma = torch.round(self.classifier.sol.gamma)
                # maybe because of this w is not optimal but given low tolerance should be okay.

                if self.high_tol and high_tol_round == False:
                    self.converged = False
                    high_tol_round = True
                    self.set_convergence_criterion_args(self.convergence_criterion_args_last)
                else:
                    self.converged = self.classifier.converged
                    break

            else:
                # calculating next lambda_P
                self.classifier.lambda_P, negative_lambda_P_flag = self.setting_lambda_P_strategy.next_lambda_P(
                    self.classifier.sol, self.classifier.lambda_P,
                    setting_lambda_P_epsilon_from_BCD(self.classifier.bcd))
                logging.info("Lambda_P is:{}".format(self.classifier.lambda_P))
                self.negative_lambda_P_flags.append(negative_lambda_P_flag)

            if (self.lambda_P_iter > self.max_relaxation_iter):
                logging.info("After {} lambda P iterations, did not find a full zeros ones solution".format(
                    self.max_relaxation_iter))
                break

            self.lambda_P_iter = self.lambda_P_iter + 1

    def to_lightweight(self, copy=True):
        if copy:
            args = list(inspect.signature(Optirank).parameters.keys())
            args.remove("kwargs")
            res = Optirank(**{el: getattr(self, el) for el in args})  # hope it works
        else:
            res = self

        res.classifier = self.classifier.to_lightweight(copy=copy)
        res.diagnostics = None
        res.res_lambda_P_0_ = self.res_lambda_P_0_.to_lightweight(copy=copy)
        res.converged = self.converged
        res.intermediate_solutions_ = None
        res.BCD_converged_flags = self.BCD_converged_flags
        res.negative_lambda_P_flags = self.negative_lambda_P_flags
        res.lambda_P_iter = self.lambda_P_iter
        return res


def optirank_transformer_pipe(bilinear_pipe, tol=10 ** (-5), max_iter=1000):
    """Takes as argument a scikit-learn pipeline ending with Optirank and returns a pipeline where optirank is replaced with a SubsetRankingLogRegrPipe."""
    new_pipe_steps = []

    for key, step in bilinear_pipe.named_steps.items():
        if key != "optirank":
            new_pipe_steps.append(step)
        else:
            new_pipe_step = SubsetRankingLogRegrPipe(step.classifier.sol,
                                                     label_binarizer=step.classifier.label_binarizer_, tol=tol,
                                                     max_iter=max_iter)
            new_pipe_steps.append(new_pipe_step)

    new_pipe = make_pipeline(*new_pipe_steps)

    return new_pipe
