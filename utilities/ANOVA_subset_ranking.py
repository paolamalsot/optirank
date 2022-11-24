# This class implements the adaptive rank transformation used in classifier ANOVA_subset_ranking_lr

import sklearn.base as base
import scipy
import time
import logging
import torch
from utilities.optirank.ranking_multiplication import ranking_transformation
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd


def calculate_F_ANOVA(X, label, dataset):
    """
    for every feature, compute the F_value corresponding to the effect of the dataset source
    (not taking into account the interaction term).
    :param X: n x d numpy array
    :param label: n numpy array with labels.
    :param dataset: n numpy array with dataset name corresponding to each observation.
    :return: F_value for each feature (the bigger the F, the greater the shift between datasets).
    """

    F_dataset = []
    for gene_index in range(X.shape[1]):
        df = pd.DataFrame({"label": label, "dataset": dataset, "expr": X[:, gene_index].flatten()})
        model = ols('expr ~ C(label) + C(dataset) + C(label):C(dataset)', data=df).fit()
        stats = sm.stats.anova_lm(model, typ=2)
        F = stats.loc["C(dataset)", "F"]
        F_dataset.append(F)

    return np.array(F_dataset)


def merge_two_datasets(X, y, X_other, y_other, mask_dataset_other):
    """
    Returns X, y, dataset.
    X is a numpy array containing the observations for two datasets, y contains
    the corresponding labels, and dataset is a str numpy array with "0" or "1" that indicate to which dataset the
    observations belong.
    There are two modes of usage:
    If X_other and y_other are None, and mask_dataset_other is provided, it implicitely indicates that the two datasets
    are already mixed in X,y. Otherwise, if X_other, y_other are given, and mask_dataset_other is None, the two datas
    are merged in a new array.
    """
    if X_other is not None:
        X_all = np.vstack([X, X_other])
        y_all = np.concatenate([y, y_other])
        dataset = np.concatenate([np.repeat("0", len(y)), np.repeat("1", len(y_other))])
        return X_all, y_all, dataset
    else:
        dataset = np.repeat("0", len(y))
        dataset[mask_dataset_other] = "1"
        return X, y, dataset


class ANOVA_subset_ranking(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, fitted=False, X=None, y=None, sum_gamma=None, perc_gamma=None, time_economy=False, X_other=None,
                 y_other=None, mask_dataset_other=None):
        """
        transformer that selects the features that are the least influenced by the dataset source, based on a two way
        ANOVA test that estimates the dependence of each feature on the dataset.
        To function, X,y values must be provided for two dataset-sources, and the dataset effect is estimated.
        There are two modes of use:
            1) with X_other, y_other
            2) with mask_dataset_other, which indicate which samples in X,y belong to the other dataset

        These two modes permit to include (or not) the other dataset in the transformed data.

        :param fitted:
        :param X: nxd numpy array with data
        :param y: n numpy array with label
        :param sum_gamma: integer indicating how many features to select as ranking reference.
        :param perc_gamma: float indicating which percentage of features to use as ranking reference.
        :param time_economy: if True, X and y are cached, and the F-values are not re-calculated for subsequent values of hyperparameters gamma.
        :param X_other: nxd numpy array with data
        :param y_other: n numpy array with label
        :param mask_dataset_other: boolean mask that selects the "secondary dataset" samples.
        """

        super(ANOVA_subset_ranking, self).__init__()
        self.sum_gamma = sum_gamma
        self.perc_gamma = perc_gamma
        self.time_economy = time_economy
        self.fitted = fitted
        self.X = X
        self.y = y
        self.X_other = X_other
        self.y_other = y_other
        self.mask_dataset_other = mask_dataset_other

    def fit(self, X, y):

        n_genes = X.shape[1]

        # sum_gamma-perc_gamma agreement
        if self.sum_gamma is None:
            self.sum_gamma = int(self.perc_gamma * n_genes)

        if isinstance(y, list):
            y = np.array(y)

        if self.time_economy:
            start = time.time()
            if self.fitted == False or not (np.all(X == self.X)) or not (
            np.all(y == self.y)):  # hope it doesn't throw an error when X is not fitted
                # storing X and parameters
                self.X = X
                self.y = y

                # calculate F values
                X_merged, y_merged, dataset_merged = merge_two_datasets(X, y, self.X_other, self.y_other,
                                                                        self.mask_dataset_other)
                self.F_ = calculate_F_ANOVA(X_merged, y_merged, dataset_merged)
                stop = time.time()
                logging.debug('__time_economy:calculation:{}'.format(stop - start))
            else:
                stop = time.time()
                logging.debug('__time_economy:rentability:{}'.format(stop - start))
        else:
            X_merged, y_merged, dataset_merged = merge_two_datasets(X, y, self.X_other, self.y_other,
                                                                    self.mask_dataset_other)
            self.F_ = calculate_F_ANOVA(X_merged, y_merged, dataset_merged)

        self.fitted = True
        return self

    def transform(self, X):
        n_genes = X.shape[1]
        ranking_F_indices = self.F_.argsort()

        selection_indices = ranking_F_indices[0:np.min([self.sum_gamma, n_genes])]

        # converting reference genes to binary gamma
        gamma = np.zeros(n_genes, dtype="bool")
        gamma[selection_indices] = True
        self.gamma_ = torch.Tensor(gamma)

        # ranking X_expr wrt gamma (in "avg" mode)
        X_ranked = ranking_transformation(X, self.gamma_, "avg", "d")

        return X_ranked

    def to_lightweight(self, copy=False):
        if copy:
            new_lightweight = ANOVA_subset_ranking(fitted=True, X=None, y=None, sum_gamma=self.sum_gamma,
                                                   perc_gamma=self.perc_gamma, time_economy=self.time_economy,
                                                   X_other=None, y_other=None)
            new_lightweight.gamma_ = self.gamma_
            new_lightweight.F_ = self.F_
            return new_lightweight
        else:
            self.X_other = None
            self.y_other = None
            self.X = None
            self.y = None
            return self
