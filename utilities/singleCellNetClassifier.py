import warnings
import pySingleCellNet as pySCN
from pySingleCellNet.utils import *
from pySingleCellNet.tsp_rf import *
from sklearn import base
from utilities.small_functions import unlogging_fun, logging_fun
import anndata
import scipy
import time
from utilities.small_functions import dictionaries_equal
import logging

warnings.filterwarnings('ignore')  # si les pros l'ont fait j'ai aussi le droit


# pysingle cell transform
# unlog the data
# pysingle cell transforms

def getAnnData(X, y=None):
    if y is None:  # return a dummy y
        y = np.zeros(X.shape[0], dtype=int).astype(str)
    return anndata.AnnData(X=scipy.sparse.csr_matrix(X), obs=pd.DataFrame(data={"obs": y.astype(str)}, dtype=str),
                           var=pd.DataFrame(data={"var": np.arange(X.shape[1]).tolist()}))


def expTNormTransform(X, y, counts_per_cell_after=1e4, scaleMax=10, limitToHVG=False):
    # unlog the data
    X_copy = X.copy()
    X_copy = unlogging_fun(X_copy)
    aTrain = getAnnData(X_copy, y)
    # pysingle cell transforms
    stTrain = aTrain.obs

    adNorm = aTrain.copy()
    # normalize per cell
    sc.pp.normalize_per_cell(adNorm, counts_per_cell_after=counts_per_cell_after)
    # logging
    sc.pp.log1p(adNorm)

    print("HVG")
    if limitToHVG:
        sc.pp.highly_variable_genes(adNorm, min_mean=0.0125, max_mean=4, min_disp=0.5)
        adNorm = adNorm[:, adNorm.var.highly_variable]

    sc.pp.scale(adNorm, max_value=scaleMax)
    expTnorm = pd.DataFrame(data=adNorm.X, index=adNorm.obs.index.values, columns=adNorm.var.index.values)
    expTnorm = expTnorm.loc[stTrain.index.values]

    return expTnorm


class singleCellNetTransform(base.BaseEstimator, base.TransformerMixin):
    fitting_parameters = ["nTopGenes", "nTopGenePairs", "nRand", "nTrees", "stratify", "counts_per_cell_after",
                          "scaleMax", "limitToHVG"]

    def __init__(self, nTopGenes=500, nTopGenePairs=100, nRand=100, nTrees=1000, stratify=True,
                 counts_per_cell_after=1e4, scaleMax=10, limitToHVG=False, time_economy=False):
        super(singleCellNetTransform, self).__init__()
        self.nTopGenes = nTopGenes
        self.nTopGenePairs = nTopGenePairs
        self.nRand = nRand
        self.nTrees = nTrees
        self.stratify = stratify
        self.counts_per_cell_after = counts_per_cell_after
        self.scaleMax = scaleMax
        self.limitToHVG = limitToHVG
        self.time_economy = time_economy

    def fit(self, X, y=None, **fit_params):
        start = time.time()

        if self.time_economy:
            new_params = {key: self.get_params()[key] for key in singleCellNetTransform.fitting_parameters}

            if hasattr(self, "fitted_"):
                if (np.all(X == self.X_)) and (dictionaries_equal(self.fitted_params_, new_params)) and (
                np.all(y == self.y_)):
                    stop = time.time()
                    logging.debug("time-economy: rentability mode:{}".format(stop - start))
                    return self

        # unlog the data
        X_copy = X.copy()
        X_copy = unlogging_fun(X_copy)
        aTrain = getAnnData(X_copy, y)
        # pysingle cell transforms
        stTrain = aTrain.obs

        expRaw = pd.DataFrame(data=aTrain.X.toarray(), index=aTrain.obs.index.values, columns=aTrain.var.index.values)
        expRaw = expRaw.loc[stTrain.index.values]

        expTnorm = expTNormTransform(X, y, counts_per_cell_after=self.counts_per_cell_after, scaleMax=self.scaleMax,
                                     limitToHVG=self.limitToHVG)

        print("Matrix normalized")
        ### cgenesA, grps, cgenes_list = findClassyGenes(expTnorm,stTrain, dLevel = dLevel, topX = nTopGenes)
        self.cgenesA, self.grps, self.cgenes_list = findClassyGenes(expTnorm, stTrain, dLevel="obs",
                                                                    topX=self.nTopGenes)
        print("There are ", len(self.cgenesA), " classification genes\n")

        if self.time_economy:
            self.X_ = X.copy()
            self.y_ = y.copy()
            self.fitted_params_ = new_params
            self.fitted_ = True

        stop = time.time()
        logging.debug("time-economy: calculation: {}".format(stop - start))

        return self  # as in the end it is what is used in the rf_classifier

    def transform(self, X):
        # keep the data logged
        X_copy = X.copy()
        ann_Data = getAnnData(X_copy)

        # pysingle cell transforms
        expDat = pd.DataFrame(data=ann_Data.X.toarray(), index=ann_Data.obs.index.values,
                              columns=ann_Data.var.index.values)
        expDat = expDat.reindex(labels=self.cgenesA, axis='columns', fill_value=0)

        # select the selected genes!
        return expDat.to_numpy()

    def to_lightweight(self, copy=True):
        if copy:
            res = singleCellNetTransform(self.nTopGenes, self.nTopGenePairs, self.nRand, self.nTrees, self.stratify,
                                         self.counts_per_cell_after, self.scaleMax, self.limitToHVG, self.time_economy)
        else:
            res = self

        # to_keep_or_not!
        if self.time_economy:
            res.X_ = None
            res.y_ = None
            res.fitted_params_ = None
            res.fitted_ = False

        # to keep
        res.cgenesA = self.cgenesA

        return res


# pysingle cell classifier
class singleCellNetWholePipeline(base.BaseEstimator, base.ClassifierMixin):
    def __init__(self, nTopGenes=100, nTopGenePairs=100, nRand=100, nTrees=1000, stratify=True,
                 counts_per_cell_after=1e4, scaleMax=10, limitToHVG=False):  # stratify is the same as balanced!
        super(singleCellNetWholePipeline).__init__()
        self.nTopGenes = nTopGenes
        self.nTopGenePairs = nTopGenePairs
        self.nRand = nRand
        self.nTrees = nTrees
        self.stratify = stratify
        self.counts_per_cell_after = counts_per_cell_after
        self.scaleMax = scaleMax
        self.limitToHVG = limitToHVG

    def fit(self, X, y):
        # we assume the data has NOT already gotten through pySingleCell transform raw data with correct genes!
        # unlog the data
        X_copy = X.copy()
        self.nRand = min(X.shape[0], self.nRand)  # otherwise error
        X_copy = unlogging_fun(X_copy)
        aTrain = getAnnData(X_copy, y)
        self.cgenesA_, self.xpairs_, self.tspRF_ = pySCN.scn_train(aTrain, nTopGenes=self.nTopGenes, nRand=self.nRand,
                                                                   counts_per_cell_after=self.counts_per_cell_after,
                                                                   scaleMax=self.scaleMax, nTrees=self.nTrees,
                                                                   nTopGenePairs=self.nTopGenePairs, dLevel="obs",
                                                                   stratify=self.stratify, limitToHVG=self.limitToHVG)

        self.classes_, y = np.unique(y, return_inverse=True)
        return self

    def predict_proba(self, X):
        # unlog the data
        X_copy = X.copy()
        X_copy = unlogging_fun(X_copy)
        X_ann = getAnnData(X_copy)
        # we assume the data has NOT already gotten through pySingleCell transform
        adVal = pySCN.scn_classify(X_ann, self.cgenesA_, self.xpairs_, self.tspRF_,
                                   nrand=0)  # adval.X contient les scores
        probas = {}
        probas[True] = adVal.X[:, np.isin(adVal.var_names, ["True"])]
        probas[False] = np.sum(adVal.X[:, np.isin(adVal.var_names, ["False", "rand"])], axis=1, keepdims=True)
        probas_rand_merged = np.hstack([probas[key] for key in self.classes_])
        return probas_rand_merged  # a checquer que c'est le bon truc, et que c'est dans le bon ordre!

    def predict(self, X):
        D = self.predict_proba(X)
        return self.classes_[np.argmax(D, axis=1)]
