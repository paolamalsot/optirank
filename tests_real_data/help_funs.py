from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np

def calculate_metrics(y_true, y_probas, y_pred):
    class_weights = compute_sample_weight("balanced", y_true)
    #accuracy
    accuracy = accuracy_score(y_true, y_pred)
    #balanced_accuracy
    balanced_accuracy = balanced_accuracy_score(y_true,y_pred)
    #cross_entropy_loss

    labels = np.array([True, False])
    cross_entropy = log_loss(y_true, y_pred, labels=labels)
    #balanced cross_entropy_loss
    balanced_cross_entropy = log_loss(y_true, y_pred, sample_weight = class_weights, labels = labels)

    if not(np.any(np.isnan(y_probas))):
        #ROC curve
        roc_curve_ = roc_curve(y_true, y_probas)
        #auc
        roc_auc_score = auc(*roc_curve_[0:2])
    else:
        roc_curve_ = np.nan
        roc_auc_score = np.nan


    metrics = {"accuracy":accuracy, "balanced_accuracy":balanced_accuracy, "cross_entropy":cross_entropy,
               "roc_auc_score":roc_auc_score, "balanced_cross_entropy":balanced_cross_entropy, "roc_curve":roc_curve_}

    return metrics

averageable_metrics = ["accuracy", "balanced_accuracy", "cross_entropy", "roc_auc_score", "balanced_cross_entropy"]

metrics_dict = {}