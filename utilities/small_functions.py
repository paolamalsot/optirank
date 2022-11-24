import h5py
import os
import logging
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_curve
import numpy as np
import traceback
import warnings
import sys
import matplotlib
import matplotlib.pyplot as plt


def loadh5array(hf_path):
    """
    Returns numpy array corresponding to h5 array stored at hf_path.
    :param hf_path: string corresponding to hf_path
    :return: numpy array
    """
    hf = h5py.File(hf_path, 'r')
    myData = hf.get('data')
    myData = np.array(myData)
    hf.close()
    return myData


def createh5array(hf_path, array):
    """
    Creates a h5 array storing a numpy array.
    :param hf_path: string corresponding to hf_path
    :param array: numpy array to store
    :return: None
    """
    # array must be a numpy array
    aggl_data_hf = h5py.File(hf_path, 'w')
    aggl_data_hf['data'] = array
    aggl_data_hf.close()


def mkDir_if_not(folder_path):
    """
    Creates folder at folder_path
    :param folder_path: path of folder to create
    :return: folder path
    """
    if not os.path.exists(folder_path) and folder_path != '':
        if not os.path.exists(os.path.dirname(folder_path)) and os.path.dirname(folder_path) != '':
            mkDir_if_not(os.path.dirname(folder_path))
        try:  # useful for multiple programs are trying to create the same folder!!
            os.mkdir(folder_path)
        except FileExistsError:
            pass
    if folder_path == '':
        raise ValueError('cannot create an empty path')
    return folder_path


def setLoglevel(loglevel, matplotlib_off=True):
    logger = logging.getLogger(__name__)
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(level=numeric_level, format='%(levelname)s:%(message)s')
    if matplotlib_off:
        matplotlib.pyplot.set_loglevel("Warning")
    return logger


def dictionaries_equal(d1, d2):
    """
    verifies if two dictionaries are equal (even if their values are np.array)
    :param d1:
    :param d2:
    :return:
    """
    truth_array = [d1[key] == d2[key] if (type(d1[key]) != np.ndarray) else (np.all(d1[key] == d2[key])) for key in
                   d1.keys()]
    return np.all(truth_array) and d1.keys() == d2.keys()


def get_list_for_key(list_of_dictionaries, key):
    return [dictionary[key] for dictionary in list_of_dictionaries]


def logging_fun(X):
    return np.log2(X + 1)


def unlogging_fun(X):
    # reverses logging (R function)
    return 2 ** X - 1


def percentage_zero(my_array, tol=0):
    return np.count_nonzero(np.isclose(my_array, 0, atol=tol)) / my_array.size


def percentage_ones(my_array, tol=0):
    return np.count_nonzero(np.isclose(my_array, 1, atol=tol)) / my_array.size


def put_to_zero_ones(X):
    if torch.is_tensor(X):
        to_torch = True
        X = X.numpy()
    else:
        to_torch = False

    indices_0 = np.argwhere(np.isclose(X, 0, atol=10 ** (-3)))
    np.put(X, indices_0, 0)
    indices_1 = np.argwhere(np.isclose(X, 1, atol=10 ** (-3)))
    np.put(X, indices_1, 1)
    if to_torch:
        X = torch.from_numpy(X)
    return X


def calculate_metrics(y_true, y_probas, y_pred):
    class_weights = compute_sample_weight("balanced", y_true)
    # accuracy
    accuracy = accuracy_score(y_true, y_pred)
    # balanced_accuracy
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    # cross_entropy_loss
    cross_entropy = log_loss(y_true, y_pred)
    # balanced cross_entropy_loss
    balanced_cross_entropy = log_loss(y_true, y_pred, sample_weight=class_weights)
    # ROC curve
    roc_curve_ = roc_curve(y_true, y_probas)

    metrics = {"accuracy": accuracy, "balanced_accuracy": balanced_accuracy, "cross_entropy": cross_entropy,
               "balanced_cross_entropy": balanced_cross_entropy, "roc_curve": roc_curve_}

    return metrics


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


def dispatch_arguments_for_classes(classes_list, kwargs):
    """
    From a dictionary with kwargs and a list of classes, dispatches the arguments to the corresponding class via inspecting their signature.
    When a parameter starts with Class_Name__argument:value
    :param classes_list: list with Class Objects
    :param kwargs: dictionary with param: value pairs
    :return: a dictionary with classes_names: kwargs
    """

    arguments = {}
    for class_ in classes_list:
        arguments_names_of_interest = []
        for argname, val in kwargs.items():
            if arg_for_class(argname, class_):
                arguments_names_of_interest.append(argname)
        new_dict = {key: kwargs[key] for key in arguments_names_of_interest}
        new_dict_stripped = strip_from_convention(new_dict, class_)
        arguments[class_.__name__] = new_dict_stripped

    return arguments


def zip_with_convention(argument_dict_or_list, class_object):
    """
    transforms key:arg into class_name__key:arg
    """
    classname = class_object.__name__.lower()
    if isinstance(argument_dict_or_list, dict):
        new_dict = {"__".join([classname, key]): arg for key, arg in argument_dict_or_list.items()}
        return new_dict
    elif isinstance(argument_dict_or_list, list):
        new_list = ["__".join([classname, key]) for key in argument_dict_or_list]
        return new_list


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  # or whatever


def strip_from_convention(argument_dict, class_object):
    """
    undoes zip_with_convention
    """
    classname = class_object.__name__.lower()
    new_dict = {remove_prefix(key, classname + "__"): arg for key, arg in argument_dict.items()}
    return new_dict


def arg_for_class(argname, class_object):
    """returns True/False depending if arg was stripped for the class"""
    classname = class_object.__name__.lower()
    return argname.startswith(classname + "__")


def str_to_node(node_str):
    if not (node_str in ['STAD_CIN', 'STAD_EBV', 'STAD_GS',
                         'STAD_MSI', 'STAD_POLE', 'Stomach_al']):
        node = node_str.replace('_', ' ')
    else:
        node = node_str

    return node


def node_to_str(node):
    node_str = str(node).replace(' ', '_')
    return node_str
