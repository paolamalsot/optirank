import numpy as np
from utilities.small_functions import loadh5array
from joblib import dump, load
import os
import pandas as pd
from utilities.singleCellNetClassifier import singleCellNetTransform
from utilities.small_functions import mkDir_if_not, str_to_node

three_datasets_names = ["Baron_Segerstolpe_Murano", "TCGA_PCAWG_met500", "MWS_TMfacs_TM10x"]
three_datasets_names_merged = [el + "_01_merged" for el in three_datasets_names]
three_datasets_names_sub_merged = [el + "_01_sub_merged" for el in three_datasets_names]
single_cell_2_datasets = ["Baron_Murano", "Baron_Segerstolpe", "MWS_TM10x", "MWS_TMfacs", "TM10x_TMfacs", "TMfacs_MWS",
                          "TM10x_MWS"]


def get_unmerged_dataset_name(name_dataset):
    if "_01_merged" in name_dataset:
        return name_dataset.replace("_01_merged", "")
    elif "_01_sub_merged" in name_dataset:
        return name_dataset.replace("_01_sub_merged", "")
    else:
        raise NotImplementedError


def get_output_dir(name_dataset):
    output_dir = mkDir_if_not(os.path.join("tests_real_data/results", name_dataset + "_dataset"))
    return output_dir


def get_CV_splits(name_dataset):
    if name_dataset == "BRCA":
        CV_dir = "datasets/BRCA"
        CV_splits = load(os.path.join(CV_dir, "CV_indices_nested.pkl"))

    if name_dataset == "TCGA":
        CV_splits = load("datasets/TCGA/CV_indices.pkl")

    # all_single_cell comparison

    if name_dataset in single_cell_2_datasets + three_datasets_names + three_datasets_names_merged + three_datasets_names_sub_merged:
        data_dir = "datasets/" + name_dataset
        CV_splits = load(os.path.join(data_dir, "CV_indices.pkl"))

    return CV_splits


def get_filter(name_dataset, filter_name):
    if filter_name not in ["single_cell_net"]:
        raise ValueError("Wrong filter")

    filter = singleCellNetTransform()
    return filter


def get_X_all(name_dataset):
    if name_dataset in ["BRCA", "TCGA", "met-500", "PCAWG"]:
        X_all = loadh5array("datasets/{}/logged_raw_cpm.h5".format(name_dataset))

    # single-cell data
    elif name_dataset in single_cell_2_datasets:
        data_dir = "datasets/" + name_dataset
        X_train = loadh5array(os.path.join(data_dir, "X_train_logged.h5"))
        X_test = loadh5array(os.path.join(data_dir, "X_test_logged.h5"))
        X_all = np.vstack((X_train, X_test))
    elif name_dataset in three_datasets_names:
        data_dir = "datasets/" + name_dataset
        X_train = loadh5array(os.path.join(data_dir, "X_train_0_logged.h5"))
        X_test = loadh5array(os.path.join(data_dir, "X_test_logged.h5"))
        X_all = np.vstack((X_train, X_test))
    elif name_dataset in three_datasets_names_sub_merged:
        unmerged_dataset_name = get_unmerged_dataset_name(name_dataset)
        data_dir = "datasets/" + unmerged_dataset_name
        X_train_0 = loadh5array(os.path.join(data_dir, "X_train_0_logged.h5"))
        X_train_1 = loadh5array(os.path.join(data_dir, "X_train_1_logged.h5"))
        X_test = loadh5array(os.path.join(data_dir, "X_test_logged.h5"))
        X_all = np.vstack((X_train_0, X_train_1, X_test))
    else:
        raise ValueError("Wrong dataset name")

    return X_all


def get_Xy_train_1(three_dataset_name):
    data_dir = "datasets/" + three_dataset_name
    if three_dataset_name != "TCGA_PCAWG_met500":
        X = loadh5array(os.path.join(data_dir, "X_train_1_small_logged.h5"))
        y = get_y_train_1_small(three_dataset_name)
    else:
        X = loadh5array(os.path.join(data_dir, "X_train_1_logged.h5"))
        y = get_y_train_1(three_dataset_name)
    return X, y


def get_y_all(name_dataset):
    if name_dataset == "BRCA":
        y_all = np.squeeze(pd.read_csv("datasets/BRCA/metadata.csv")["BRCA_hot"].to_numpy(dtype="str"))

    elif name_dataset == "Leukemia":
        data_dir = "../../data/Leukemia_Laia"
        y_all = np.squeeze(pd.read_csv(os.path.join(data_dir, "y.csv"), index_col=0).to_numpy())

    elif name_dataset in ["TCGA", "met-500", "PCAWG"]:
        y_all = pd.read_csv("datasets/{}/tissue_of_origin.csv".format(name_dataset)).to_numpy().flatten()

    # all_single_cell comparison
    elif name_dataset in single_cell_2_datasets:
        data_dir = "datasets/" + name_dataset
        y_train = np.squeeze(pd.read_csv(os.path.join(data_dir, "y_train.csv")).to_numpy())
        y_test = np.squeeze(pd.read_csv(os.path.join(data_dir, "y_test.csv")).to_numpy())
        y_all = np.concatenate((y_train, y_test))

    elif name_dataset in three_datasets_names:
        y_train = get_y_train_0(name_dataset)
        y_test = get_y_test(name_dataset)
        y_all = np.concatenate((y_train, y_test))

    elif name_dataset in three_datasets_names_sub_merged:
        unmerged_dataset_name = get_unmerged_dataset_name(name_dataset)
        y_train_0 = get_y_train_0(unmerged_dataset_name)
        y_train_1 = get_y_train_1(unmerged_dataset_name)
        y_test = get_y_test(unmerged_dataset_name)
        y_all = np.concatenate((y_train_0, y_train_1, y_test))

    return y_all


##### THREE DATASETS CONVENIENCE FUNCTIONS

def get_unmerged_dataset_dir(three_dataset_name):
    if (three_dataset_name not in three_datasets_names_sub_merged) and (three_dataset_name not in three_datasets_names):
        raise NotImplementedError
    else:
        if three_dataset_name in three_datasets_names_sub_merged:
            unmerged_dataset_name = get_unmerged_dataset_name(three_dataset_name)
            data_dir = get_dataset_dir(unmerged_dataset_name)
        else:
            data_dir = get_dataset_dir(three_dataset_name)
    return data_dir


def get_dataset_dir(dataset_name):
    data_dir = "datasets/" + dataset_name
    return data_dir


def get_y_train_0(three_dataset_name):
    if three_dataset_name not in three_datasets_names:
        raise NotImplementedError
    else:
        data_dir = get_dataset_dir(three_dataset_name)
        y_train_0 = np.squeeze(pd.read_csv(os.path.join(data_dir, "y_train_0.csv")).to_numpy())
    return y_train_0


def get_X_train_0(three_dataset_name):
    if three_dataset_name not in three_datasets_names:
        raise NotImplementedError
    else:
        data_dir = get_dataset_dir(three_dataset_name)
        X_train_0 = loadh5array(os.path.join(data_dir, "X_train_0_logged.h5"))
    return X_train_0


def exists_small_y_train_1(three_dataset_name):
    return not ((three_dataset_name not in three_datasets_names) or (three_dataset_name == "TCGA_PCAWG_met500"))


def get_y_train_1_small(three_dataset_name):
    if exists_small_y_train_1(three_dataset_name):
        data_dir = get_dataset_dir(three_dataset_name)
        y = np.squeeze(pd.read_csv(os.path.join(data_dir, "y_train_1_small.csv")).to_numpy())
    else:
        raise NotImplementedError
    return y


def get_X_train_1_small(three_dataset_name):
    if exists_small_y_train_1(three_dataset_name):
        data_dir = get_dataset_dir(three_dataset_name)
        X = loadh5array(os.path.join(data_dir, "X_train_1_small_logged.h5"))
    else:
        raise NotImplementedError
    return X


def path_indices_y_train_1_small(three_datasets_name):
    if exists_small_y_train_1(three_datasets_name):
        folder = get_dataset_dir(three_datasets_name)
        path = os.path.join(folder, "indices_y_train_1_small.pkl")
        return path
    else:
        raise NotImplementedError


def get_indices_y_train_1_small(three_datasets_name):
    return np.array(load(path_indices_y_train_1_small(three_datasets_name)))


def get_X_train_1(three_dataset_name):
    if three_dataset_name not in three_datasets_names:
        raise NotImplementedError
    else:
        data_dir = get_dataset_dir(three_dataset_name)
        return loadh5array(os.path.join(data_dir, "X_train_1_logged.h5"))


def get_y_train_1(three_dataset_name):  # or equivalently y_train_1_big
    if three_dataset_name not in three_datasets_names:
        raise NotImplementedError
    else:
        data_dir = get_dataset_dir(three_dataset_name)
        y_train_1 = np.squeeze(pd.read_csv(os.path.join(data_dir, "y_train_1.csv")).to_numpy())
    return y_train_1


def get_y_test(three_dataset_name):
    if three_dataset_name not in three_datasets_names:
        raise NotImplementedError
    else:
        data_dir = get_dataset_dir(three_dataset_name)
        y_test = np.squeeze(pd.read_csv(os.path.join(data_dir, "y_test.csv")).to_numpy())
    return y_test


def get_X_test(dataset_name):
    if dataset_name not in three_datasets_names + single_cell_2_datasets:
        raise NotImplementedError
    else:
        data_dir = get_dataset_dir(dataset_name)
        X_test = loadh5array(os.path.join(data_dir, "X_test_logged.h5"))
    return X_test


def get_dataset_index(name_dataset):
    # returns an array with 0,1,2 corresponding to the dataset of the sample!
    if name_dataset in three_datasets_names_sub_merged:
        unmerged_dataset_name = get_unmerged_dataset_name(name_dataset)
        y_train_0 = get_y_train_0(unmerged_dataset_name)
        y_train_1 = get_y_train_1(unmerged_dataset_name)
        y_test = get_y_test(unmerged_dataset_name)
        index_dataset_all = np.concatenate(
            (np.repeat(0, len(y_train_0)), np.repeat(1, len(y_train_1)), np.repeat(2, len(y_test))))
    elif name_dataset in three_datasets_names:
        y_train = get_y_train_0(name_dataset)
        y_test = get_y_test(name_dataset)
        index_dataset_all = np.concatenate((np.repeat(0, len(y_train)), np.repeat(2, len(y_test))))
    else:
        raise NotImplementedError

    return index_dataset_all


#################################


def get_setup(name_dataset, filter="ce"):
    X_all = get_X_all(name_dataset)
    CV_splits = get_CV_splits(name_dataset)
    y_all = get_y_all(name_dataset)
    filter_zero = get_filter(name_dataset, filter)
    output_dir = get_output_dir(name_dataset)

    return X_all, y_all, CV_splits, filter_zero, output_dir


def path_data(dataset_name, i_CV_split, class_name, filter_name):
    output_dir = mkDir_if_not(
        os.path.join("processed_data__{}".format(filter_name),
                     dataset_name, class_name))
    return os.path.join(output_dir, "data_{}.pkl".format(str(i_CV_split)))


def get_untransformed_data(dataset_name, i_CV_split, class_name):
    # mimics what is saved in path_data
    X_all, y_all, CV_splits, filter_zero, output_dir = get_setup(dataset_name, filter="single_cell_net")
    if dataset_name == "TCGA":
        features_list = ["genes_raw_cpm"]
        # predict on PCAWG
        X_PCAWG = get_X_all("PCAWG")
        y_PCAWG = get_y_all("PCAWG")

        # predict on met-500
        X_met500 = get_X_all("met-500")
        y_met500 = get_y_all("met-500")

    res = CV_splits[i_CV_split]
    indices_train, indices_validation, indices_test, index_test_run = res

    if dataset_name == "TCGA":
        class_ = str_to_node(class_name)
    else:
        class_ = class_name

    named_indices_splits = [("train", indices_train), ("validation", indices_validation), ("test", indices_test)]
    Xys = {split_name: (X_all[indices_split], (y_all == class_)[indices_split]) for split_name, indices_split in
           named_indices_splits}

    if dataset_name == "TCGA":
        y_PCAWG_binary = y_PCAWG == class_
        y_met500_binary = y_met500 == class_
        Xys["PCAWG"] = (X_PCAWG, y_PCAWG_binary)
        Xys["met-500"] = (X_met500, y_met500_binary)

    if dataset_name in three_datasets_names:
        X_train_1, y_train_1 = get_Xy_train_1(dataset_name)
        y_train_1_binary = y_train_1 == class_
        Xys["train_1"] = (X_train_1, y_train_1_binary)

    return Xys, index_test_run
