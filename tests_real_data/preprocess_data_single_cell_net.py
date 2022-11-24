# preprocess with single-cell net all datasets, for all splits (train, test, val, and also met-500/PCAWG for TCGA)
# big time and money gain
from utilities.small_functions import mkDir_if_not, str_to_node
import os
from tests_real_data.loading_data import get_setup, singleCellNetTransform, get_y_all, get_X_all, three_datasets_names, \
    get_Xy_train_1, get_untransformed_data, path_data
from joblib import dump, load
import argparse

filter_name = "single_cell_net"
filter_zero = singleCellNetTransform()


def preprocess_data(i_CV_split, dataset_name, class_name):
    Xys, index_test_run = get_untransformed_data(dataset_name, i_CV_split, class_name)

    # train single cell net
    filter_zero.fit(*Xys["train"])

    path = path_data(dataset_name, i_CV_split, class_name, filter_name)
    Xtransformedy = {}

    # pour tous les split-names:
    for split_name, (X, y_binary) in Xys.items():
        # single cell_net transform
        X_transformed = filter_zero.transform(X)
        # dump: output_name = split_name
        Xtransformedy[split_name] = (X_transformed, y_binary)

    dump((Xtransformedy, index_test_run), path, compress=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CV')
    parser.add_argument('dataset')
    parser.add_argument('CV_split', type=str)
    parser.add_argument("class_of_interest", type=str)
    args = parser.parse_args()
    argv = vars(args)

    i_CV_split = int(argv["CV_split"])
    dataset_name = argv["dataset"]
    class_name = argv["class_of_interest"]

    preprocess_data(i_CV_split, dataset_name, class_name)
