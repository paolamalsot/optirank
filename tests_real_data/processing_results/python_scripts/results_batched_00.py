#makes a .csv table per dataset and class with all metrics for every parameter
#processed_results/dataset/scores_aggr/class_classifier[_detailed].csv

import argparse
import time
from tests_real_data.processing_results.funs.results_funs import agglomerate_per_class_of_interest, rank_per_class_of_interest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CV')
    parser.add_argument('dataset')
    parser.add_argument("class_of_interest", type=str)
    args = parser.parse_args()
    argv = vars(args)
    dataset_name = argv["dataset"]
    class_of_interest = argv["class_of_interest"]
    start = time.time()
    agglomerate_per_class_of_interest(dataset_name, class_of_interest)
    stop_1 = time.time()
    print("time aggl:{}".format(stop_1-start))