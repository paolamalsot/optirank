#finds best hyper-parameters for each dataset, class, test_run, and choosing_mode (one_std_rule or ...)
#processed_results/dataset/scores_for_best_hp/class.csv
import argparse
import time
from tests_real_data.processing_results.funs.results_funs import rank_per_class_of_interest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CV')
    parser.add_argument('dataset')
    parser.add_argument("class_of_interest", type=str)
    args = parser.parse_args()
    argv = vars(args)
    dataset_name = argv["dataset"]
    class_of_interest = argv["class_of_interest"]
    stop_1 = time.time()
    rank_per_class_of_interest(dataset_name, class_of_interest)
    stop_2 = time.time()
    print("stop time:{}".format(stop_2 - stop_1))