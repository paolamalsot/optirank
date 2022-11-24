from tests_real_data.processing_results.funs.results_constants import dataset_names, get_classes
from tests_real_data.processing_results.funs.results_dir_organisation import path_results_for_best_hp_dataset_class_of_interest, path_results_for_dataset
import pandas as pd
#agglomerate all results_batched into one per dataset! (should be fast)

if __name__ == "__main__":
    for dataset in dataset_names:
        print(dataset)
        result = []
        # pulling the best results of each classifier (the ranking was already performed)
        for class_of_interest in get_classes(dataset):
            res_per_class = pd.read_csv(path_results_for_best_hp_dataset_class_of_interest(dataset, class_of_interest))
            result.append(res_per_class)

        result = pd.concat(result)
        result.to_csv(path_results_for_dataset(dataset))