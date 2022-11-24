# Instructions for reproducing benchmark results

For every dataset, split_index, and class of interest (which defines a One VS Rest problem), generate the preprocessed data:

```
tests_real_data/preprocess_data_single_cell_net.py dataset split_index class_of_interest
```

For every classifier, dataset, split_index, and class of interest, train and evaluate on the full parameter grid:

```
tests_real_data/training_classifier.py dataset classifier_name split_index class_of_interest -f
```

For the classifier optirank, we must refit the logistic regression coefficients with sklearn, in order to avoid discrepancies in the results due to different converging criteria.

```
tests_real_data/training_on_top.py dataset classifier_name split_index class_of_interest -f
```

📋 It is necessary to parallelize these jobs in a computing cluster to produce the results.

## Processing results

 Follow [these instructions](processing_results/README.md) to process the results and produce final tables.