from sklearn.metrics import auc, precision_recall_curve
from tests_real_data.processing_results.funs.results_constants import *
from tests_real_data.processing_results.funs.results_dir_organisation import *
from tests_real_data.processing_results.investigations.supplementary_investigations_funs import retrieving_supp_metrics_bilinear_ranking_classifier
from tests_real_data.classifiers import logistic_regression_classifier_names
from itertools import cycle
from joblib import load
import logging

def agglomerate_per_class_of_interest(dataset, class_of_interest):
    for classifier in classifiers_list:
        agglomerate_results_cv_grid(dataset, class_of_interest, classifier)

def rank_per_class_of_interest(dataset, class_of_interest):
    result = []
    for choosing_mode in choosing_modes:
        for test_run_index in range(get_n_test_splits(dataset)):
            for scoring_metric in scoring_metrics:
                best_params_dict_for_all_classifiers = create_best_params_dict_for_all_classifiers(
                    dataset,
                    class_of_interest,
                    choosing_mode,
                    test_run_index,
                    scoring_metric,
                    classifiers_list)
                for index_split in pd.unique(best_params_dict_for_all_classifiers[
                                                 "CV_index"]):  # not all CV_index are present in each test split!
                    df = best_params_dict_for_all_classifiers.loc[
                        best_params_dict_for_all_classifiers.CV_index == index_split,]
                    best_params_dict_for_all_classifiers_ranked = rank_best_params_dict(df, scoring_metric,
                                                                                        classifiers_list,
                                                                                        dataset)
                    result.append(best_params_dict_for_all_classifiers_ranked)

    result = pd.concat(result)
    result.to_csv(path_results_for_best_hp_dataset_class_of_interest(dataset, class_of_interest))


def key_split_metric(splitname, metric):
    return splitname + "__" + metric

def retrieving_metrics(classifier_name, dataset_name, filter_zero_name, class_of_interest):
    """ returns param_index: metrics dictionary with avg and std """
    output_dir = get_output_dir(dataset_name)
    n_params_for_classifier = get_n_params_for_classifier(classifier_name)
    n_splits = meta_dataset.loc[meta_dataset.dataset_name == dataset_name, "n_splits"].item()

    output_df = []
    #and others for the metrics

    for index_param in range(n_params_for_classifier):
        for CV_index in np.arange(0,n_splits):
            results_dir = os.path.join(output_dir, "raw_results")
            name = "__".join([dataset_name, class_of_interest, filter_zero_name, classifier_name, str(CV_index), str(index_param)])
            filename = os.path.join(results_dir, name + "__metrics.pkl")
            metrics = load(filename)
            split_names = get_split_names(dataset_name)

            #getting additional metrics
            split_metric_dicts = {}
            for split_name in split_names:
                if "roc_auc_score" not in metrics[split_name + "_metrics"].keys():
                    if not np.any(np.isnan(metrics[split_name + "_metrics"]["roc_curve"])):
                        metrics[split_name + "_metrics"]["roc_auc_score"] = auc(*metrics[split_name + "_metrics"]["roc_curve"][0:2])
                    else:
                        metrics[split_name + "_metrics"]["roc_auc_score"] = np.nan

                #precision recall curve
                if "y_" + split_name in metrics.keys():
                    y_true_bool_or_str = metrics["y_" +split_name]
                    if y_true_bool_or_str.dtype != bool:
                        y_true_bool_or_str = y_true_bool_or_str == class_of_interest
                    y_true = y_true_bool_or_str.astype("int")
                if "y_" +split_name + "_probas" in metrics.keys():
                    y_proba = metrics["y_" +split_name + "_probas"]

                if "auprc" not in metrics[split_name + "_metrics"].keys():
                    if not np.any(np.isnan(y_proba)):
                        precision, recall, thresh = precision_recall_curve(y_true, y_proba)
                        metrics[split_name + "_metrics"]["auprc"] = auc(recall, precision)
                    else:
                        metrics[split_name + "_metrics"]["auprc"] = np.nan

                #saving the keys in a flattened format for the df
                split_metric_dicts[split_name] = {key_split_metric(split_name, key): value for key, value in metrics[split_name + "_metrics"].items() if (key!= "roc_curve")}

            split_metric_dicts_fused = {key: val for dummy_key, dicti in split_metric_dicts.items() for key, val in dicti.items()} #a voir

            test_run_index = metrics["test_run_index"]

            if classifier_name in optirank_classifier_names + optirank_derived_classifiers_list + ["ANOVA_subset_ranking_lr"]:
                converged_dict = {"converged": metrics["converged"],
                                  "per_gamma_0": metrics["per_gamma_0"],
                                  "per_w_0": metrics["per_w_0"]}

            elif classifier_name in logistic_regression_classifier_names:
                converged_dict = {"per_w_0": metrics["per_w_0"],
                                  "per_gamma_0": 0,
                                  "converged": metrics["converged"]}
            else:
                converged_dict = {"converged": True,
                                  "per_gamma_0": 0.0,
                                  "per_w_0": 0.0} #techniquement faux mais pas grave...
            if "fitting_time" in metrics.keys():
                converged_dict["fitting_time"] = metrics["fitting_time"]
            else:
                converged_dict["fitting_time"] = None

            if classifier_name == "logistic_regression":
                n_genes_dict = {"n_genes": (1-metrics["per_w_0"]) * d}
            elif classifier_name in ["random_forest", "logistic_regression_based_on_rankings", "single_cell_net"]: #TODO: I should actually add n_genes for singlecell net
                n_genes_dict = {"n_genes": d}
            else: #for bilinear classifiers it should be added in the supp_cv!
                n_genes_dict = {}

            df_to_append = pd.DataFrame({**{"classifier_name": classifier_name, "dataset_name": dataset_name,
                                              "class_of_interest": class_of_interest, "index_param": index_param,
                                              "CV_index": CV_index, "test_run_index": test_run_index},
                                              **split_metric_dicts_fused, **converged_dict, **n_genes_dict}, index = [0])

            output_df.append(df_to_append)


    output_df = pd.concat(output_df).reset_index()
    return output_df

def get_best_param_index(metrics_df, scoring_metric, choosing_mode = "average_scoring", only_converged = False):
    #get the index_param corresponding the best classifier parameters for choosing mode! (in validation splits!)

    orientation = meta_metrics.loc[meta_metrics["name"] == scoring_metric,"best_orientation"].item()
    scoring_metric_validation = key_split_metric(validation_split_name, scoring_metric)

    if choosing_mode != "one_standard_error_rule":
        subset_of_interest = metrics_df[["index_param", "converged", scoring_metric_validation, "n_genes"]].groupby(["index_param"]).agg(["mean", "sem", "std"]).reset_index()
    else:
        subset_of_interest = metrics_df[["index_param", "n_genes", "per_gamma_0", "per_w_0", "converged", scoring_metric_validation]].groupby(["index_param"]).agg(["mean", "sem", "std"]).reset_index()
        subset_of_interest[("per_gamma_0+per_w_0", "mean")] = subset_of_interest[("per_gamma_0", "mean")] + subset_of_interest[("per_w_0", "mean")]
        subset_of_interest[("per_gamma_0+per_w_0", "sem")] = np.sqrt(subset_of_interest[("per_gamma_0", "sem")]**2 + \
                                                              subset_of_interest[("per_w_0", "sem")]**2) #treat them as two independent variables

    if only_converged:
        subset_of_interest = subset_of_interest.loc[subset_of_interest[("converged", "mean")] == 1]

    if choosing_mode == "average_scoring":
        rules = [(orientation, (scoring_metric_validation, "mean")),  ("min", (scoring_metric_validation, "sem")), ("min", ("n_genes", "mean")), ("min", ("n_genes", "sem"))]
        # if ("per_gamma_0", "mean") in subset_of_interest.columns:
        #     rules.append(("max", ("per_gamma_0", "mean")))
        return find_best_index(subset_of_interest, rules)

    elif choosing_mode == "one_standard_error_rule":

        if orientation == "max":
            best_value = subset_of_interest[(scoring_metric_validation, "mean")].max()
            idx_max = subset_of_interest[(scoring_metric_validation, "mean")].idxmax()
            standard_error = subset_of_interest.loc[idx_max, [(scoring_metric_validation, "sem")]].item()
            cutoff = best_value - standard_error
            eligible_subset = subset_of_interest.loc[subset_of_interest[(scoring_metric_validation, "mean")] >= cutoff,]

        if orientation == "min":
            best_value =  subset_of_interest[(scoring_metric_validation, "mean")].min()
            idx_min = subset_of_interest[(scoring_metric_validation, "mean")].idxmin()
            standard_error = subset_of_interest.loc[idx_min, [(scoring_metric_validation, "sem")]].item()
            cutoff = best_value + standard_error
            eligible_subset = subset_of_interest.loc[subset_of_interest[(scoring_metric_validation, "mean")] <= cutoff,]
        #("max",("per_gamma_0+per_w_0", "mean")), ("max", ("per_gamma_0", "mean"))
        return find_best_index(eligible_subset, [("min", ("n_genes", "mean")), (orientation, (scoring_metric_validation, "mean")), ("min", (scoring_metric_validation, "sem")), ("min", ("n_genes", "sem"))])

def get_orientation_metric(metric):
    #returns min or max depending on which is the preferred orientation of the metric. For instance, it would return max for balanced accuracy
    return meta_metrics.loc[meta_metrics.name == metric, "best_orientation"].item()

def rank_result(result, split_names = ["train", "test", "validation"]):
    """
    Rank classifiers according to performance inside every CV run inside every validation_run for every class_of_interest and choosing mode...!
    :param result: result df
    :return:
    """
    for split_name in split_names:
        result["rank_" + split_name] = np.nan
        for index, row in meta_metrics.iterrows():
            metric = row["name"]
            best_orientation = row["best_orientation"]
            if best_orientation == "max":
                ascending = False
            elif best_orientation == "min":
                ascending = True
            else:
                continue


            rank_computed = result.loc[
                result["metric_name"] == metric, ["class_of_interest", "choosing_mode", "test_run_index", "CV_index"
                                                  , "metric_name", split_name + "_value"]].groupby(
                ["class_of_interest", "choosing_mode", "test_run_index", "metric_name", "CV_index"]).rank(
                ascending=ascending).copy()
            result.loc[result["metric_name"] == metric, ["rank_" + split_name]] = rank_computed[split_name + "_value"]

    return result

def find_best_index(df, rules):

    """
    Created this function to avoid potential bias coming from ties when doing idx_max(metric).
    Here rules are applied sequentially to resolve ties and find the best index.
    Prints if there are still ties un-resolved
    :param df: pandas
    :param rules: list of rules to apply sequencially. Each rule is expressed in a tuple of the form ("min" or "max", "field")
    :return: best param index
    """
    unresolved_ties = True
    funs = {"min": np.min, "max": np.max}
    df_subselection = df
    for orientation, field in rules:
        best_value = funs[orientation](df_subselection[field].values)
        #getting the index corresponding to the best value
        indices = df_subselection.loc[df_subselection[field] == best_value].index
        df_subselection = df_subselection.loc[indices]
        if len(indices) == 1:
            unresolved_ties = False
            break
    if unresolved_ties:
        logging.info("There are still unresolved ties")
    return df_subselection["index_param"].to_numpy()[0]



def agglomerate_results_cv_grid(dataset, class_of_interest, classifier):
    if classifier in classifiers_with_supp_metrics:
        metrics_dict_usual = retrieving_metrics(classifier, dataset, filter_zero_name, class_of_interest)
        supp_dict = retrieving_supp_metrics_bilinear_ranking_classifier(dataset, filter_zero_name, class_of_interest,
                                                                        classifier)
        on_columns = ["class_of_interest", "classifier_name", "dataset_name", "index_param", "CV_index"]
        total_metrics_dict = metrics_dict_usual.merge(supp_dict, on=on_columns)
        total_metrics_dict.to_csv(results_total_cv_grid_path(dataset, class_of_interest, classifier))
    else:
        metrics_dict = retrieving_metrics(classifier, dataset, filter_zero_name, class_of_interest)
        metrics_dict.to_csv(results_cv_grid_path(dataset, class_of_interest, classifier))


def create_best_params_dict_for_all_classifiers(dataset, class_of_interest, choosing_mode, test_run_index, scoring_metric, classifiers_list):
    #returns a pandas dataframe with for each classifiers:
    #the best performing (in validation split) index param (in terms of choosing mode , scoring metric)
    split_names = get_split_names(dataset)
    result = []

    for classifier in classifiers_list:
        if not (classifier in optirank_classifier_names + logistic_regression_classifier_names) and choosing_mode == "one_standard_error_rule":
            #no reduced complexity possible (number of genes fixed)
            effective_choosing_mode = "average_scoring"
        else:
            effective_choosing_mode = choosing_mode

        if classifier in classifiers_with_supp_metrics:
            cv_df = pd.read_csv(results_total_cv_grid_path(dataset, class_of_interest, classifier))
        else:
            cv_df = pd.read_csv(results_cv_grid_path(dataset, class_of_interest, classifier))

        metrics_dict_subset = cv_df.loc[cv_df.test_run_index == test_run_index,]
        best_param_index = get_best_param_index(metrics_dict_subset, scoring_metric, effective_choosing_mode)
        best_param_dict = metrics_dict_subset.loc[metrics_dict_subset["index_param"] == best_param_index,]

        df_res = pd.DataFrame(
            columns=["classifier_name", "index_param", "metric_name", "test_value", "validation_value",
                     "test_run_index", "converged", "class_of_interest", "choosing_mode", "fitting_time"])
        for split_name in split_names:
            df_res[split_name + "_value"] = best_param_dict[key_split_metric(split_name, scoring_metric)]

        df_res["converged"] = best_param_dict["converged"]
        df_res["CV_index"] = best_param_dict["CV_index"]
        df_res["classifier_name"] = classifier
        df_res["index_param"] = best_param_index
        df_res["metric_name"] = scoring_metric
        df_res["test_run_index"] = test_run_index
        df_res["class_of_interest"] = class_of_interest
        df_res["choosing_mode"] = choosing_mode
        df_res["fitting_time"] = best_param_dict["fitting_time"]

        if classifier in all_optirank_classifiers:
            df_res["per_gamma_0"] = best_param_dict["per_gamma_0"]
            df_res["per_w_0"] = best_param_dict["per_w_0"]

        if classifier in logistic_regression_classifier_names:
            df_res["per_w_0"] = best_param_dict["per_w_0"]

        df_res["n_genes"] = best_param_dict["n_genes"]

        result.append(df_res)
    result = pd.concat(result)
    return result

def rank_best_params_dict(best_params_dict_for_all_classifiers, scoring_metric, classifiers_list, dataset_name):
    """
    Rank column scoring_metric for all classifiers.
    :param best_params_dict_for_all_classifiers: df with one line per classifier
    :param scoring_metric:
    :param classifiers_list: list of all classifiers which should be present
    :return: df with one column scoring_metric__rank added, per split. NB: keep in mind that the classifier was added for its performance on the test-split!
    """

    #checking each classifier is present only once
    for classifier_name in classifiers_list:
        corresponding_rows = best_params_dict_for_all_classifiers.loc[best_params_dict_for_all_classifiers.classifier_name == classifier_name,]
        if corresponding_rows.shape[0] == 0:
            raise ValueError("{} not present in dataframe".format(classifier_name))
        elif corresponding_rows.shape[0] > 1:
            raise ValueError("{} present multiple times in dataframe".format(classifier_name))

    #rank and add the corresponding column
    split_names = get_split_names(dataset_name)
    result = best_params_dict_for_all_classifiers.copy()

    for split_name in split_names:
        result["rank_" + split_name] = np.nan
        best_orientation = meta_metrics.loc[meta_metrics.name == scoring_metric, "best_orientation"].item()

        if best_orientation == "max":
            ascending = False
        elif best_orientation == "min":
            ascending = True
        else:
            raise ValueError("The score for the metric {} cannot be ranked!".format(scoring_metric))

        rank_computed = result[split_name + "_value"].rank(ascending=ascending).copy()
        result.loc[result["metric_name"] == scoring_metric, ["rank_" + split_name]] = rank_computed

    return result

def agglomerate_all():
    for dataset in dataset_names:
        for class_of_interest in get_classes(dataset):
            print(dataset)
            for classifier in classifiers_list:
                agglomerate_results_cv_grid(dataset, class_of_interest, classifier)



def rank_all():
    for dataset in dataset_names:
        print(dataset)
        result = []
        # pulling the best results of each classifier and perform the ranking
        for class_of_interest in get_classes(dataset):
            for choosing_mode in choosing_modes:
                for test_run_index in range(get_n_test_splits(dataset)):
                    for scoring_metric in scoring_metrics:
                        best_params_dict_for_all_classifiers = create_best_params_dict_for_all_classifiers(
                            dataset,
                            class_of_interest,
                            choosing_mode,
                            test_run_index,
                            scoring_metric,
                            classifiers_list)
                        for index_split in pd.unique(best_params_dict_for_all_classifiers[
                                                         "CV_index"]):  # not all CV_index are present in each validation split!
                            df = best_params_dict_for_all_classifiers.loc[
                                best_params_dict_for_all_classifiers.CV_index == index_split,]
                            best_params_dict_for_all_classifiers_ranked = rank_best_params_dict(df, scoring_metric,
                                                                                                classifiers_list,
                                                                                                dataset)
                            result.append(best_params_dict_for_all_classifiers_ranked)
        result = pd.concat(result)
        result.to_csv(path_results_for_dataset(dataset))


def cv_selection_fun_indicator(res, choosing_modes_suffixes):
    #indicates in which choosing mode a param was chosen : both, avg_scoring, 1_std, none
    res = {"selected_" + suffix: res["is_best_param" + suffix] for suffix in choosing_modes_suffixes}
    if np.all(list(res.values())):
        return "both"
    elif np.any(list(res.values())):
        return np.array(list(res.keys()))[np.array(list(res.values()))].item()
    else:
        return "none"

def put_all_together(classifiers_list, metric_name, choosing_mode):
    #takes only the best hyper-parameter
    #for all classifiers above
    #for all datasets in the experiment
    results_list = []
    for dataset in dataset_names:
        res = pd.read_csv(path_results_for_dataset(dataset))
        res_selection = res.loc[res.classifier_name.isin(classifiers_list) & (res.metric_name == metric_name) & (res.choosing_mode == choosing_mode)]
        res_selection["overfit_margin"] = res_selection[validation_split_name + "_value"] - res_selection[test_split_name + "_value"]
        res_selection["dataset_name"] = dataset
        results_list.append(res_selection)
    res = pd.concat(results_list)
    return res