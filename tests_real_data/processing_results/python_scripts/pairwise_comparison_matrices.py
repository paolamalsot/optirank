#pairwise comparisons between classifiers
#Student paired t-test or Wilcoxon signed rank test
#per_class or per_dataset
#processed_results/dataset/pairwise_comparisons/[per_class;per_dataset]_[student;wilcoxon]

from tests_real_data.processing_results.funs.results_funs import *
import numpy as np
import itertools
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel

def is_1_better_than_2(x, c_1, c_2, orientation, val):
    res_c1 = x.loc[x.classifier_name == c_1, val].item()
    res_c2 = x.loc[x.classifier_name == c_2, val].item()
    diff = res_c1 - res_c2
    if orientation == "max":
        return pd.Series({"win": res_c1 > res_c2, "diff": diff})
    elif orientation == "min":
        return pd.Series({"win": res_c1 < res_c2, "diff": diff})
    else:
        raise NotImplementedError

def scores_side_by_side(x, c_i, c_j, val):
    res_ci = x.loc[x.classifier_name == c_i, val].item()
    res_cj = x.loc[x.classifier_name == c_j, val].item()
    return pd.Series({"classifier_i_score": res_ci, "classifier_j_score": res_cj})


def opposite_sign(sign):
    if sign == "+":
        return "-"
    elif sign == "-":
        return "+"
    elif sign == "":
        return ""
    else:
        raise NotImplementedError

def get_percentage_time_with_color(perc_times, sign):
    main_str = '{:.0f}'.format(perc_times*100)
    if sign == "+":
        return r"$\zb{" + main_str + "}$"
    elif sign == "-":
        return r"$\zw{" + main_str + "}$"
    elif sign == "":
        return r"$" + main_str + r"$"
    else:
        raise NotImplementedError

def get_sign_significance(diff, orientation, level = 0.05):
    if np.all(diff == 0):
        return "", 0.0
    w, p_two_sided = wilcoxon(diff)
    if p_two_sided > level:
        return "", p_two_sided
    else:
        w, p_greater = wilcoxon(diff, alternative = "greater")
        w, p_less = wilcoxon(diff, alternative = "less")
        if p_greater < level and p_less > level:
            p = p_greater
            if orientation == "max":
                sign = "+"
            else:
                sign = "-"
        elif p_less < level and p_greater > level:
            p = p_less
            if orientation == "max":
                sign = "-"
            else:
                sign = "-"
        else:
            raise NotImplementedError
    return sign, p

def get_sign_significance_student_t(scores_side_by_side, orientation, level = 0.05):
    if np.all(scores_side_by_side["classifier_i_score"].to_numpy() - scores_side_by_side["classifier_j_score"].to_numpy() == 0):
        return "", 0
    stat_object = ttest_rel(scores_side_by_side["classifier_i_score"].to_numpy(), scores_side_by_side["classifier_j_score"].to_numpy())
    p_two_sided = stat_object.pvalue
    if p_two_sided > level:
        return "", p_two_sided
    else:
        stat_object = ttest_rel(scores_side_by_side["classifier_i_score"], scores_side_by_side["classifier_j_score"], alternative = "greater")
        p_greater = stat_object.pvalue
        stat_object = ttest_rel(scores_side_by_side["classifier_i_score"], scores_side_by_side["classifier_j_score"], alternative = "less")
        p_less = stat_object.pvalue
        if p_greater < level and p_less > level:
            p = p_greater
            if orientation == "max":
                sign = "+"
            else:
                sign = "-"
        elif p_less < level and p_greater > level:
            p = p_less
            if orientation == "max":
                sign = "-"
            else:
                sign = "-"
        else:
            raise NotImplementedError
    return sign, p

comparison_test_suffix = {"Wilcoxon": "_wilcoxon", "Student": "_student"}

def get_matrix_path(dataset, comparison_test, class_of_interest):
    comparison_suffix = comparison_test_suffix[comparison_test]
    if class_of_interest is None:
        dir = mkDir_if_not(os.path.join(processed_results_dir(dataset), "pairwise_comparison", "pairwise_comparison" + comparison_suffix))
        return os.path.join(dir, "results_{}.csv".format(dataset))
    else:
        dir = mkDir_if_not(os.path.join(processed_results_dir(dataset), "pairwise_comparison", "pairwise_comparison_per_class{}".format(comparison_suffix)))
        return os.path.join(dir, "results_{}.csv".format(class_of_interest))

def get_matrix_path_latex(dataset, comparison_test, class_of_interest):
    comparison_suffix = comparison_test_suffix[comparison_test]
    if class_of_interest is None:
        dir = mkDir_if_not(os.path.join(processed_results_dir(dataset),
                           "pairwise_comparison", "pairwise_comparison"+ comparison_suffix))
        return os.path.join(dir, "results_{}.txt".format(dataset))
    else:
        dir = mkDir_if_not(
            os.path.join(processed_results_dir(dataset), "pairwise_comparison", "pairwise_comparison_per_class{}".format(comparison_suffix)))
        return os.path.join(dir, "results_{}.txt".format(class_of_interest))

def make_comparison_matrix(dataset, classifiers_list, metric, choosing_mode, label = None):
    #if label = None, aggregates all classes

    orientation = get_orientation_metric(metric)

    #load dataset...
    results = pd.read_csv(path_results_for_dataset(dataset))
    if label is None:
        df = results.loc[(results.metric_name == metric) & (results.choosing_mode == choosing_mode),]
    else:
        df = results.loc[(results.metric_name == metric) & (results.choosing_mode == choosing_mode) & (results.class_of_interest.astype(str) == label),]
    splits = [(test_split_name, dataset)] #WARNING: was validation before
    if dataset == "TCGA":
        splits = splits + [("PCAWG", "PCAWG"), ("met-500", "met-500")]

    n_classifiers = len(classifiers_list)

    for comparison_test in ["Wilcoxon", "Student"]:
        for split, dataset in splits:
            val = split + "_value"
            res = np.empty((n_classifiers, n_classifiers),dtype="object")
            for i in range(n_classifiers):
                for j in range(i+1, n_classifiers):
                    classifier_i = classifiers_list[i]
                    classifier_j = classifiers_list[j]
                    classifiers_comparison = [classifier_i, classifier_j]
                    df_comparison = df.loc[df.classifier_name.isin(classifiers_comparison), [val, "class_of_interest", "CV_index", "classifier_name"]]

                    #add at position j i the sign of significance of the wilcoxon test for j versus i
                    if comparison_test == "Wilcoxon":
                        df_comparison_gb = df_comparison[[val, "classifier_name", "class_of_interest", "CV_index"]].groupby(["class_of_interest", "CV_index"]).apply(is_1_better_than_2, c_1 = classifier_j, c_2 = classifier_i, orientation = orientation, val = val).reset_index()
                        sign, p = get_sign_significance(df_comparison_gb["diff"].values, orientation)
                        # count how many times classifier_i > classifier_j across classes and cv_folds (not necessarily significantly!)
                        perc_times = float(np.count_nonzero(df_comparison_gb["win"])) / len(df_comparison_gb["win"])
                    elif comparison_test == "Student":
                        df_comparison_gb = df_comparison[[val, "classifier_name", "class_of_interest", "CV_index"]].groupby(["class_of_interest", "CV_index"]).apply(scores_side_by_side, c_i = classifier_j, c_j = classifier_i, val = val).reset_index()
                        sign, p = get_sign_significance_student_t(df_comparison_gb, orientation)
                        perc_times = float(np.count_nonzero(df_comparison_gb["classifier_i_score"] > df_comparison_gb["classifier_j_score"])) / len(df_comparison_gb["classifier_j_score"])

                    res[j, i] = sign
                    # add at position i j the perc_times formatted
                    res[i, j] = get_percentage_time_with_color(perc_times, opposite_sign(sign))

            for i in range(n_classifiers):
                res[i,i]=""

            classifiers_acronyms = [acronyms_dict[classifier] for classifier in classifiers_list]
            df_out = pd.DataFrame(res, columns=classifiers_acronyms, index=classifiers_acronyms)
            df_out.to_csv(get_matrix_path(dataset, comparison_test = comparison_test, class_of_interest = label))

            classifiers_acronyms = [acronyms_dict_latex[classifier] for classifier in classifiers_list]
            df_out = pd.DataFrame(res, columns=classifiers_acronyms, index=classifiers_acronyms)
            col_format = "|l|" + "c"*(n_classifiers- 1 ) +"r|"
            df_out.to_latex(get_matrix_path_latex(dataset, comparison_test = comparison_test, class_of_interest = label), caption = "Pairwise comparisons for the task {}".format(dataset.replace("_", "-").replace("-01-sub-merged", "")), position = "h!", column_format= col_format, label="table:pairwise-{}".format(dataset), escape = False)

def is_1_not_significantly_different_than_2(df, c_1, c_2): #by reading the table!
    classifiers_list = df.index
    i_c_1 = np.where(classifiers_list == c_1)
    i_c_2 = np.where(classifiers_list == c_2)
    if i_c_1 < i_c_2:
        sign = df.loc[c_2, c_1]
    else:
        sign = df.loc[c_1, c_2]
    return sign == "" or (sign is np.nan)

if __name__ == "__main__":
    metric = "balanced_accuracy"
    choosing_mode = "one_standard_error_rule"
    for dataset in dataset_names:
        classifiers_list = classifiers_figures
        make_comparison_matrix(dataset, classifiers_list, metric, choosing_mode)

    for dataset in dataset_names:
        for label in get_classes(dataset):
            make_comparison_matrix(dataset, classifiers_list, metric, choosing_mode, label = label)