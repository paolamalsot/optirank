import pandas as pd
from simulation_model.create_params_grid import default_params, path_experiment_table
from utilities.small_functions import mkDir_if_not
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from simulation_model.gather_results import bilinear_name


def get_results_experiment_name_path(experiment_name):
    results_dir = mkDir_if_not("simulation_model/results")
    return os.path.join(results_dir, "{}_results_cv.pkl".format(experiment_name))


def min_of_gt_0(x):
    return np.min(x[x > 0])


acronyms_dict = {"optirank": "optirank",
                 "logistic_regression_on_ranks": "rank-lr",
                 "logistic_regression": "lr"}


def acronym(serie):
    """df is a dataframe with column classifier_name. Add column classifier_acronym"""
    fun = lambda x: acronyms_dict[x]
    return serie.apply(fun)


dict_param_name_to_latex = {"d": "$d$",  # correspondences to the paper
                            "n_perturbing": "$d_{P}$",
                            "n_samples": "$n_{samples}$",
                            "tau": "$\\tau$",
                            "sigma": "$\sigma$"}

if __name__ == "__main__":
    experiment_table = pd.read_pickle(path_experiment_table)
    sns.set(font_scale=1.5)  # was 1.5
    sns.set_style("whitegrid")

    for i_row, row in experiment_table.iterrows():

        experiment_name = row["experiment_name"]
        n_params = row["n_params"]
        params_grid = list(row["param_grid"])
        results = pd.read_pickle(get_results_experiment_name_path(experiment_name))
        results["classifier"] = acronym(results["classifier_name"])

        for with_legend in [True, False]:
            output_dir = "simulation_model/results"
            if with_legend:
                legend = "full"
                location = "upper left"
                output_dir_plots = mkDir_if_not(os.path.join(output_dir, "plots", "legend_on"))

            else:
                legend = False
                location = "best"  # should have no effect
                output_dir_plots = mkDir_if_not(os.path.join(output_dir, "plots", "legend_off"))

            plt.figure()
            p = sns.lineplot(data=results.reset_index(), x="param_value", y="test_balanced_accuracy", hue="classifier",
                             legend=legend)
            lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncol=3)
            p.axes.set_xlabel(dict_param_name_to_latex[pd.unique(results["param_name"]).item()], )
            p.axes.set_ylabel("test balanced accuracy (%)")
            # if pd.unique(results["param_name"]).item() == "sigma_m":
            #     p.axes.set_xscale("symlog", linthresh=min_of_gt_0(results[["param_value"]].values))
            figpath = os.path.join(output_dir_plots, "results_{}.pdf".format(experiment_name))
            plt.tight_layout()
            if with_legend:
                lgd.set_visible(True)
            else:
                lgd.set_visible(False)
            plt.savefig(figpath)
            plt.close()

            # overlap figure
            plt.figure()
            results_of_interest = results.loc[results.classifier_name == bilinear_name]
            p = sns.lineplot(data=results_of_interest.reset_index(), x="param_value", y="overlap", legend=legend)

            lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncol=3)
            p.axes.set_xlabel(dict_param_name_to_latex[pd.unique(results["param_name"]).item()], )
            p.axes.set_ylabel("overlap")
            p.axes.set_ylim([0, 1])
            # if pd.unique(results["param_name"]).item() == "sigma_m":
            #     p.axes.set_xscale("symlog", linthresh=min_of_gt_0(results[["param_value"]].values))
            figpath = os.path.join(output_dir_plots, "results_overlap_{}.pdf".format(
                experiment_name))
            plt.tight_layout()
            if with_legend:
                lgd.set_visible(True)
            else:
                lgd.set_visible(False)
            plt.savefig(figpath)
            plt.close()

    # results for default parameters (taken from the experiment on d for instance)
    experiment_name = "different_d"
    results = pd.read_pickle(get_results_experiment_name_path(experiment_name))
    results_per_classifier = results.loc[
        results.param_value == default_params["d"], ["test_balanced_accuracy", "classifier_name", "overlap"]].groupby(
        ["classifier_name"]).agg(['mean', 'sem'])
    results_per_classifier.to_csv("simulation_model/results/results_default_parameters.csv")

    #output file for markdown
    outfile_md = "simulation_model/results/results_default_parameters.md"
    out_md = (100*results_per_classifier[("test_balanced_accuracy", "mean")]).map('{:,.0f}'.format) + " ± " + (100*results_per_classifier[("test_balanced_accuracy", "sem")]).map('{:,.0f}'.format)
    out_md = out_md.to_frame(name="test balanced accuracy")
    markdown_table = out_md.to_markdown()
    f = open(outfile_md, "w")
    f.write(markdown_table)
    f.close()
