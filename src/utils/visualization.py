"""Provides DTW visualization fuctionality"""

# pylint: disable=E0401
import json
import uuid

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams

import imputation
from utils.load_ucr_data import load_ucr
from utils.preprocessing import summarize_km_and_knn_scores

rcParams.update({"figure.autolayout": True})


def generate_evaluation_plots(scores, seq_len_list, missing_rates_list):
    # All imputation methods over different missing rates and missing sequences
    visualize_evaluation_hyperparameters(scores=scores,
                                         seq_len_list=seq_len_list,
                                         missing_rates_list=missing_rates_list,
                                         central_measure="mean")
    visualize_evaluation_hyperparameters(scores=scores,
                                         seq_len_list=seq_len_list,
                                         missing_rates_list=missing_rates_list,
                                         central_measure="median")

    # DTW KM cluster factors over different missing rates and missing sequences
    visualize_evaluation_hyperparameters(scores=scores,
                                         seq_len_list=seq_len_list,
                                         missing_rates_list=missing_rates_list,
                                         summarize_km_and_knn=False,
                                         only_show="dtw_km",
                                         central_measure="mean")
    visualize_evaluation_hyperparameters(scores=scores,
                                         seq_len_list=seq_len_list,
                                         missing_rates_list=missing_rates_list,
                                         summarize_km_and_knn=False,
                                         only_show="dtw_km",
                                         central_measure="median")

    # DTW KNN neighbors size over different missing rates and missing sequences
    visualize_evaluation_hyperparameters(scores=scores,
                                         seq_len_list=seq_len_list,
                                         missing_rates_list=missing_rates_list,
                                         summarize_km_and_knn=False,
                                         only_show="dtw_knn",
                                         central_measure="mean")
    visualize_evaluation_hyperparameters(scores=scores,
                                         seq_len_list=seq_len_list,
                                         missing_rates_list=missing_rates_list,
                                         summarize_km_and_knn=False,
                                         only_show="dtw_knn",
                                         central_measure="median")

    # All imputation methods averaged over different missing rates and missing sequences
    visualize_evaluation_overall(scores=scores,
                                 seq_len_list=seq_len_list,
                                 missing_rates_list=missing_rates_list,
                                 central_measure="mean")
    visualize_evaluation_overall(scores=scores,
                                 seq_len_list=seq_len_list,
                                 missing_rates_list=missing_rates_list,
                                 central_measure="median")


def plot_timeseries(X, title="Time series"):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_visible(False)

    # Title
    ax.set_title(title, fontsize=30, fontweight="bold", pad=30)

    # Labels
    ax.set_xlabel("Time", fontsize=25)
    ax.set_ylabel("Value", fontsize=25)

    # Ticks
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Padding labels
    ax.xaxis.labelpad = 30
    ax.yaxis.labelpad = 30

    ax.plot(X, "-bo", label="y", linewidth=3, markersize=0)

    # Save image
    appendix = str(uuid.uuid4())[:4]
    title = title.replace(" - ", " ").replace(" ", "_")
    plt.savefig(f"results/{title}_{appendix}.png", bbox_inches="tight")
    
    plt.show()


def plot_ssg(X, X_mean, f=None):
    # Plot SSG mean
    plt.plot(X[0], color='gray', label='sample', linewidth=0.5)
    for i in range(1, X.shape[0]):
        plt.plot(X[i], color='gray', linewidth=0.5)
    plt.plot(X_mean, '-k', linewidth=3, label='SSG')
    plt.legend()

    # Save image
    appendix = str(uuid.uuid4())[:4]
    plt.savefig(f"results/ssg_{appendix}.png", bbox_inches="tight")
    
    plt.show()

    # Plot Fréchet variations
    if f is not None:
        plt.plot(f)
        plt.xlabel('Epoch')
        plt.ylabel('Frechet variation')

        # Save image
        appendix = str(uuid.uuid4())[:4]
        plt.savefig(f"results/ssg_frechet_variations_{appendix}.png", bbox_inches="tight")
        
        plt.show()


def plot_warping_path(x, y, warping_path, offset=0):
    if type(warping_path) is not np.ndarray:
        warping_path = np.array(warping_path)

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_visible(False)

    # Title
    ax.set_title("DTW - Warping Path", fontsize=30, fontweight="bold", pad=30)

    # Labels
    ax.set_xlabel("Time", fontsize=25)
    ax.set_ylabel("Value", fontsize=25)

    # Ticks
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Padding labels and title
    ax.xaxis.labelpad = 30
    ax.yaxis.labelpad = 30

    y += offset

    ax.plot(x, "-ro", label="x", linewidth=3, markersize=15)
    ax.plot(y, "-bo", label="y", linewidth=3, markersize=15)

    warping_path = warping_path.astype(int)

    if np.any(np.isnan(x)):
        missing_x = np.argwhere(np.isnan(x))
        x_interpolated = imputation.linear_interpolation(x)
        for idx, missing in enumerate(missing_x):
            ax.plot(missing, x_interpolated[missing], "Dy", label="NaN" if idx == 0 else "", linewidth=3, markersize=25)

        for x_value, y_value in warping_path:
            ax.plot([x_value, y_value], [x_interpolated[x_value], y[y_value]], "--k", linewidth=2)

    if np.any(np.isnan(y)):
        missing_y = np.argwhere(np.isnan(y)).flatten()
        y_interpolated = imputation.linear_interpolation(y)
        for idx, missing in enumerate(missing_y):
            ax.plot(missing, y_interpolated[missing], "Dy", label="NaN" if idx == 0 else "", linewidth=3, markersize=25)

        for x_value, y_value in warping_path:
            ax.plot([x_value, y_value], [x[x_value], y_interpolated[y_value]], "--k", linewidth=2)

    ax.legend(loc="upper left", fontsize="xx-large", markerscale=0.7, bbox_to_anchor=(1.04, 1))

    # Save image
    appendix = str(uuid.uuid4())[:4]
    plt.savefig(f"results/warping_path_{appendix}.png", bbox_inches="tight")
    
    plt.show()


def plot_k_means(x, centroids, cluster):
    yValues = x
    xValues = np.arange(0, x.shape[1], 1)
    colorValues = cluster.astype(float)

    for y, c in zip(yValues, colorValues):
        plt.plot(xValues, y, color=plt.cm.cividis(c), linewidth=0.9)
    for c in centroids:
        plt.plot(xValues, c, c='red', label = 'Centroid')

    plt.title("K-means clusters", fontsize=17, pad=20)

    # Save image
    appendix = str(uuid.uuid4())[:4]
    plt.savefig(f"results/k_means_{appendix}.png", bbox_inches="tight")
    
    plt.show()


def visualize_evaluation_per_dataset(scores=None,
                                     seq_len_list=None,
                                     missing_rates_list=None):
    if type(scores) is not dict:
        # Load scores from file
        with open(scores, "r") as file:
            scores = json.loads(file.read())

    assert type(scores) is dict

    num_of_datasets = len(scores.keys())
    if num_of_datasets == 1:
        visualize_evaluation_hyperparameters(scores=scores,
                                             seq_len_list=seq_len_list,
                                             missing_rates_list=missing_rates_list)
        return

    fig, ax = plt.subplots(num_of_datasets, len(seq_len_list), figsize=(18, 8), sharex=True, sharey=True)

    for i, dataset_name in enumerate(scores):
        for method in scores[dataset_name]:
            for j, seq_len in enumerate(scores[dataset_name][method]):
                ax[i][j].plot(missing_rates_list, scores[dataset_name][method][str(seq_len)]["r2"], label=method)
                ax[i][j].spines.right.set_visible(False)
                ax[i][j].spines.top.set_visible(False)
                if i == 0:
                    ax[i][j].set_title("Interval: " + str(seq_len))
                else:
                    ax[i][j].set_xlabel('Missing rate')

                if j == 0:
                    ax[i][j].set_ylabel(f"$\\bf{dataset_name}$\n R$^2$ score")
                elif j == len(seq_len_list) - 1:
                    ax[i][j].yaxis.set_label_position("right")
                    ax[i][j].yaxis.tick_right()
                    ax[i][j].axes.get_yaxis().set_visible(False)
                    ax[i][j].set_ylabel(str(dataset_name), labelpad=10)

    plt.xticks(missing_rates_list)
    plt.yticks([0, 0.25, 0.5, 0.75, 1])
    plt.ylim([0, 1])

    imputation_methods = scores[list(scores.keys())[0]].keys()
    ax[num_of_datasets // 2][len(seq_len_list) - 1].legend(labels=imputation_methods,
                                                           title="Imputer models",
                                                           loc=(1.04, 0.5))
    
    # Save image
    appendix = str(uuid.uuid4())[:4]
    plt.savefig(f"results/scores_evaluation_{appendix}.png", bbox_inches="tight")

    plt.show()


def visualize_evaluation_hyperparameters(scores=None,
                                         seq_len_list=None,
                                         missing_rates_list=None,
                                         central_measure="mean",
                                         summarize_km_and_knn=True,
                                         only_show=None):
    if type(scores) is not dict:
        # Load scores from file
        with open(scores, "r") as file:
            scores = json.loads(file.read())

    assert type(scores) is dict

    # Different neighbors size or cluster factor configurations into one score
    if summarize_km_and_knn:
        scores = summarize_km_and_knn_scores(scores, seq_len_list, missing_rates_list)

    scores_per_method = {}
    num_of_datasets = len(scores.keys())

    imputation_methods = scores[list(scores.keys())[0]].keys()
    if only_show:
        imputation_methods = [method for method in imputation_methods if method.startswith(only_show)]

    for imputation_method in imputation_methods:
        for seq_len in seq_len_list:
            scores_np = np.full(fill_value=np.nan, shape=(num_of_datasets, len(missing_rates_list)))

            for idx, dataset in enumerate(scores):
                scores_np[idx] = scores[dataset][imputation_method][str(seq_len)]["r2"]

            if central_measure == "median":
                method_center_per_seq_len = np.nanmedian(scores_np, axis=0)
            else:
                method_center_per_seq_len = np.nanmean(scores_np, axis=0)

            if imputation_method not in scores_per_method:
                scores_per_method[imputation_method] = {}
            scores_per_method[imputation_method][seq_len] = np.around(method_center_per_seq_len, 2)

    fig, ax = plt.subplots(ncols=len(seq_len_list), figsize=(18, 4.5), sharex=True, sharey=True)

    for imputation_method in scores_per_method:
        for j, seq_len in enumerate(seq_len_list):
            ax[j].set_xlabel("Missing rate", fontsize=14, labelpad=10)

            ax[j].plot(missing_rates_list, scores_per_method[imputation_method][seq_len], label=imputation_method)
            ax[j].spines.right.set_visible(False)
            ax[j].spines.top.set_visible(False)
            ax[j].set_title(f"Interval: {seq_len}", fontsize=16, pad=20)

            # Increase tick fontsize
            for tick in ax[j].xaxis.get_major_ticks():
                tick.label.set_fontsize(14)
            for tick in ax[j].yaxis.get_major_ticks():
                tick.label.set_fontsize(14)

            if j == 0:
                ax[j].set_ylabel(f"Average of different datasets\n R$^2$ score", fontsize=14, labelpad=10)

    plt.xticks(missing_rates_list)
    plt.yticks([0, 0.25, 0.5, 0.75, 1])
    plt.ylim([0, 1])

    ax[len(seq_len_list) - 1].legend(labels=imputation_methods, title="Imputer models", loc=(1.04, 0.5), fontsize=14, title_fontsize=14)

    if central_measure == "mean":
        central_measure = "average"
    plt.suptitle(f"{central_measure.capitalize()} of all runs", fontsize=20)

    # Save image
    appendix = str(uuid.uuid4())[:4]
    plt.savefig(f"results/scores_evaluation_average_{central_measure.replace(' ', '_')}_{appendix}.png", bbox_inches="tight")
    
    plt.show()


def visualize_evaluation_overall(scores=None,
                                 seq_len_list=None,
                                 missing_rates_list=None,
                                 central_measure="mean"):
    if type(scores) is not dict:
        # Load scores from file
        with open(scores, "r") as file:
            scores = json.loads(file.read())

    assert type(scores) is dict

    num_of_datasets = len(scores.keys())

    # Get list of all imputation methods
    imputation_methods = []
    for dataset in scores:
        for method in scores[dataset]:
            if method not in imputation_methods:
                imputation_methods.append(method)

    scores_per_method = {}
    scores_dtw_km = {}
    scores_dtw_knn = {}
    for imputation_method in imputation_methods:

        scores_np = np.full(fill_value=np.nan, shape=(num_of_datasets * len(seq_len_list), len(missing_rates_list)))

        for seq_idx, seq_len in enumerate(seq_len_list):
            for idx, dataset in enumerate(scores):
                scores_np[idx * len(seq_len_list) + seq_idx] = scores[dataset][imputation_method][str(seq_len)]["r2"]

        if central_measure == "median":
            method_center = np.around(np.nanmedian(scores_np), 2)
        else:
            method_center = np.around(np.nanmean(scores_np), 2)

        if "dtw_km" in imputation_method and imputation_method not in scores_dtw_km:
            scores_dtw_km[imputation_method] = {}
        elif "dtw_knn" in imputation_method and imputation_method not in scores_dtw_knn:
            scores_dtw_knn[imputation_method] = {}
        elif imputation_method not in scores_per_method:
            scores_per_method[imputation_method] = {}

        if "dtw_km" in imputation_method:
            scores_dtw_km[imputation_method] = method_center
        elif "dtw_knn" in imputation_method:
            scores_dtw_knn[imputation_method] = method_center
        else:
            scores_per_method[imputation_method] = method_center

    # Aggregate different DTW K-Means / KNN hyperparameter configurations
    if scores_dtw_km:
        dtw_km_score = np.nanmean(list(scores_dtw_km.values()))
        scores_per_method["dtw_km"] = np.around(dtw_km_score, 2)

    if scores_dtw_knn:
        dtw_knn_score = np.nanmean(list(scores_dtw_knn.values()))
        scores_per_method["dtw_knn"] = np.around(dtw_knn_score, 2)

    plt.figure(figsize=(8, 4))
    plt.ticklabel_format(useOffset=False)

    for idx, imputation_method in enumerate(scores_per_method):
        plt.xlabel("Imputation methods", fontsize=14, labelpad=10)
        plt.ylabel(f"Average imputation performance\n R$^2$ score", fontsize=14, labelpad=10)

        plt.bar(idx, scores_per_method[imputation_method], width=0.4)

        if central_measure == "mean":
            central_measure = "average"
        plt.title(f"{central_measure.capitalize()} of all runs", fontsize=16, pad=20)

    imputation_methods = list(scores_per_method.keys())
    plt.xticks(range(len(imputation_methods)), imputation_methods, fontsize=14)
    plt.yticks([0, 0.25, 0.5, 0.75, 1], fontsize=14)
    plt.ylim([0, 1])

    # Save image
    appendix = str(uuid.uuid4())[:4]
    plt.savefig(f"results/scores_evaluation_method_average_{central_measure.replace(' ', '_')}_{appendix}.png", bbox_inches="tight")

    plt.show()


def plot_r2_frechet_variances(scores=None,
                              frechet_variances=None,
                              seq_len_list=[1, 2, 4, 8],
                              missing_rates_list=[0.1, 0.2, 0.3, 0.4],
                              clusters_factor_list=[0.25, 0.5, 1, 1.5, 2],
                              normalize=True):
    if type(scores) is not dict:
        # Load scores from file
        with open(scores, "r") as file:
            scores = json.loads(file.read())

    if type(frechet_variances) is not dict:
        # Load Fréchet variances from file
        with open(frechet_variances, "r") as file:
            frechet_variances = json.loads(file.read())

    assert type(scores) is dict and type(frechet_variances) is dict

    imputation_methods = list(scores[list(scores.keys())[0]].keys())
    dtw_kmeans_list = [m for m in imputation_methods if m.startswith("dtw_km")]

    scores_per_dataset = {}
    for dataset in scores:
        scores_per_dataset[dataset] = {}

        for method_idx, method in enumerate(dtw_kmeans_list):
            scores_np = np.full(fill_value=np.nan,
                                shape=(len(seq_len_list) * len(clusters_factor_list), len(missing_rates_list)))

            for seq_idx, seq_len in enumerate(seq_len_list):
                scores_np[seq_idx + len(seq_len_list) * method_idx] = scores[dataset][method][str(seq_len)]["r2"]

            scores_per_dataset[dataset][method] = np.around(np.nanmean(scores_np), 2)

    frechet_variances_per_dataset = {}
    frechet_max_per_dataset = {}
    for dataset in frechet_variances:
        current_max = np.nan

        frechet_variances_per_dataset[dataset] = {}
        for method in frechet_variances[dataset]:
            f_vars = frechet_variances[dataset][method]
            for v in f_vars:
                if np.isnan(current_max) or np.abs(v) > current_max:
                    current_max = np.abs(v)

            frechet_variances_per_dataset[dataset][method] = np.around(np.mean(f_vars), 2)

        frechet_max_per_dataset[dataset] = current_max

    # Normalize Fréchet variances
    normalized_frechet_variances = {}
    for dataset in scores_per_dataset:
        x, _ = load_ucr(dataset)

        normalized_frechet_variances[dataset] = {}
        for method in scores_per_dataset[dataset]:
            frechet_max = frechet_max_per_dataset[dataset]

            frechet_variance = frechet_variances_per_dataset[dataset][method.split("_")[-1]]
            if normalize:
                variation = frechet_variance / frechet_max
            else:
                variation = frechet_variance
            normalized_frechet_variances[dataset][method] = np.around(variation, 2)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(17, 8.27))
    fontsize = 20
    norm_frechet = []
    r2 = []

    colors = ["red", "green", "mediumblue", "orange", "violet"]
    for idx, factor in enumerate(dtw_kmeans_list):
        for i, dataset in enumerate(scores_per_dataset):
            x = normalized_frechet_variances[dataset][factor]
            y = scores_per_dataset[dataset][factor]

            ax.scatter(x, y, label=factor, c=colors[idx], marker="o", s=100)

            norm_frechet.append(x)
            r2.append(y)

    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(),
               by_label.keys(),
               title="Factor of the number of classes",
               loc=(1.04, 0.57),
               fontsize=fontsize,
               title_fontsize=fontsize)

    # Labels and ticks
    ax.set_ylabel(f"R$^2$ score", fontsize=fontsize, labelpad=10)
    if normalize:
        ax.set_xlabel(f"Normalized Fréchet variance", fontsize=fontsize, labelpad=10)
    else:
        ax.set_xlabel(f"Fréchet variance", fontsize=fontsize, labelpad=10)
    ax.grid(True)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # Save image
    appendix = str(uuid.uuid4())[:4]
    plt.savefig(f"results/evaluation_frechet_variances_{appendix}.png", bbox_inches="tight")
    
    plt.show()


def align_ts_data(x, y, warping_path, offset=None):
    """It plots the alignment between two time series

    Parameters
    ----------
    x
        the first time series
    y
        the time series to be aligned
    warping_path
        The warping path of the DTW algorithm.
    offset
        The offset to add to the y-axis.

    """
    if type(warping_path) is not np.ndarray:
        warping_path = np.array(warping_path)

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_visible(False)

    # Title
    ax.set_title("DTW - Alignment", fontsize=30, fontweight="bold", pad=30)

    # Labels
    ax.set_xlabel("Time", fontsize=25)
    ax.set_ylabel("Value", fontsize=25)

    # Ticks
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Padding labels
    ax.xaxis.labelpad = 30
    ax.yaxis.labelpad = 30

    x = x[warping_path.astype(int).T[0]]
    y = y[warping_path.astype(int).T[1]] + offset

    ax.plot(x, "-ro", label="x", linewidth=3, markersize=15)
    ax.plot(y, "-bo", label="y", linewidth=3, markersize=15)

    if np.any(np.isnan(x)):
        missing_x = np.argwhere(np.isnan(x))
        imputation.linear_interpolation(x.reshape(1, -1))
        ax.plot(missing_x, x[missing_x], "Dy", label="NAN", linewidth=3, markersize=25)

    if np.any(np.isnan(y)):
        missing_y = np.argwhere(np.isnan(y))
        imputation.linear_interpolation(y.reshape(1, -1))
        ax.plot(missing_y, y[missing_y], "Dy", label="NAN", linewidth=3, markersize=25)

    # Plot
    for i in range(len(x)):
        ax.plot([i, i], [x[i], y[i]], "--k", linewidth=2)

    ax.legend(loc="upper left", fontsize="xx-large", markerscale=0.7, bbox_to_anchor=(1.04, 1))

    # Save image
    appendix = str(uuid.uuid4())[:4]
    plt.savefig(f"results/alignment_{appendix}.png", bbox_inches="tight")

    plt.show()
