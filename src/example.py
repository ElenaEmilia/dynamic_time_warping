"""Runs DTW-Mean example"""
import logging
import numpy as np
import random

import tslearn.metrics
import tslearn.barycenters

import imputation
from dtw_mean import get_warp_val_mat
from utils.preprocessing import remove_values
from utils.load_ucr_data import load_ucr
from utils.visualization import plot_warping_path, align_ts_data, plot_ssg, plot_timeseries, generate_evaluation_plots, \
    plot_r2_frechet_variances
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s : %(levelname)s : %(message)s")


def dtw_example():
    logging.info('dtw_example')

    train, test = load_ucr("Chinatown")
    data = np.concatenate([train, test])

    x = data[0, :10]
    y = data[1, :7]

    p, d = tslearn.metrics.dtw_path(x, y)

    plot_warping_path(x, y, p, 500)
    align_ts_data(x, y, p, 500)
    __print_dtw_info(d, p)


def dtw_zero_cost_heuristic_example():
    logging.info('dtw_zero_cost_heuristic_example')

    x = np.array([5, 4, 1, 8, 4, 2, 5])
    y = np.array([1, 3, np.nan, 5, np.nan, 3, 1, 3])
    p, d = tslearn.metrics.zc_dtw_path(x, y)

    plot_warping_path(x, y, p, 10)
    __print_dtw_info(d, p)


def ssg_example():
    logging.info('ssg_example')
    train, test = load_ucr("Chinatown")
    X = np.concatenate([train, test])

    X_mean = np.squeeze(tslearn.barycenters.dtw_barycenter_averaging_subgradient(X))
    plot_ssg(X, X_mean)


def linear_interpolation_example():
    train, test = load_ucr("Chinatown")
    data = np.concatenate([train, test])
    missing_timeseries = remove_values(data, missing_rate=0.4, seq_len=5)
    np_fixed_timeseries = imputation.linear_interpolation(missing_timeseries)
    idx_missing = np.argwhere(np.isnan(np.ndarray.astype(missing_timeseries[0], float))).flatten()

    plt.plot(data[0], 'g--', label="Time Series with missing values")
    plt.scatter(idx_missing, data[0][idx_missing], color='red')
    plt.plot(np_fixed_timeseries[0], 'm', label="Imputed time series with interpolation")

    fontsize = 14
    pad = 10

    plt.title("Linear interpolation", fontsize=17, pad=pad)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.ylabel(f"Value", fontsize=fontsize, labelpad=pad)

    # Add legend below plot
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    plt.savefig(f"results/linear_interpolation.png", bbox_inches="tight")

    plt.show()


def arithmetic_mean_imputation_example():
    train, test = load_ucr("Chinatown", class_label=1)
    data = train.copy()
    missing_timeseries = remove_values(data, missing_rate=0.4, seq_len=5)
    imputed_timeseries = imputation.arithmetic_mean_imputation(missing_timeseries)

    idx_missing = np.argwhere(np.isnan(np.ndarray.astype(missing_timeseries[0], float))).flatten()

    plt.plot(train[0], 'g--', label="Time Series with missing values")
    plt.scatter(idx_missing, data[0][idx_missing], color='red')
    for j, data in enumerate(data):
        if j > 0:
            plt.plot(data, 'y', alpha=0.2)

    plt.plot(imputed_timeseries[0], 'm', label="Imputed Time Series with arithmetic mean")

    fontsize = 14
    pad = 10

    plt.title("Linear interpolation", fontsize=17, pad=pad)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.ylabel(f"Value", fontsize=fontsize, labelpad=pad)

    plt.title("Arithmetic mean", fontsize=17, pad=pad)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    plt.savefig(f"results/arithmetic_mean.png", bbox_inches="tight")

    plt.show()


def dtw_knn_example():
    logging.info('dtw_knn_example')

    train, test = load_ucr("Chinatown")
    x = np.concatenate([train, test])

    random_idx = random.choice(range(len(x)))

    x_missing = remove_values(x, missing_rate=0.4, seq_len=8)
    plot_timeseries(x_missing[random_idx], title="DTW KNN - incomplete")

    x_imputed = imputation.dtw_knn_imputation(x_missing, neighbors=3)
    plot_timeseries(x_imputed[random_idx], title="DTW KNN - imputed")


def dtw_k_means_example():
    logging.info('dtw_km_example')

    train, test = load_ucr("Chinatown")
    x = np.concatenate([train, test])

    random_idx = random.choice(range(len(x)))

    x_missing = remove_values(x, missing_rate=0.4, seq_len=8)
    plot_timeseries(x_missing[random_idx], title="DTW K-Means - incomplete")

    x_imputed = imputation.dtw_kmeans_imputation(x_missing,
                                                 class_labels=False,
                                                 k=3,
                                                 dist="dtw",
                                                 mean="frechet",
                                                 init="lin",
                                                 plot=True)
    plot_timeseries(x_imputed[random_idx], title="DTW K-Means - imputed")


def evaluation_precomputed_example():
    logging.info('evaluation_precomputed_example')

    scores_path = "results/exp1_short_ts/scores_exp1.json"
    seq_len_list = [1, 2, 4, 8]
    missing_rates_list = [0.1, 0.2, 0.3, 0.4]

    generate_evaluation_plots(scores_path, seq_len_list, missing_rates_list)
    plot_r2_frechet_variances(scores=scores_path,
                              frechet_variances="results/exp1_short_ts/scores_exp1_frechet_variances_dtw_km.json",
                              seq_len_list=seq_len_list,
                              missing_rates_list=missing_rates_list,
                              clusters_factor_list=[0.25, 0.5, 1, 1.5, 2])


def __print_dtw_info(d, p):
    logging.info('__print_dtw_info')

    logging.info('dist = %.2f' % d)
    W, V = get_warp_val_mat(p)
    logging.info(f"Warping matrix:\n{W}")
    logging.info(f"Warping path:\n{p}")
    logging.info(f"Diagonal of valence matrix:\n{V}")


if __name__ == "__main__":
    dtw_example()
    dtw_zero_cost_heuristic_example()
    ssg_example()
    linear_interpolation_example()
    arithmetic_mean_imputation_example()
    dtw_knn_example()
    dtw_k_means_example()
    evaluation_precomputed_example()
