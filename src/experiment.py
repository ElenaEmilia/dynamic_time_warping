"""Reproducibility: execute and replicate our experimental results."""

import logging

from evaluation import evaluate_multiple
from utils.visualization import generate_evaluation_plots, plot_r2_frechet_variances

logging.basicConfig(level=logging.INFO, format="%(asctime)s : %(levelname)s : %(message)s")


def precomputed_experiment_1():
    clusters_factor_list = [0.25, 0.5, 1, 1.5, 2]
    missing_rates_list = [0.1, 0.2, 0.3, 0.4]
    scores_path = "results/exp1_short_ts/scores_exp1.json"
    frechet_variances_path = "results/exp1_short_ts/scores_exp1_frechet_variances_dtw_km.json"
    seq_len_list = [1, 2, 4, 8]

    generate_evaluation_plots(scores_path, seq_len_list, missing_rates_list)
    plot_r2_frechet_variances(scores=scores_path,
                              frechet_variances=frechet_variances_path,
                              seq_len_list=seq_len_list,
                              missing_rates_list=missing_rates_list,
                              clusters_factor_list=clusters_factor_list)


def precomputed_experiment_2():
    clusters_factor_list = [0.25, 0.5, 1, 1.5, 2]
    missing_rates_list = [0.1, 0.2, 0.3, 0.4]
    scores_path = "results/exp2_long_ts/scores_exp2.json"
    frechet_variances_path = "results/exp2_long_ts/scores_exp2_frechet_variances_dtw_km.json"
    seq_len_list = [1, 5, 15, 30, 45]

    generate_evaluation_plots(scores_path, seq_len_list, missing_rates_list)
    plot_r2_frechet_variances(scores=scores_path,
                              frechet_variances=frechet_variances_path,
                              seq_len_list=seq_len_list,
                              missing_rates_list=missing_rates_list,
                              clusters_factor_list=clusters_factor_list)


def conduct_experiment_1():
    """
    Experiment 1 (short time series) from our paper. Evaluates the imputation methods on the given datasets,
    with the given parameters, and saves the results in a JSON file.
    """
    logging.info('Starting experiment 1...')

    seq_len_list = [1, 2, 4, 8]
    missing_rates_list = [0.1, 0.2, 0.3, 0.4]
    imputation_methods = ["interpolation", "arithmetic_mean", "dtw_km", "dtw_knn"]
    clusters_factor_list = [0.25, 0.5, 1, 1.5, 2]
    neighbors_list = [1, 2, 4, 8]
    runs_per_dataset = 10
    scores = evaluate_multiple(dataset_names=[
                                              "Chinatown",
                                              "DistalPhalanxOutlineAgeGroup",
                                              "DistalPhalanxOutlineCorrect",
                                              "DistalPhalanxTW",
                                              "ECG200",
                                              "ItalyPowerDemand",
                                              "MedicalImages",
                                              "MiddlePhalanxOutlineAgeGroup",
                                              "MiddlePhalanxOutlineCorrect",
                                              "MiddlePhalanxTW",
                                              "MoteStrain",
                                              "ProximalPhalanxOutlineAgeGroup",
                                              "ProximalPhalanxOutlineCorrect",
                                              "ProximalPhalanxTW",
                                              "SmoothSubspace",
                                              "SonyAIBORobotSurface1",
                                              "SonyAIBORobotSurface2",
                                              "SyntheticControl",
                                              "TwoLeadECG"
                                              ],
                               seq_len_list=seq_len_list,
                               missing_rates_list=missing_rates_list,
                               imputation_methods=imputation_methods,
                               clusters_factor_list=clusters_factor_list,
                               neighbors_list=neighbors_list,
                               runs_per_dataset=runs_per_dataset)

    print_scores(scores, missing_rates_list)
    generate_evaluation_plots(scores, seq_len_list, missing_rates_list)

    if "dtw_km" in imputation_methods:
        plot_r2_frechet_variances(scores=scores,
                                  frechet_variances="results/dtw_km_frechet_variances.json",
                                  seq_len_list=seq_len_list,
                                  missing_rates_list=missing_rates_list,
                                  clusters_factor_list=clusters_factor_list)

    logging.info('Experiment 1 finished.')


def conduct_experiment_2():
    """
    Experiment 2 (long time series) from our paper. Evaluates the imputation methods on the given datasets,
    with the given parameters, and saves the results in a JSON file.
    """
    logging.info('Starting experiment 2...')

    seq_len_list = [1, 5, 15, 30, 45]
    missing_rates_list = [0.1, 0.2, 0.3, 0.4]
    imputation_methods = ["interpolation", "arithmetic_mean", "dtw_km", "dtw_knn"]
    clusters_factor_list = [0.25, 0.5, 1, 1.5, 2]
    neighbors_list = [1, 2, 4, 8]
    runs_per_dataset = 10
    scores = evaluate_multiple(dataset_names=[
                                            "Beef",
                                            "BirdChicken",
                                            "Car",
                                            "Earthquakes",
                                            "Fish",
                                            "Lightning2",
                                            "InsectEPGSmallTrain"
                                            ],
                               seq_len_list=seq_len_list,
                               missing_rates_list=missing_rates_list,
                               imputation_methods=imputation_methods,
                               clusters_factor_list=clusters_factor_list,
                               neighbors_list=neighbors_list,
                               runs_per_dataset=runs_per_dataset)

    print_scores(scores, missing_rates_list)
    generate_evaluation_plots(scores, seq_len_list, missing_rates_list)

    if "dtw_km" in imputation_methods:
        plot_r2_frechet_variances(scores=scores,
                                  frechet_variances="results/dtw_km_frechet_variances.json",
                                  seq_len_list=seq_len_list,
                                  missing_rates_list=missing_rates_list,
                                  clusters_factor_list=clusters_factor_list)

    logging.info('Experiment 2 finished.')


def print_scores(scores, missing_rates_list):
    for dataset in scores:
        for method in scores[dataset]:
            logging.info(f"{dataset} - {method} imputation evaluation")
            for seq_len in scores[dataset][method]:
                logging.info(f"Sequence length: {seq_len}, missing rates: {missing_rates_list}. "
                             f"R2: {scores[dataset][method][seq_len]['r2']} "
                             f"MSE: {scores[dataset][method][seq_len]['mse']}")


if __name__ == '__main__':
    conduct_experiment_1()
    conduct_experiment_2()
