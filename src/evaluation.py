import json
import logging
import uuid

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

import imputation as imputation
from utils.load_ucr_data import load_ucr
from utils.preprocessing import remove_values

logging.basicConfig(level=logging.INFO, format="%(asctime)s : %(levelname)s : %(message)s")


def evaluate_multiple(dataset_names=None,
                      seq_len_list=[1, 2, 4, 8],
                      missing_rates_list=[0.1, 0.2, 0.3, 0.4],
                      imputation_methods=["dtw_km", "dtw_knn", "interpolation", "arithmetic_mean"],
                      clusters_factor_list=[0.25, 0.5, 1, 1.5, 2],
                      neighbors_list=[1, 2, 4, 8],
                      runs_per_dataset=10):
    """
    Experiment 1 from our paper (short time series). Evaluates the imputation methods on the given datasets,
    with the given parameters, and saves the results in a JSON file.

    ----------
    Parameters
    ----------
    dataset_names
        A list of the datasets to evaluate.
    seq_len_list
        Lengths of the subsequences to delete at once.
    missing_rates_list
        Missing rates to test.
    imputation_methods
        Methods to be evaluated
    clusters_factor_list
        Number of clusters is calculated as the number of classes in the dataset times the factor.
    neighbors_list
        Number of neighbors to use for the DTW KNN imputation method.
    runs_per_dataset, optional
        Number of runs per dataset

    Returns
    -------
    dict:
        A dictionary of the results.

    """

    assert dataset_names is not None, "Dataset names must be provided."

    all_scores = {}
    frechet_variances_per_factor = {}
    for dataset_name in dataset_names:
        train, _ = load_ucr(dataset_name)
        logging.info(f"Loaded dataset {dataset_name}")

        x_true = train

        all_scores[dataset_name] = {}
        frechet_variances_per_factor[dataset_name] = {}

        for method in imputation_methods:

            if method == "dtw_km":
                for factor in clusters_factor_list:

                    # Factor number of classes into clusters
                    df = pd.read_csv("data/UCR_DataSummary.csv")
                    n_classes = df.loc[df["Name"] == dataset_name, "Class"].values[0]
                    clusters = round(n_classes * factor) if round(n_classes * factor) > 0 else 1

                    scores, frechet_variances = evaluate(x_true,
                                                         seq_len_list=seq_len_list,
                                                         missing_rates_list=missing_rates_list,
                                                         imputation_method=method,
                                                         clusters=clusters,
                                                         runs=runs_per_dataset)

                    all_scores[dataset_name][f"{method}_{factor}"] = scores
                    frechet_variances_per_factor[dataset_name][factor] = list(frechet_variances)

            elif method == "dtw_knn":
                for neighbors in neighbors_list:
                    scores = evaluate(x_true,
                                      seq_len_list=seq_len_list,
                                      missing_rates_list=missing_rates_list,
                                      imputation_method=method,
                                      neighbors=neighbors,
                                      runs=runs_per_dataset)
                    all_scores[dataset_name][f"{method}_{neighbors}"] = scores
            else:
                all_scores[dataset_name][method] = evaluate(x_true=x_true,
                                                            seq_len_list=seq_len_list,
                                                            missing_rates_list=missing_rates_list,
                                                            imputation_method=method,
                                                            runs=runs_per_dataset)

    # Scores
    appendix = str(uuid.uuid4())[:4]  # Prevent overwriting of previous scores
    with open(f"results/scores_{appendix}.json", "w") as file:
        file.write(json.dumps(all_scores, indent=4))

    if "dtw_km" in imputation_methods:
        with open(f"results/dtw_km_frechet_variances.json", "w") as file:
            file.write(json.dumps(frechet_variances_per_factor, indent=4))

    return all_scores


def evaluate(
    x_true,
    seq_len_list=[1, 2, 4, 8],
    missing_rates_list=[0.1, 0.2, 0.3, 0.4],
    imputation_method="dtw_km",
    neighbors=None,
    clusters=None,
    runs=10
):
    """
    Evaluation for single dataset, and single imputation method. We recommend using the wrapper `evaluate_multiple`
    to evaluate multiple datasets and multiple imputation methods at once.
    """
    logging.info(f"Imputing... Method: {imputation_method}, "
                 f"sequence length to remove: {seq_len_list}, "
                 f"missing rates: {missing_rates_list}")

    scores = []
    if imputation_method == "dtw_km":
        frechet_variances_scores = np.full(fill_value=np.nan,
                                           shape=(runs * len(seq_len_list) * len(missing_rates_list), clusters))

    iteration = 0
    for run in range(runs):

        run_score = {}

        for seq_len in seq_len_list:
            for p in missing_rates_list:
                logging.info(f"{imputation_method} - Run {run+1} - Sequence length - {seq_len}, missing rate - {p}")

                x_incomplete = remove_values(x_true, seq_len=seq_len, missing_rate=p)

                if not np.any(np.isnan(x_incomplete)):
                    logging.info("Skipped due to incompatible missing rate and missing sequence length combination.")

                    if str(seq_len) not in run_score:
                        run_score[str(seq_len)] = {"r2": [np.nan], "mse": [np.nan]}
                    else:
                        run_score[str(seq_len)]["r2"].append(np.nan)
                        run_score[str(seq_len)]["mse"].append(np.nan)

                    continue

                if imputation_method == "dtw_km":
                    logging.info(f"Clusters: {clusters}")
                    assert clusters is not None, "Cluster factors must be provided for DTW K-Means."
                    x_pred, frechet_variances = imputation.dtw_kmeans_imputation(x_incomplete,
                                                                                 class_labels=False,
                                                                                 k=clusters,
                                                                                 dist="dtw",
                                                                                 mean="frechet",
                                                                                 init="arithmetic",
                                                                                 frechet_variance=True)
                    frechet_variances_scores[iteration] = frechet_variances

                elif imputation_method == "dtw_knn":
                    logging.info(f"Neighbors: {neighbors}")
                    assert neighbors is not None, "Neighbors must be provided for DTW KNN."
                    x_pred = imputation.dtw_knn_imputation(x_incomplete,
                                                           neighbors=neighbors)

                elif imputation_method == "interpolation":
                    x_pred = imputation.linear_interpolation(x_incomplete)

                elif imputation_method == "arithmetic_mean":
                    x_pred = imputation.arithmetic_mean_imputation(x_incomplete)

                else:
                    logging.info(f"Unknown imputation method {imputation_method}.")
                    return

                r2 = calculate_metric(x_true, x_pred, metric="r2")
                mse = calculate_metric(x_true, x_pred, metric="mse")

                if str(seq_len) not in run_score:
                    run_score[str(seq_len)] = {"r2": [r2], "mse": [mse]}
                else:
                    run_score[str(seq_len)]["r2"].append(r2)
                    run_score[str(seq_len)]["mse"].append(mse)

                iteration += 1

        scores.append(run_score)

    # Calculate mean between runs
    average_scores = {}
    for seq_len in run_score:
        r2_scores = [score[seq_len]["r2"] for score in scores]
        mse_scores = [score[seq_len]["mse"] for score in scores]

        r2_scores_mean = np.mean(r2_scores, axis=0)
        mse_scores_mean = np.mean(mse_scores, axis=0)

        r2_scores_rounded = np.around(r2_scores_mean, decimals=2).tolist()
        mse_scores_rounded = np.around(mse_scores_mean, decimals=2).tolist()

        average_scores[seq_len] = {"r2": r2_scores_rounded, "mse": mse_scores_rounded}

    if imputation_method == "dtw_km":
        frechet_variances_mean = np.nanmean(frechet_variances_scores, axis=0)
        return average_scores, frechet_variances_mean
    else:
        return average_scores


def calculate_metric(x_true, x_pred, metric="r2"):
    """It takes in two arrays, `x_true` and `x_pred`, and calculates the `r2` or `mse` score

    Parameters
    ----------
    x_true
        the true values of the data
    x_pred
        the imputed values
    metric, optional
        the metric to use to evaluate the model.

    Returns
    -------
    the score of the model.

    """
    if metric == "r2":
        score = r2_score(x_true, x_pred)
    elif metric == "mse":
        score = mean_squared_error(x_true, x_pred)
    else:
        logging.info(f"Unknown metric {metric}.")
        return None

    return round(score, 2)
