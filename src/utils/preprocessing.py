"""Provides preprocessing functionality"""
import logging
from copy import deepcopy

import numpy as np
import numpy.typing as npt


logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')


def remove_values(x: npt.NDArray,
                  missing_rate: float = 0.2,
                  seq_len: int = 1,
                  seed=None) -> npt.NDArray:
    """
    Wrapper of the remove_values_1d function for 2D arrays.
    Removes random sequences of length `seq_len` from the input array `x` until desired `missing_rate` reached.

    Parameters
    ----------
    x : ndarray
        2D input data
    missing_rate : float
        The fraction of the data to be deleted.
    seq_len : int, optional
        The length of the sequence to be removed.
    seed : int, optional
        The seed for the random number generator.

    Returns
    -------
    ndarray
        Array of same length as input array with randomly deleted NaN sequences

    Returns
    -------

    """
    if not seed:
        seed = np.random.default_rng()

    x = x.copy()
    p_per_iteration = seq_len / x.shape[1]

    # Input sanity check
    if seq_len > x.shape[1]:
        logging.info(f"Higher sequence length {seq_len} than number of features {x.shape[1]}")
        return x
    elif p_per_iteration > missing_rate:
        logging.info(f"Sequence length {seq_len} too high for missing rate {missing_rate}")
        return x

    return np.apply_along_axis(_remove_values_1d, axis=1, arr=x, missing_rate=missing_rate, seq_len=seq_len, seed=seed)


def _remove_values_1d(x,
                      missing_rate=0.2,
                      seq_len=1,
                      seed=None):
    """Removes random sequences of length `seq_len` from the input array `x` until desired `missing_rate` reached.

        Parameters
        ----------
        x : ndarray
            1D input data
        missing_rate : float
            The fraction of the data to be deleted.
        seq_len : int, optional
            The length of the sequence to be removed.
        seed : int, optional
            The seed for the random number generator.

        Returns
        -------
        ndarray
            Array of same length as input array with randomly deleted NaN sequences

        """
    if not seed:
        seed = np.random.default_rng()
        
    current_p = 1
    p_per_iteration = seq_len / x.shape[0]
    removed_idx = np.array([])
    while current_p - p_per_iteration >= (1 - missing_rate):
        random_idx = seed.choice(range(x.shape[0]))
        upper_bound = random_idx + seq_len
        # Bounds check
        if upper_bound <= x.shape[0]:
            idx_to_remove = np.arange(random_idx, upper_bound)
            # Previously removed indices check
            if not np.any(np.intersect1d(idx_to_remove, removed_idx)):
                # Remove
                x[idx_to_remove] = np.nan
                removed_idx = np.append(removed_idx, idx_to_remove)
                current_p -= p_per_iteration
        else:
            continue
    return x


def group_by(X, cluster):
    """
    Used for DTW K-Means. It takes two arrays, and then splits the first array into groups based on the values in
    the second array.

    Parameters
    ----------
    X
        the array to be grouped
    cluster
        the array to group by

    Returns
    -------
    ndarray:
        X grouped by clusters

    """
    # Get argsort indices, to be used to sort a and b in the next steps
    sidx = cluster.argsort(kind='mergesort')
    a_sorted = X[sidx]
    b_sorted = cluster[sidx]

    # Get the group limit indices (start, stop of groups)
    cut_idx = np.flatnonzero(np.r_[True,b_sorted[1:] != b_sorted[:-1], True])

    # Split input array with those start, stop ones
    out = [a_sorted[i:j] for i,j in zip(cut_idx[:-1],cut_idx[1:])]
    return out


def summarize_km_and_knn_scores(scores,
                                seq_len_list,
                                missing_rate_list):
    """
    Takes the scores dictionary and returns a new dictionary with different cluster factors
    (DTW K-Means) and different neighbor sizes (KNN) summarized. Used for overall imputation performance plot.

    Parameters
    ----------
    scores
        the scores dictionary
    seq_len_list
        the list of sequence lengths used
    missing_rate_list
        list of missing rates used

    Returns
    -------
    dict:
        A dictionary with the following structure:
    {
        dataset_name: {
            method_name: {
                seq_len: {
                    "r2": [r2_score_for_missing_rate_1, r2_score_for_missing_rate_2, ...],
                    "mse": [mse_score_for_missing, ...]
                }
            }
        }
    }

    """
    methods_to_summarize = ["dtw_km", "dtw_knn"]

    summary = deepcopy(scores)
    for method_to_summarize in methods_to_summarize:
        for dataset in scores:
            for seq_len in seq_len_list:

                num_of_methods = len(scores[dataset])
                num_of_missing_rates = len(missing_rate_list)
                scores_r2 = np.full(fill_value=np.nan, shape=(num_of_methods, num_of_missing_rates))
                scores_mse = np.full(fill_value=np.nan, shape=(num_of_methods, num_of_missing_rates))

                for idx, method in enumerate(scores[dataset]):
                    if method.startswith(method_to_summarize):
                        scores_r2[idx] = scores[dataset][method][str(seq_len)]["r2"]
                        scores_mse[idx] = scores[dataset][method][str(seq_len)]["mse"]

                if method_to_summarize not in summary[dataset]:
                    summary[dataset][method_to_summarize] = {}

                summary[dataset][method_to_summarize][str(seq_len)] = {}
                summary[dataset][method_to_summarize][str(seq_len)]["r2"] = list(np.around(np.nanmean(scores_r2, axis=0), 2))
                summary[dataset][method_to_summarize][str(seq_len)]["mse"] = list(np.around(np.nanmean(scores_mse, axis=0), 2))

            for method in summary[dataset].copy().keys():
                if method.startswith(f"{method_to_summarize}_"):
                    del summary[dataset][method]

    return summary
