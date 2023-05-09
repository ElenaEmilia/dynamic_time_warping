"""Provides functionality to load UCR datasets"""

# pylint: disable=E0401

import csv

import numpy as np
import pandas as pd

DATA_PATH = "data/"


def load_ucr(name, class_label=None, keep_label=False, trailing_nan=False, _dtype=float):
    """Load defined UCR dataset using `Numpy` functionality

    Parameters
    ----------
    name: str
        Dataset to load
    class_label: int
        Class to extract from the dataset
    keep_label: bool
        Keep first index containing the class label
    trailing_nan: bool
        Remove trailing NaN values for times series with different length
    _dtype: type
        Desired data type for the dataset

    Returns
    -------
    tuple
        tuple of train and test data as `Numpy` arrays

    """
    test_set = np.loadtxt(f"{DATA_PATH}/{name}/{name}_TEST.tsv", dtype=_dtype)
    train_set = np.loadtxt(f"{DATA_PATH}/{name}/{name}_TRAIN.tsv", dtype=_dtype)

    if class_label is not None:
        train_set = train_set[train_set[:, 0] == class_label]
        test_set = test_set[test_set[:, 0] == class_label]

    # Optionally remove class label
    if not keep_label:
        test_set = test_set[:, 1:]
        train_set = train_set[:, 1:]

    # Behavior when varying length of time series
    if not trailing_nan:
        if np.any(np.isnan(test_set)):
            test_set = np.array([np.array(a[~np.isnan(a)]) for a in test_set], dtype=object)
        if np.any(np.isnan(train_set)):
            train_set = np.array([np.array(a[~np.isnan(a)]) for a in train_set], dtype=object)

    return train_set, test_set


def load_ucr_pd(name, keep_label=False, _dtype=float):
    """Load defined UCR dataset using `Pandas` functionality

    Parameters
    ----------
    name: str
        Dataset to load
    keep_label: bool
        Keep first index containing the class label
    _dtype: type
        Desired data type for the dataset

    Returns
    -------
    tuple
        tuple of train and test data as `DataFrames`

    """
    test_set = pd.read_csv(f"{DATA_PATH}/{name}/{name}_TEST.tsv",
                           sep="\t",
                           dtype=_dtype,
                           header=None)
    train_set = pd.read_csv(f"{DATA_PATH}/{name}/{name}_TRAIN.tsv",
                            sep="\t",
                            dtype=_dtype,
                            header=None)

    # Optionally remove class label
    if not keep_label:
        test_set = test_set.iloc[:, 1:]
        train_set = train_set.iloc[:, 1:]

    return train_set, test_set


def load_ucr_csv(name, keep_label=False, trailing_nan=False, _dtype=float):
    """Load defined UCR dataset using `csv` module functionality

    Parameters
    ----------
    name: str
        Dataset to load
    keep_label: bool
        Keep first index containing the class label
    _dtype: type
        Desired data type for the dataset

    Returns
    -------
    tuple
        tuple of train and test data `Numpy` arrays

    """
    test_set = []
    train_set = []

    with open(f"{DATA_PATH}/{name}/{name}_TEST.tsv") as file:
        reader = csv.reader(file, delimiter="\t")
        for line in reader:
            test_set.append(line)

    with open(f"{DATA_PATH}/{name}/{name}_TRAIN.tsv") as file:
        reader = csv.reader(file, delimiter="\t")
        for line in reader:
            train_set.append(line)

    train_set = np.array(train_set, dtype=_dtype)
    test_set = np.array(test_set, dtype=_dtype)

    # Optionally remove class label
    if not keep_label:
        test_set = test_set[:, 1:]
        train_set = train_set[:, 1:]

    # Behavior when varying length of time series
    if not trailing_nan:
        if np.any(np.isnan(test_set)):
            test_set = np.array([np.array(a[~np.isnan(a)]) for a in test_set], dtype=object)
        if np.any(np.isnan(train_set)):
            train_set = np.array([np.array(a[~np.isnan(a)]) for a in train_set], dtype=object)

    return train_set, test_set
