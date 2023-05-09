import logging

import numpy as np
import pandas as pd
from numpy import linalg as la
import warnings

import utils.preprocessing as preprocessing
from tslearn import metrics, barycenters


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')


def k_means(X, k, dist='dtw', mean='frechet'):
    """
    Parameters
    ----------
    X : list(ndarray)
        List of source time series from which to impute
    k : int
        number of clusters
    dist : 'eucl', 'dtw'
    mean : 'arithmetic', 'frechet'

    Returns
    -------
    ndarray centroids
    array cluster
    """
    diff = 1
    cluster = np.zeros(X.shape[0])
    centroids = pd.DataFrame(X).sample(n=k).values
    new_centroids = np.zeros((k, np.shape(X)[1]))

    while diff:
        for i, row in enumerate(X):
            mn_dist = float('inf')
            for idx, centroid in enumerate(centroids):
                if dist == 'eucl':
                    d = la.norm(centroid - row)
                    if mn_dist > d:
                        mn_dist = d
                        cluster[i] = idx
                elif dist == 'dtw':
                    d = metrics.dtw(centroid, row)
                    if mn_dist > d:
                        mn_dist = d
                        cluster[i] = idx

        if mean == 'arithmetic':
            new_centroids = pd.DataFrame(X).groupby(by=cluster).mean().values
        elif mean == 'frechet':
            X_grouped = preprocessing.group_by(X, cluster)  # group X into its clusters
            for i in range(0, len(X_grouped)):
                new_centroids[i] = np.squeeze(barycenters.dtw_barycenter_averaging_subgradient(X_grouped[i]))

        if np.count_nonzero(centroids - new_centroids) == 0:
            diff = 0
        else:
            centroids = new_centroids

    return centroids, cluster
