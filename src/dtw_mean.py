"""Stoachstic Subgradient (SSG) Method for Averaging Time Series
under Dynamic Time Warping (DTW), and DTW Zero Cost Heuristic.

Translation by Khaled Sakallah, based on the Matlab code
of the SSG algorithm in https://doi.org/10.5281/zenodo.216233
Adapated from David Schultz, DAI-Lab, TU Berlin, Germany, 2017
"""

# pylint: disable=E0401

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist
from tqdm import tqdm
from tslearn import metrics

import imputation


def ssg(X, n_epochs=None, eta=None, init_sequence=None, return_f=False):
    """ Calculates an approximate sample mean under dynamic time warping
    using the Stochastic Subgradient method.

    Parameters
    ----------
    X: ndarray
        n x d matrix consisting of time series n time series with length d each | OR
        array (of arrays), suitable for time series with different length
    n_epochs: int
        number of epochs
    eta: ndarray
        vector of step sizes, eta(i) is used in the i-th update
    init_sequence: array_like
        if it is a time series --> use it
        if None  --> use a random sample of
        if > 0   --> use X[init_sequence]
        if <= 0  --> use medoid of X
    return_f: bool
        if True  --> Frechet variations for each epoch are returned

    Returns
    -------
    z: ndarray
        solution found by SSG (an approximate sample mean under dynamic time warping)
    f: ndarray
        vector of Frechet variations. Is only returned if return_f=True

    """

    N = X.shape[0]  # number of samples

    if n_epochs is None:
        n_updates = 1000
        n_epochs = int(np.ceil(n_updates / N))

    if eta is None:
        eta = np.linspace(0.1, 0.01, N)

    # initialize mean z
    if init_sequence is None:
        z = X[np.random.randint(N)]

    elif init_sequence > 0:
        z = X[int(init_sequence)]

    elif init_sequence <= 0:
        z = medoid_sequence(X)

    z = imputation.linear_interpolation(z)

    if return_f:
        f = np.zeros(n_epochs + 1)
        f[0] = frechet(z, X)

    # stochastic subgradient optimization
    with tqdm(total=n_epochs * N) as pbar:
        for k in range(1, n_epochs + 1):
            perm = np.random.permutation(N)
            for i in range(1, N + 1):
                pbar.update(1)
                x_i = X[perm[i - 1]]
                # x_i = imputation.linear_interpolation(x_i)

                # Check for missing values (nan)
                if pd.isna(X).any():
                    # _, p = dtw_zero_cost_heuristic(z, x_i, path=True)
                    p, _ = metrics.zc_dtw_path(z, x_i)
                    p = np.array(p)
                else:
                    p, _ = metrics.dtw_path(z, x_i)
                    p = np.array(p)

                W, V = get_warp_val_mat(p)

                subgradient = 2 * (V * z - W.dot(x_i))

                c = (k - 1) * N + i
                if c <= eta.shape[0]:
                    lr = eta[c - 1]
                else:
                    lr = eta[-1]

                # update rule
                z = z - lr * subgradient

            if return_f:
                f[k] = frechet(z, X)

    if return_f:
        f = f[0:n_epochs + 1]
        return z, f

    else:
        return z


def dtw(x, y, path=False, zero_cost=None, impute=False, return_impute_idx=False):
    """The function takes two time series, `x` and `y`, and returns the
    DTW-distance between them.

    Parameters
    ----------
    x: ndarray
        the first time series [n]
    y: ndarray
        the second time series [m],
        may have X missing values when used with `zero_cost`
    path: bool, optional
        If True, return the warping path.
    zero_cost: ndarray, optional
        a list of indices of `y` that are missing values [X]
    impute: bool, optional
        If True, imputes the time series with missing values
        Overrides `path` to True
        Overrides `return_impute_idx` to True
    return_impute_idx: bool, optional
        If True, returns the indices of x to use for imputation

    Returns
    -------
    int
        The DTW-distance between the two time series
    ndarray, optional
        The DTW-path of length L between the two time series [L,2]
    ndarray, optional
        The indices of the TS without missing values to use for imputation [X]

    """
    x = x.astype(float)
    y = y.astype(float)

    if return_impute_idx:
        impute = True

    return_imputed = impute
    if impute:
        path = True
        target = np.copy(y)

    N = x.shape[0]
    M = y.shape[0]

    # cdist requires 2-dim array
    if len(x.shape) == 1:
        x = x[:, np.newaxis]
    if len(y.shape) == 1:
        y = y[:, np.newaxis]

    D = cdist(x, y) ** 2
    # Heuristic "Zero cost" - Set column in distance matrix to zero at missing value indices
    if zero_cost is not None:
        D[:, zero_cost] = 0

    C = np.zeros((N, M))
    C[:, 0] = np.cumsum(D[:, 0])
    C[0, :] = np.cumsum(D[0, :])

    for n in range(1, N):
        for m in range(1, M):
            if (zero_cost is not None) and (m in zero_cost):
                C[n, m] = D[n, m] + min(C[n - 1, m - 1], C[n, m - 1])
            else:
                C[n, m] = D[n, m] + min(C[n - 1, m - 1], C[n - 1, m], C[n, m - 1])

    d = np.sqrt(C[N - 1, M - 1])

    # % compute warping path p
    if path:
        n = N - 1
        m = M - 1
        p = np.zeros((N + M - 1, 2))
        p[-1, :] = (n, m)
        k = 1

        while n + m >= 0:
            if (zero_cost is not None) and (m in zero_cost):
                if impute:
                    target[m] = x[n]

            if n == 0:
                m = m - 1
            elif m == 0:
                n = n - 1
                impute = False

            else:
                C_diag = C[n - 1, m - 1]
                C_r = C[n, m - 1]
                C_d = C[n - 1, m]

                if (zero_cost is not None) and (m in zero_cost):
                    steps = [C_diag, C_r]
                else:
                    steps = [C_diag, C_r, C_d]

                min_step = np.argmin(steps)
                if min_step == 0:  # C_diag is min
                    n, m = n - 1, m - 1
                elif min_step == 1:  # C_r is min
                    m = m - 1
                elif min_step == 2:  # C_d is min
                    n = n - 1

            p[-1 - k, :] = (n, m)
            k = k + 1

        p = p[-1 - k + 1:, :]

        if return_impute_idx:
            return d, p, target, zero_cost
        elif return_imputed:
            return d, p, target
        else:
            return d, p

    return d


def dtw_zero_cost_heuristic(x, y, path=False, impute=False, return_impute_idx=False):
    """DTW call for zero cost heuristic  passing the indices of the missing values

    Parameters
    ----------
    x: ndarray
        the first time series
    y: ndarray
        the time series to be compared to x
    path: bool, optional
        If True, return the path aswell.
    impute: bool, optional
        If True, imputes the time series with missing values
        Overrides `path` to True
        Overrides `return_impute_idx` to True
    return_impute_idx: bool, optional
        If True, returns the indices of x to use for imputation

    Returns
    -------
    int
        The DTW-distance between the two time series
    ndarray, optional
        The DTW-path between the two time series
    ndarray, optional
        The indices of the TS without missing values to use for imputation

    """
    x = np.ravel(x).astype(float)
    y = np.ravel(y).astype(float)
    has_nans = [np.isnan(x).any(), np.isnan(y).any()]

    assert not np.all(has_nans), "Only one time series can have missing values"

    if has_nans[0]:
        missing = np.argwhere(np.isnan(x))
        r = dtw(y, x, path=path, zero_cost=missing, impute=impute, return_impute_idx=return_impute_idx)
        r = list(r)
        r[1] = r[1][:, [1,0]]
        return tuple(r)

    if has_nans[1]:
        missing = np.argwhere(np.isnan(y))
        return dtw(x, y, path=path, zero_cost=missing, impute=impute, return_impute_idx=return_impute_idx)

    return dtw(x, y, path=path)


def frechet(x, X):
    N = X.shape[0]
    f = 0
    for i in range(N):
        dist = metrics.dtw(x, X[i])
        f = f + dist**2

    f = f / N
    return f


def medoid_sequence(X):
    """
    Compute the medoid sequence of a set of time series. A medoid is an element of X that
    minimizes the Frechet function among all elements in X

    Parameters
    ----------
    X : ndarray
        A set of time series.

    Returns
    -------
    ndarray
        The medoid sequence of the set of time series.
    """
    N = X.shape[0]
    f_min = np.inf
    i_min = 0
    for i in range(N):
        f = frechet(X[i], X)
        if f < f_min:
            f_min = f
            i_min = i

    x = X[i_min]
    return x


def get_warp_val_mat(warping_path):
    """
    Takes a warping path and returns the warping matrix and the valence matrix.

    Parameters
    ----------
    warping_path
        a list of tuples, where each tuple is a (i,j) pair.

    Returns
    -------
    ndarray
        W is the (sparse) warping matrix of p
    ndarray
        V is a vector representing the diagonal of the valence matrix

    """
    if type(warping_path) is not np.ndarray:
        warping_path = np.array(warping_path)

    L = warping_path.shape[0]
    N = int(warping_path[-1, 0]) + 1
    M = int(warping_path[-1, 1]) + 1
    W = coo_matrix((np.ones(L), (warping_path[:, 0], warping_path[:, 1])), shape=(N, M)).toarray()
    V = np.sum(W, axis=1, keepdims=True)[:, 0]
    return W, V
