import numpy as np
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors
from tslearn import metrics

from dtw_mean import dtw_zero_cost_heuristic, frechet
from k_means import k_means
from utils.visualization import plot_k_means


def linear_interpolation(x_removed):
    """
    Linear interpolation of missing values in time series.

    Parameters
    ----------
    x_removed: ndarray
        time series with missing values

    Returns
    -------
    ndarray:
        Interpolated time series

    """
    x_removed = x_removed.copy()

    if not np.any(np.isnan(x_removed)):
        return x_removed

    for row in x_removed:
        if x_removed.ndim == 1:
            row = x_removed

        x = np.arange(0, row.shape[0])
        idx_missing_ = np.argwhere(np.isnan(row)).flatten()
        x = np.delete(x, idx_missing_, None)

        y = row
        y = np.delete(y, idx_missing_, None)
        f = interp1d(x, y, fill_value="extrapolate")

        xnew = idx_missing_
        ynew = f(xnew)
        row[idx_missing_] = ynew

    return x_removed


def arithmetic_mean_imputation(x_removed):
    """
    Replaces missing values with the mean of all time series at the same time.
    If the value is missing at an index for all time series, this index is imputed through linear interpolation.

    Parameters
    ----------
    x_removed: ndarray
        Time series with missing values

    Returns
    -------
    ndarray:
        Arithmetic mean imputed time series

    """
    x_removed = x_removed.copy()
    means_columns = np.nanmean(x_removed, axis=0)
    for row in x_removed:
        idx_missing = np.argwhere(np.isnan(row)).flatten()
        for i in idx_missing:
            row[i] = means_columns[i]

    # Linear interpolate if all values are missing at an index
    if np.isnan(x_removed).any():
        x_removed = linear_interpolation(x_removed)

    return x_removed


def dtw_knn_imputation(x_removed, neighbors):
    x_removed = np.copy(x_removed)
    x_interpolated = linear_interpolation(x_removed)

    # Get neighbors per time series
    neighbors = NearestNeighbors(n_neighbors=neighbors + 1, metric=metrics.dtw).fit(x_interpolated)
    neighbors_per_timeseries = neighbors.kneighbors(x_interpolated, return_distance=False)[:, 1:]

    # DTW Imputation
    x_imputed = np.full(x_removed.shape, np.nan)
    for i in range(x_interpolated.shape[0]):
        source = []
        for j in neighbors_per_timeseries[i]:
            source.append(np.ravel(x_interpolated[j]))

        x_imputed_part = np.full((len(source), x_removed.shape[1]), np.nan)
        for j, s in enumerate(source):
            imputed, _ = metrics.zc_dtw_imputation(s, x_removed[i])
            x_imputed_part[j] = imputed
        x_imputed_part = np.array(x_imputed_part, dtype=float)
        x_imputed[i] = np.mean(x_imputed_part, axis=0)

    return x_imputed


def dtw_kmeans_imputation(x_removed, class_labels, k, dist, mean, init='arithmetic', frechet_variance=False, plot=False):
    """
    Parameters
    ----------
    x_removed : ndarray with missing values
    class_labels: Boolean
        True, if first column contains class labels
    k : int
        number of clusters
    dist : 'eucl', 'dtw'
    mean : 'arithmetic', 'frechet'
    init: 'lin', 'arithmetic'
        initial Imputation Method
    frechet_variance: bool
        If True, calculate Fréchet variance
    plot: Boolean
        If True, plot the results

    Returns
    -------
    ndarray imputed TS
    """
    # remove first column if needed
    if class_labels:
        ts = np.copy(x_removed[:, 1:])
    else:
        ts = np.copy(x_removed)

    # initialisation
    X1 = np.copy(ts)
    if init == 'lin':
        X1 = linear_interpolation(x=X1)
    if init == 'arithmetic':
        X1 = arithmetic_mean_imputation(x_removed=X1)

    # k_means
    centroids, cluster = k_means(X1, k=k, dist=dist, mean=mean)
    if plot:
        plot_k_means(X1, centroids, cluster)

    # replace missing values with centroids
    x_imputed = np.copy(ts)

    for idx, t in enumerate(x_imputed):
        imputed, _ = metrics.zc_dtw_imputation(centroids[cluster[idx].astype(int)], t)
        x_imputed[idx] = imputed

    if frechet_variance:
        frechet_variance = np.full(fill_value=np.nan, shape=centroids.shape[0])

        for idx, c in enumerate(centroids):
            cluster_idx = np.argwhere(cluster == idx).flatten()

            if np.any(cluster_idx):
                frechet_variance[idx] = frechet(c, x_imputed[cluster_idx])

        return x_imputed, frechet_variance
    else:
        return x_imputed


def dtw_imputation(source,
                   target,
                   zero_cost =True,
                   dtw_path=None,
                   alignable=None):
    """ Imputes the missing values of the `target` time series with the average of
    the corresponding values of all `source` time series after alignment using DTW.

    Parameters
    ----------
    source : list(ndarray)
        List of source time series from which to impute
    target : ndarray
        The target time series to be imputed.
    zero_cost : bool, optional
        If True, the DTW path is computed using the zero-cost constraint.
        This is the default method.
    dtw_path : list(ndarray), optional
        List of DTW-paths between target and sources
    alignable : ndarray, optional
        Time series without missing values corresponding to target

    Returns
    -------
    ndarray
        The `target` time series with the missing values imputed.

    """
    target = np.copy(target)

    target = np.asarray(target, dtype=float)
    source = np.asarray(source, dtype=object)

    src_vals = []

    if dtw_path is None:
        assert alignable is not None or zero_cost, \
            "Error: `alignable` time series or `zero_cost` argument is " \
            "required to compute DTW path."

        if zero_cost:

            for s in source:
                s = s.astype(float)
                i, _ = metrics.zc_dtw_imputation(s, target)

                src_vals.append(np.ravel(s[i]).astype(float))

            src_vals = np.asarray(src_vals, dtype=float)
            mean_vals = np.mean(src_vals, axis=0)

        else:
            alignable = np.asarray(alignable, dtype=float)

            for s in source:
                s = s.astype(float)
                _, _, dtw_path = dtw_alignment(s, alignable)

                target_aligned, source_aligned, _ = dtw_alignment(s, target, dtw_path=dtw_path)

                src_vals.append(source_aligned[np.isnan(target_aligned)])

            src_vals = np.asarray(src_vals, dtype=float)
            mean_vals = np.mean(src_vals, axis=0)

    else:
        dtw_path = np.asarray(dtw_path, dtype=int)
        if np.ndim(dtw_path) == 1:
            dtw_path = dtw_path[:, np.newaxis]

        for i, s in enumerate(source):
            s = s.astype(float)
            target_aligned, source_aligned, _ = dtw_alignment(s, target, dtw_path=dtw_path[i])

            src_vals.append(source_aligned[np.isnan(target_aligned)])

        src_vals = np.asarray(src_vals, dtype=float)
        mean_vals = np.mean(src_vals, axis=0)

    assert target[np.isnan(target)].shape == mean_vals.shape, \
        f'The number of MVs in target {target[np.isnan(target)].shape} ' \
        f'and imputation values {mean_vals.shape} do not match.'

    target[np.isnan(target)] = mean_vals

    return target


def dtw_alignment(
        x,
        y,
        dtw_path=None,
        zero_cost=False):
    """Given two time series, `x` and `y` and an optional DTW path,
    `dtw_path`, return the aligned time series and the DTW path

    Parameters
    ----------
    x : ndarray
    y : ndarray
    dtw_path : ndarray, optional
        The path of the DTW alignment. If not provided, it will be calculated.
    zero_cost: bool, optional
        If True, use the zero-cost heuristic for DTW computation.

    Returns
    -------
    tuple(ndarray, ndarray, ndarray)
        The aligned x and y and the DTW-path.

    """
    if dtw_path is None:
        if zero_cost:
            _, dtw_path = dtw_zero_cost_heuristic(x, y, path=True)
        else:
            dtw_path, _ = metrics.dtw_path(x, y)
            dtw_path = np.array(dtw_path)

    x_aligned = x[dtw_path.astype(int).T[0]]
    y_aligned = y[dtw_path.astype(int).T[1]]

    assert len(x_aligned) == len(y_aligned), \
        "Error: Aligned lengths do not match."

    return x_aligned, y_aligned, dtw_path
