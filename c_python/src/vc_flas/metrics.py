import numpy as np


def mean_neighbor_distance(
        sorted_features: np.ndarray, valid: np.ndarray | None = None, wrap: bool = False, ord: int = 2
) -> float:
    """
    Calculate the mean squared distances between neighbors of sorted features.
    If holes are present, they do not contribute to the distance.

    :param sorted_features: features array with shape (h, w, d).
    :param valid: None or array with shape (h, w), where each has a 1 except holes having a 0.
    :param wrap: If True, distances are calculated for opposite edges of the grid as well.
    :param ord: The p-norm to use. Defaults to L2-norm.
    :return:
    """
    if sorted_features.ndim != 3:
        raise ValueError("sorted_features must have shape (h, w, d), got: ".format(sorted_features.shape))

    if wrap:
        x_orig = sorted_features
        x_moved = np.roll(sorted_features, 1, axis=1)
        y_orig = sorted_features
        y_moved = np.roll(sorted_features, 1, axis=0)
    else:
        x_orig = sorted_features[:, :-1]
        x_moved = sorted_features[:, 1:]
        y_orig = sorted_features[:-1]
        y_moved = sorted_features[1:]

    diffs_x = np.linalg.norm(x_orig - x_moved, axis=-1, ord=ord)
    diffs_y = np.linalg.norm(y_orig - y_moved, axis=-1, ord=ord)

    num_present_x = np.prod(diffs_x.shape)
    num_present_y = np.prod(diffs_y.shape)

    if valid is not None:
        if valid.shape != sorted_features.shape[:-1]:
            raise ValueError("holes must have same size as features (h, w) = {}, got: {}".format(
                    sorted_features.shape[:-1], valid.shape))
        present = valid.astype(np.uint8)
        if wrap:
            present_x = present * np.roll(present, 1, axis=1)
            present_y = present * np.roll(present, 1, axis=0)
        else:
            present_x = present[:, :-1] * present[:, 1:]
            present_y = present[:-1] * present[1:]
        num_present_x = np.sum(present_x)
        num_present_y = np.sum(present_y)

        diffs_x = diffs_x * present_x
        diffs_y = diffs_y * present_y

    sum_x = 0
    if num_present_x:
        sum_x = np.sum(diffs_x) / num_present_x

    sum_y = 0
    if num_present_y:
        sum_y = np.sum(diffs_y) / num_present_y

    return sum_x + sum_y


def ratio_to_optimum(features: np.ndarray, wrap: bool = False):
    pass


def distance_preservation_quality(sorted_x: np.ndarray, wrap: bool = False, p: int = 2):
    """
    Computes the Distance Preservation Quality DPQ_p(S)

    :param sorted_x: sorted features with shape (h, w, d).
    :param wrap: Whether the grid should be wrapped around.
    :param p: The p-norm to use. Defaults to L2-norm.
    """
    # setup of required variables
    grid_shape = sorted_x.shape[:-1]
    n = np.prod(grid_shape)
    _h, _w = grid_shape
    flat_x = sorted_x.reshape((n, -1))

    # compute matrix of Euclidean distances in the high dimensional space
    dists_hd = np.sqrt(_squared_l2_distance(flat_x, flat_x))

    # sort HD distance matrix rows in ascending order (first value is always 0 zero now)
    sorted_d = np.sort(dists_hd, axis=1)

    # compute the expected value of the HD distance matrix
    mean_d = sorted_d[:, 1:].mean()

    # compute spatial distance matrix for each position on the 2D grid
    dists_spatial = _compute_spatial_distances_for_grid(grid_shape, wrap)

    # sort rows of HD distances by the values of spatial distances
    sorted_hd_by_2_d = _sort_hd_dists_by_2d_dists(dists_hd, dists_spatial)

    # get delta DP_k values
    delta_dp_k_2d = _get_distance_preservation_gain(sorted_hd_by_2_d, mean_d)
    delta_dp_k_hd = _get_distance_preservation_gain(sorted_d, mean_d)

    # compute p norm of DP_k values
    normed_delta_d_2d_k = np.linalg.norm(delta_dp_k_2d, ord=p)
    normed_delta_d_hd_k = np.linalg.norm(delta_dp_k_hd, ord=p)

    # DPQ(s) is the ratio between the two normed DP_k values
    return normed_delta_d_2d_k/normed_delta_d_hd_k


def _get_distance_preservation_gain(sorted_d_mat, d_mean):
    """
    Computes the Distance Preservation Gain delta DP_k(S)
    """
    # range of numbers [1, K], with K = N-1
    nums = np.arange(1, len(sorted_d_mat))

    # compute cumulative sum of neighbor distance values for all rows, shape = (N, K)
    cum_sum = np.cumsum(sorted_d_mat[:, 1:], axis=1)

    # compute average of neighbor distance values for all rows, shape = (N, K)
    d_k = (cum_sum / nums)

    # compute average of all rows for each k, shape = (K,)
    d_k = d_k.mean(axis=0)

    # compute Distance Preservation Gain and set negative values to 0, shape = (K,)
    return np.clip((d_mean - d_k) / d_mean, 0, np.inf)


def _compute_spatial_distances_for_grid(grid_shape, wrap):
    """
    Converts a given grid_shape to a grid index matrix and calculates the squared spatial distances
    """
    if wrap:
        return _compute_spatial_distances_for_grid_wrapped(grid_shape)
    else:
        return _compute_spatial_distances_for_grid_non_wrapped(grid_shape)


def _compute_spatial_distances_for_grid_wrapped(grid_shape):
    n_x = grid_shape[0]
    n_y = grid_shape[1]

    wrap1 = [[0,   0], [0,   0], [0,     0], [0, n_y], [0,   n_y], [n_x, 0], [n_x,   0], [n_x, n_y]]
    wrap2 = [[0, n_y], [n_x, 0], [n_x, n_y], [0,   0], [n_x,   0], [0  , 0], [0  , n_y], [0  ,   0]]

    # create 2D position matrix with tuples, i.e. [(0,0), (0,1)...(H-1, W-1)]
    a, b = np.indices(grid_shape)
    mat = np.concatenate([np.expand_dims(a, -1), np.expand_dims(b, -1)], axis=-1)
    mat_flat = mat.reshape((-1, 2))

    # use this 2D matrix to calculate spatial distances between on positions on the grid
    d = _squared_l2_distance(mat_flat, mat_flat)
    for i in range(8):
        # look for smaller distances with wrapped coordinates
        d_i = _squared_l2_distance(mat_flat + wrap1[i], mat_flat + wrap2[i])
        d = np.minimum(d, d_i)

    return d


def _compute_spatial_distances_for_grid_non_wrapped(grid_shape):
    # create 2D position matrix with tuples, i.e. [(0,0), (0,1)...(H-1, W-1)]
    a, b = np.indices(grid_shape)
    mat = np.concatenate([np.expand_dims(a, -1), np.expand_dims(b, -1)], axis=-1)
    mat_flat = mat.reshape((-1, 2))

    # use this 2D matrix to calculate spatial distances between on positions on the grid
    return _squared_l2_distance(mat_flat, mat_flat)


def _sort_hd_dists_by_2d_dists(hd_dists, ld_dists):
    """
    Sorts a matrix so that row values are sorted by the spatial distance and in case they are equal, by the HD distance
    """
    max_hd_dist = np.max(hd_dists) * 1.0001

    ld_hd_dists = hd_dists/max_hd_dist + ld_dists  # add normed HD dists (0 .. 0.9999) to the 2D int dists
    ld_hd_dists = np.sort(ld_hd_dists)  # then a normal sorting of the rows can be used

    return np.fmod(ld_hd_dists, 1) * max_hd_dist


def _squared_l2_distance(q, p):
    """
    Calculates the squared L2 (Euclidean) distance using numpy.
    :param q: numpy array with shape (n,) or (n, m)
    :param p: numpy array with shape (n,) or (n, m)
    """
    ps = np.sum(np.square(p), axis=-1, keepdims=True)
    qs = np.sum(np.square(q), axis=-1, keepdims=True)
    distance = ps - 2*np.matmul(p, q.T) + qs.T
    return np.maximum(distance, 0)


__all__ = ['distance_preservation_quality', 'mean_neighbor_distance']
