import numpy as np


def compute_spatial_distances_for_grid(grid_shape, wrap):
    """
    Converts a given grid_shape to a grid index matrix and calculates the squared spatial distances
    """
    if wrap:
        return compute_spatial_distances_for_grid_wrapped(grid_shape)
    else:
        return compute_spatial_distances_for_grid_non_wrapped(grid_shape)


def compute_spatial_distances_for_grid_wrapped(grid_shape):
    n_x = grid_shape[0]
    n_y = grid_shape[1]

    wrap1 = [[0,   0], [0,   0], [0,     0], [0, n_y], [0,   n_y], [n_x, 0], [n_x,   0], [n_x, n_y]]
    wrap2 = [[0, n_y], [n_x, 0], [n_x, n_y], [0,   0], [n_x,   0], [0  , 0], [0  , n_y], [0  ,   0]]

    # create 2D position matrix with tuples, i.e. [(0,0), (0,1)...(H-1, W-1)]
    a, b = np.indices(grid_shape)
    mat = np.concatenate([np.expand_dims(a, -1), np.expand_dims(b, -1)], axis=-1)
    mat_flat = mat.reshape((-1, 2))

    # use this 2D matrix to calculate spatial distances between on positions on the grid
    d = squared_l2_distance(mat_flat, mat_flat)
    for i in range(8):
        # look for smaller distances with wrapped coordinates
        d_i = squared_l2_distance(mat_flat + wrap1[i], mat_flat + wrap2[i])
        d = np.minimum(d, d_i)

    return d


def compute_spatial_distances_for_grid_non_wrapped(grid_shape):
    # create 2D position matrix with tuples, i.e. [(0,0), (0,1)...(H-1, W-1)]
    a, b = np.indices(grid_shape)
    mat = np.concatenate([np.expand_dims(a, -1), np.expand_dims(b, -1)], axis=-1)
    mat_flat = mat.reshape((-1, 2))

    # use this 2D matrix to calculate spatial distances between on positions on the grid
    return squared_l2_distance(mat_flat, mat_flat)


def get_distance_preservation_gain(sorted_d_mat, d_mean):
    """
    Computes the Distance Preservation Gain delta DP_k(S)
    """
    # range of numbers [1, K], with K = N-1
    nums = np.arange(1, len(sorted_d_mat))

    # compute cumulative sum of neighbor distance values for all rows, shape = (N, K)
    cumsum = np.cumsum(sorted_d_mat[:, 1:], axis=1)

    # compute average of neighbor distance values for all rows, shape = (N, K)
    d_k = (cumsum / nums)

    # compute average of all rows for each k, shape = (K,)
    d_k = d_k.mean(axis=0)

    # compute Distance Preservation Gain and set negative values to 0, shape = (K,)
    return np.clip((d_mean - d_k) / d_mean, 0, np.inf)


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
    dists_hd = np.sqrt(squared_l2_distance(flat_x, flat_x))

    # sort HD distance matrix rows in ascending order (first value is always 0 zero now)
    sorted_d = np.sort(dists_hd, axis=1)

    # compute the expected value of the HD distance matrix
    mean_d = sorted_d[:, 1:].mean()

    # compute spatial distance matrix for each position on the 2D grid
    dists_spatial = compute_spatial_distances_for_grid(grid_shape, wrap)

    # sort rows of HD distances by the values of spatial distances
    sorted_hd_by_2_d = sort_hddists_by_2d_dists(dists_hd, dists_spatial)

    # get delta DP_k values
    delta_dp_k_2d = get_distance_preservation_gain(sorted_hd_by_2_d, mean_d)
    delta_dp_k_hd = get_distance_preservation_gain(sorted_d, mean_d)

    # compute p norm of DP_k values
    normed_delta_d_2d_k = np.linalg.norm(delta_dp_k_2d, ord=p)
    normed_delta_d_hd_k = np.linalg.norm(delta_dp_k_hd, ord=p)

    # DPQ(s) is the ratio between the two normed DP_k values
    return normed_delta_d_2d_k/normed_delta_d_hd_k


def sort_hddists_by_2d_dists(hd_dists, ld_dists):
    """
    Sorts a matrix so that row values are sorted by the spatial distance and in case they are equal, by the HD distance
    """
    max_hd_dist = np.max(hd_dists) * 1.0001

    ld_hd_dists = hd_dists/max_hd_dist + ld_dists  # add normed HD dists (0 .. 0.9999) to the 2D int dists
    ld_hd_dists = np.sort(ld_hd_dists)  # then a normal sorting of the rows can be used

    return np.fmod(ld_hd_dists, 1) * max_hd_dist


def squared_l2_distance(q, p):
    """
    Calculates the squared L2 (Euclidean) distance using numpy.
    """
    ps = np.sum(p*p, axis=-1, keepdims=True)
    qs = np.sum(q*q, axis=-1, keepdims=True)
    distance = ps - 2*np.matmul(p, q.T) + qs.T
    return np.maximum(distance, 0)
