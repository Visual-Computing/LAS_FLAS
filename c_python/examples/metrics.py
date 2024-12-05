import time

import numpy as np

from vc_flas import Grid, flas, metrics


def create_feature_grid_arrangement(height: int, width: int, dim: int, wrap: bool):
    features = np.random.random((height, width, dim)).astype(np.float32)
    grid = Grid.from_grid_features(features)
    arrangement = flas(grid, wrap=wrap, radius_decay=0.93)
    return features, grid, arrangement


def calc_metrics():
    wrap = False
    features, grid, arrangement = create_feature_grid_arrangement(60, 60, 3, wrap)

    arrangement.get_sorted_features()

    start_time_dpq = time.perf_counter()
    sorted_dpq = arrangement.get_distance_preservation_quality()
    random_dpq = metrics.distance_preservation_quality(features, wrap=wrap)
    end_time_dpq = time.perf_counter()

    start_time_mnd = time.perf_counter()
    sorted_mnd = arrangement.get_mean_neighbor_distance()
    random_mnd = metrics.mean_neighbor_distance(features, wrap=wrap)
    end_time_mnd = time.perf_counter()

    start_time_rto = time.perf_counter()
    sorted_rto = metrics.distance_ratio_to_optimum(arrangement.get_sorted_features(), wrap=wrap)
    random_rto = metrics.distance_ratio_to_optimum(features, wrap=wrap)
    end_time_rto = time.perf_counter()

    print('METRIC  Random  Sorted  Runtime')
    print('DPQ     {:<7.3f} {:<7.3f} {:<8.4f}'.format(random_dpq, sorted_dpq, end_time_dpq - start_time_dpq))
    print('MND     {:<7.3f} {:<7.3f} {:<8.4f}'.format(random_mnd, sorted_mnd, end_time_mnd - start_time_mnd))
    print('DTO     {:<7.3f} {:<7.3f} {:<8.4f}'.format(random_rto, sorted_rto, end_time_rto - start_time_rto))


def calc_ratio_to_opt():
    wrap = True
    features, grid, arrangement = create_feature_grid_arrangement(60, 60, 3, wrap)

    sorted_features = arrangement.get_sorted_features()

    rto_sorted = metrics.distance_ratio_to_optimum(sorted_features, wrap=wrap)
    rto_random = metrics.distance_ratio_to_optimum(features, wrap=wrap)
    print(rto_sorted, rto_random)


def create_feature_plane(size):
    indices = np.indices(size).astype(np.float32)
    z_axis = np.sum(indices, axis=0)
    result = np.stack([indices[0], indices[1], z_axis])
    result = np.moveaxis(result, 0, 2)
    return result / np.max(result)


def ratio_to_opt_best_case():
    wrap = True
    features = create_feature_plane((20, 20))
    dist1 = np.linalg.norm(features[2, 2] - features[2, 3])
    dist2 = np.linalg.norm(features[2, 2] - features[3, 2])
    print('dists:', dist1, dist2)
    grid = Grid.from_grid_features(features)
    arrangement = flas(grid, wrap=wrap, radius_decay=0.93)
    arrangement.get_sorted_features()
    rat_to_opt = metrics.distance_ratio_to_optimum(arrangement.get_sorted_features(), wrap=wrap)
    print(rat_to_opt)


if __name__ == '__main__':
    # calc_ratio_to_opt()
    # ratio_to_opt_best_case()
    calc_metrics()
