import time

import numpy as np

from vc_flas import Grid, flas, metrics


def create_feature_grid_arrangement(wrap: bool):
    features = np.random.random((60, 60, 3)).astype(np.float32)
    grid = Grid.from_grid_features(features)
    arrangement = flas(grid, wrap=wrap, radius_decay=0.999)
    return features, grid, arrangement


def calc_metrics():
    wrap = False
    features, grid, arrangement = create_feature_grid_arrangement(wrap)

    arrangement.get_sorted_features()

    start_time_dpq = time.perf_counter()
    sorted_dpq = arrangement.get_distance_preservation_quality()
    random_dpq = metrics.distance_preservation_quality(features, wrap=wrap)
    end_time_dpq = time.perf_counter()

    start_time_mnd = time.perf_counter()
    sorted_mnd = arrangement.get_mean_neighbor_distance()
    random_mnd = metrics.mean_neighbor_distance(features, wrap=wrap)
    end_time_mnd = time.perf_counter()

    print('METRIC  Random  Sorted  Runtime')
    print('DPQ     {:<7.3f} {:<7.3f} {:<8.4f}'.format(random_dpq, sorted_dpq, end_time_dpq - start_time_dpq))
    print('MND     {:<7.3f} {:<7.3f} {:<8.4f}'.format(random_mnd, sorted_mnd, end_time_mnd - start_time_mnd))


if __name__ == '__main__':
    calc_metrics()
