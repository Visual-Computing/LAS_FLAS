import numpy as np

from vc_flas import flas, Grid, metrics


def create_feature_grid_arrangement(wrap: bool):
    features = np.random.random((10, 10, 7)).astype(np.float32)
    grid = Grid.from_grid_features(features)
    arrangement = flas(grid, wrap=wrap)
    return features, grid, arrangement


def test_distance_preservation_quality():
    wrap = False
    features, grid, arrangement = create_feature_grid_arrangement(wrap)
    sorted_features = arrangement.get_sorted_features()

    sorted_dpq = metrics.distance_preservation_quality(sorted_features, wrap=wrap)
    random_dpq = metrics.distance_preservation_quality(features, wrap=wrap)

    if sorted_dpq < random_dpq:
        raise ValueError('sorted_dpq < random_dpq')


def test_mean_neighbor_distance():
    wrap = False
    features, grid, arrangement = create_feature_grid_arrangement(wrap)
    sorted_features = arrangement.get_sorted_features()

    sorted_mnd = metrics.mean_neighbor_distance(sorted_features, wrap=wrap)
    random_mnd = metrics.mean_neighbor_distance(features, wrap=wrap)

    if sorted_mnd > random_mnd:
        raise ValueError('sorted_mnd > random_mnd')
