import time

import numpy as np
from PIL import Image

from vc_flas import Grid, flas, metrics


def create_feature_grid_arrangement(height: int, width: int, dim: int, wrap: bool):
    features = np.random.random((height, width, dim)).astype(np.float32)
    grid = Grid.from_grid_features(features)
    arrangement = flas(grid, wrap=wrap, radius_decay=0.93)
    return features, grid, arrangement


def create_features_grid_arrangement2(n: int, dim: int, wrap: bool):
    features = np.random.random((n, dim)).astype(np.float32)
    grid = Grid.from_features(features, aspect_ratio=1.5)
    n_padded = np.prod(grid.get_size())
    pad = np.zeros((n_padded - n, dim), dtype=np.float32)
    arrangement = flas(grid, wrap=wrap, radius_decay=0.93)
    features = np.concatenate([features, pad])
    features = features.reshape(*grid.get_size(), dim)
    return features, grid, arrangement


def try_metrics():
    wrap = True
    # features, grid, arrangement = create_feature_grid_arrangement(60, 60, 3, wrap)
    features, grid, arrangement = create_features_grid_arrangement2(60 * 60, 3, wrap)

    print('grid size:', grid.get_size())

    print('\nrandom features')
    calc_metrics(features, wrap)
    print('\nsorted features')
    calc_metrics(arrangement.get_sorted_features(), wrap)


def calc_metrics(features, wrap):
    start_time_dpq = time.perf_counter()
    random_dpq = metrics.distance_preservation_quality(features, wrap=wrap)
    end_time_dpq = time.perf_counter()

    start_time_mnd = time.perf_counter()
    random_mnd = metrics.mean_neighbor_distance(features, wrap=wrap)
    end_time_mnd = time.perf_counter()

    start_time_rto = time.perf_counter()
    random_rto = metrics.distance_ratio_to_optimum(features, wrap=wrap)
    end_time_rto = time.perf_counter()

    print('METRIC  Random  Runtime')
    print('DPQ     {:<7.3f} {:<8.4f}'.format(random_dpq, end_time_dpq - start_time_dpq))
    print('MND     {:<7.3f} {:<8.4f}'.format(random_mnd, end_time_mnd - start_time_mnd))
    print('RTO     {:<7.3f} {:<8.4f}'.format(random_rto, end_time_rto - start_time_rto))


def calc_image_metrics():
    path = 'images/rgb_4x4_2.png'
    image = Image.open(path)
    image = np.array(image) / 255.0
    height, width = image.shape[:2]
    print('image shape:', image.shape)

    print('\noriginal image')
    calc_metrics(image, wrap=False)
    print('\nshuffled image')
    dim = image.shape[-1]
    calc_metrics(np.random.permutation(image.reshape(-1, dim)).reshape(height, width, dim), wrap=False)


def calc_ratio_to_opt():
    wrap = True
    features, grid, arrangement = create_feature_grid_arrangement(60, 60, 3, wrap)

    sorted_features = arrangement.get_sorted_features()

    rto_sorted = metrics.distance_ratio_to_optimum(sorted_features, wrap=wrap)
    rto_random = metrics.distance_ratio_to_optimum(features, wrap=wrap)
    print(rto_sorted, rto_random)


def create_feature_plane(size):
    indices = np.indices(size).astype(np.float32)
    z_axis = np.zeros_like(indices[0])
    result = np.stack([indices[0], indices[1], z_axis])
    result = np.moveaxis(result, 0, 2)
    return result / np.max(result)


def ratio_to_opt_best_case():
    wrap = False
    features = create_feature_plane((10, 10))
    dist1 = np.linalg.norm(features[0, 9] - features[0, 8])
    dist2 = np.linalg.norm(features[0, 9] - features[1, 8])
    dist3 = np.linalg.norm(features[0, 0] - features[1, 1])
    print('dist [0,9] - [0,8]:', dist1)
    print('dist [0,9] - [1,8]:', dist2)
    print('dist [0,0] - [1,1]:', dist3)
    rat_to_opt = metrics.distance_ratio_to_optimum(features, wrap=wrap)
    print(rat_to_opt)

    image = Image.fromarray((features * 255).astype(np.uint8))
    image.save('images/image2.png', 'PNG')


if __name__ == '__main__':
    # calc_ratio_to_opt()
    # ratio_to_opt_best_case()
    # try_metrics()
    calc_image_metrics()
