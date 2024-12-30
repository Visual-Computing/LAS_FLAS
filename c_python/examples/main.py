#!/usr/bin/env python3
import json

from PIL import Image
import numpy as np
from vc_flas import flas, Grid
from vc_flas.metrics import distance_preservation_quality, distance_ratio_to_optimum
from simple_tables import Table

N_ALL_FEATURES = 10000
HEIGHT, WIDTH = 256, 256
QUERY_SIZE = HEIGHT * WIDTH - 5
DIM = 3


class ProgressStopper:
    def __init__(self, p):
        self.p = p

    def __call__(self, p: float):
        return p >= self.p


def create_progress():
    rng = np.random.default_rng(42)
    features = rng.random((HEIGHT, WIDTH, DIM)).astype(np.float32)

    n_steps = 100
    for i in range(n_steps):
        arrangement = flas(features, radius_decay=0.93, callback=ProgressStopper((i + 1) / n_steps), seed=42)
        sorted_features = arrangement.get_sorted_features()

        image = Image.fromarray((sorted_features * 255).astype(np.uint8))
        image.save(f'images/gif/image{i:03}.png', 'PNG')
        print(f'saved image {i+1:02}')


def try_narrow():
    h, w, d = 100, 8, 3
    features = np.random.random((h, w, d))
    grid = Grid.from_grid_features(features)

    arrangement = flas(grid, radius_decay=0.93)

    sorted_features = arrangement.get_sorted_features()

    image = Image.fromarray((sorted_features * 255).astype(np.uint8))
    image.save('images/image1.png', 'PNG')


def try_normal():
    h, w, d = 124, 124, 3
    features = np.random.random((h, w, d))
    grid = Grid.from_grid_features(features)

    arrangement = flas(grid, radius_decay=0.93)

    sorted_features = arrangement.get_sorted_features()

    image = Image.fromarray((sorted_features * 255).astype(np.uint8))
    image.save('images/image1.png', 'PNG')


def example_2d():
    N, D = 241, 7
    features = np.random.random((N, D))
    grid = Grid.from_features(features)

    arrangement = flas(grid, wrap=True, radius_decay=0.99)

    sorted_features = arrangement.get_sorted_features()
    height, width, dim = sorted_features.shape
    print(height, width, dim)

    image = Image.fromarray((sorted_features[:, :, :3] * 255).astype(np.uint8))
    image.save('images/image1.png', 'PNG')


def test_1d():
    features = np.random.random((HEIGHT * WIDTH, DIM)).astype(np.float32)

    arrangement = flas(features, wrap=False)
    print('sorting.shape:', arrangement.get_size())
    print(arrangement)

    sorted_features = arrangement.get_sorted_features()
    # print(sorted_features, sorted_features.dtype, sorted_features.shape)

    image = Image.fromarray((sorted_features[:, :, :3] * 255).astype(np.uint8))
    image.save('images/image1.png', 'PNG')


def create_grid_by_feature_list(n: int, dim: int, aspect_ratio: float = 1.0, seed: int = 1) -> Grid:
    rng = np.random.default_rng(seed)
    features = rng.random((n, dim)).astype(np.float32)
    # print(features, flush=True)
    return Grid.from_features(features, aspect_ratio=aspect_ratio)


def import_and_try():
    table = Table(('Method', 'Size', 'Sorted DPQ', 'Random DPQ', 'Sorted RTO', 'Random RTO'), float_precision=3)
    files = [
        'data/export/export_SSM6_5-cols.json',
        'data/export/export_SSM6_8-cols.json',
        'data/export/export_SSM6_10-cols.json',
        'data/export/export_FLAS1_5-cols.json',
        'data/export/export_FLAS1_8-cols.json',
        'data/export/export_FLAS1_10-cols.json',
    ]
    for path in files:
        import_grid_and_calc_metric(path, table)

    print(table)


def import_grid_and_calc_metric(path, table):
    with open(path, 'r') as f:
        data = json.load(f)

    method_name = data['methodName']
    feature_count = int(data['featureCount'])
    size = data['rows'], data['columns']
    features = np.array(data['features'], dtype=np.float32)
    assert feature_count == features.shape[0], 'feature_count != np.prod(features.shape): {} != {}'.format(
        feature_count, features.shape[0])

    mapped_positions = np.array(data['mappedPositions'], dtype=np.int32)

    sorted_features = features[mapped_positions].reshape(*size, -1)
    features = features.reshape(*size, -1)

    sorted_dpq = distance_preservation_quality(sorted_features)
    random_dpq = distance_preservation_quality(features)
    sorted_rto = distance_ratio_to_optimum(sorted_features)
    random_rto = distance_ratio_to_optimum(features)

    table.line(
        method=method_name, size=str(size),
        sorted_dpq=sorted_dpq, random_dpq=random_dpq, sorted_rto=sorted_rto, random_rto=random_rto
    )


if __name__ == '__main__':
    # test_1d()
    # create_progress()
    # try_narrow()
    # example_2d()
    # try_normal()
    import_and_try()
