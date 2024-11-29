import numpy as np

from flas import Grid, GridBuilder


def create_grid_by_feature_list(n: int, dim: int, aspect_ratio: float = 1.0) -> Grid:
    features = np.random.random((n, dim)).astype(np.float32)
    return Grid.from_data(features, aspect_ratio=aspect_ratio)


def test_grid_creations_list():
    dim = 3
    for n in [1, 7, 25, 51, 60*60-1, 60*60]:
        for aspect_ratio in [1.0, 0.5, 2.0, 16 / 9]:
            _grid = create_grid_by_feature_list(n, dim, aspect_ratio)


def create_grid_by_feature_map(height: int, width: int, dim: int) -> Grid:
    features = np.random.random((height, width, dim)).astype(np.float32)
    return Grid.from_data(features)


def test_grid_creations_map():
    dim = 3
    for height in [1, 7, 25, 51, 60]:
        for width in [1, 7, 25, 51, 60]:
            _grid = create_grid_by_feature_map(height, width, dim)


def create_grid_dynamic_builder(n: int, dim: int, aspect_ratio: float = 1.0) -> Grid:
    builder = GridBuilder(aspect_ratio=aspect_ratio)
    builder.add(np.random.random((n, dim)).astype(np.float32))
    builder.add(np.random.random((n, dim)).astype(np.float32))
    return builder.build()


def test_builder_dynamic():
    dim = 3
    for n in [1, 7, 25, 51, 60*60-1, 60*60]:
        for aspect_ratio in [1.0, 0.5, 2.0, 16 / 9]:
            _grid = create_grid_dynamic_builder(n, dim, aspect_ratio)
