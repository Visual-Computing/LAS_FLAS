import numpy as np

from vc_flas import Grid, GridBuilder, flas


def create_grid_by_feature_list(n: int, dim: int, aspect_ratio: float = 1.0) -> Grid:
    features = np.random.random((n, dim)).astype(np.float32)
    return Grid.from_features(features, aspect_ratio=aspect_ratio)


def test_grid_creations_list():
    dim = 3
    # for n in [4, 7, 25, 51, 60*60-1, 60*60]:
    for n in [4, 7, 25, 51, 60*60-1, 60*60]:
        for aspect_ratio in [1.0, 0.5, 2.0, 16 / 9]:
            grid = create_grid_by_feature_list(n, dim, aspect_ratio)
            flas(grid)


def create_grid_by_feature_map(height: int, width: int, dim: int) -> Grid:
    features = np.random.random((height, width, dim)).astype(np.float32)
    return Grid.from_grid_features(features)


def test_grid_creations_map():
    dim = 3
    for height in [2, 7, 25, 51, 60]:
        for width in [2, 7, 25, 51, 60]:
            grid = create_grid_by_feature_map(height, width, dim)
            flas(grid)


def create_grid_dynamic_builder(n: int, dim: int, aspect_ratio: float = 1.0) -> Grid:
    builder = GridBuilder(aspect_ratio=aspect_ratio)
    builder.add(np.random.random((n, dim)).astype(np.float32))
    builder.add(np.random.random((n, dim)).astype(np.float32))
    return builder.build()


def test_builder_dynamic():
    dim = 3
    for n in [4, 7, 25, 51, 60*60-1, 60*60]:
        for aspect_ratio in [1.0, 0.5, 2.0, 16 / 9]:
            grid = create_grid_dynamic_builder(n, dim, aspect_ratio)
            flas(grid)
