#!/usr/bin/env python3

from PIL import Image
import numpy as np
from flas import flas, apply_sorting, GridBuilder, Grid

N_ALL_FEATURES = 10000
HEIGHT, WIDTH = 64, 64
QUERY_SIZE = HEIGHT * WIDTH - 5
DIM = 3


def test_2d():
    all_features = np.random.random((N_ALL_FEATURES, DIM)).astype(np.float32)
    rng = np.random.default_rng()
    query_labels = rng.choice(a=N_ALL_FEATURES, size=QUERY_SIZE, replace=False, shuffle=False)

    query_features = all_features[query_labels]
    query_features[0] = np.array([1, 1, 1])

    grid_builder = GridBuilder(aspect_ratio=1.0)
    grid_builder.put(
        query_features[0],
        (32, 32),
        # query_labels[0],
    )
    grid_builder.add(
        features=query_features[1:],
        # labels=query_labels[1:]
    )

    arrangement = flas(grid_builder.build(freeze_holes=False), wrap=False, radius_decay=0.93)

    sorted_features = arrangement.get_sorted_features()

    image = Image.fromarray((sorted_features[:, :, :3] * 255).astype(np.uint8))
    image.save('images/image1.png', 'PNG')


def test_1d():
    features = np.random.random((HEIGHT * WIDTH, DIM)).astype(np.float32)

    sorting = flas(features, wrap=False)
    print('sorting.shape:', sorting.shape)
    print(sorting)

    sorted_features = apply_sorting(features, sorting)
    # print(sorted_features, sorted_features.dtype, sorted_features.shape)

    image = Image.fromarray((sorted_features[:, :, :3] * 255).astype(np.uint8))
    image.save('images/image1.png', 'PNG')


def create_grid_by_feature_list(n: int, dim: int, aspect_ratio: float = 1.0, seed: int = 1) -> Grid:
    rng = np.random.default_rng(seed)
    features = rng.random((n, dim)).astype(np.float32)
    # print(features, flush=True)
    return Grid.from_data(features, aspect_ratio=aspect_ratio)


def do_something():
    dim = 3
    n = 7
    aspect_ratio = 0.5
    for _ in range(10):
        grid = create_grid_by_feature_list(n, dim, aspect_ratio, seed=1)
        print(n, aspect_ratio, grid.get_size())
        result = flas(grid)
        pixels = result.get_sorted_features()
        image = Image.fromarray((pixels * 255).astype(np.uint8))
        image.save('images/test.png', 'PNG')

        print('done')


if __name__ == '__main__':
    test_2d()
    # test_1d()
    # do_something()
