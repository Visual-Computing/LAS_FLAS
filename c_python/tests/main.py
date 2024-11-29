#!/usr/bin/env python3

from PIL import Image
import numpy as np
from flas import flas, apply_sorting, GridBuilder

N_ALL_FEATURES = 10000
QUERY_SIZE = 100
HEIGHT, WIDTH = 64, 128
DIM = 3


def test_2d():
    all_features = np.random.random((N_ALL_FEATURES, DIM)).astype(np.float32)
    rng = np.random.default_rng()
    query_labels = rng.choice(a=N_ALL_FEATURES, size=QUERY_SIZE, replace=False, shuffle=False)

    query_features = all_features[query_labels]

    grid_builder = GridBuilder(aspect_ratio=1.0)
    grid_builder.put(
        query_features[0],
        (5, 5),
        query_labels[0],
    )
    grid_builder.add(
        features=query_features[1:],
        labels=query_labels[1:]
    )

    sorting = flas(grid_builder.build(), wrap=True)

    sorted_features = apply_sorting(query_features, sorting)
    # print(sorted_features, sorted_features.dtype, sorted_features.shape)

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


if __name__ == '__main__':
    test_2d()
    # test_1d()
