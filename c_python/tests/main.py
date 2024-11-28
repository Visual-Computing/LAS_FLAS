#!/usr/bin/env python3

from PIL import Image
import numpy as np
from flas import flas, apply_sorting


HEIGHT, WIDTH = 64, 64
DIM = 3


def test_2d():
    features = np.random.random((HEIGHT, WIDTH, DIM)).astype(np.float32)
    # print(features)
    frozen = np.zeros((HEIGHT, WIDTH)).astype(np.bool)
    frozen[:10, :10] = True
    # frozen = np.logical_not(frozen)
    # print(frozen)

    sorting = flas(features, frozen, wrap=True)

    sorted_features = apply_sorting(features, sorting)
    # print(sorted_features, sorted_features.dtype, sorted_features.shape)

    image = Image.fromarray((sorted_features[:, :, :3] * 255).astype(np.uint8))
    image.save('images/image1.png', 'PNG')


def test_1d():
    features = np.random.random((HEIGHT * WIDTH, DIM)).astype(np.float32)

    sorting = flas(features, aspect_ratio=16 / 9, wrap=True, freeze_holes=True)
    print('sorting.shape:', sorting.shape)
    print(sorting)

    sorted_features = apply_sorting(features, sorting)
    # print(sorted_features, sorted_features.dtype, sorted_features.shape)

    image = Image.fromarray((sorted_features[:, :, :3] * 255).astype(np.uint8))
    image.save('images/image1.png', 'PNG')


if __name__ == '__main__':
    # test_2d()
    test_1d()
