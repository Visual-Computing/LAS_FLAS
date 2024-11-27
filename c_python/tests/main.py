#!/usr/bin/env python3

from PIL import Image
import numpy as np
from flas import flas, apply_sorting


HEIGHT, WIDTH = 64, 32
DIM = 3


class Grid:
    def __init__(self):
        self.grid = np.zeros((HEIGHT, WIDTH))

    def add(self, feature):
        ...

    def put(self, feature, x, y, frozen=False):
        self.grid[y, x] = feature

    def get(self, x, y):
        return self.grid[y, x]


def main():
    features = np.random.random((13, DIM)).astype(np.float32)
    frozen = [(10, 5)]
    features_2d = np.random.random((13, 13, DIM)).astype(np.float32)
    field = np.random.random((13, 13)).astype(np.uint8)
    valid = ...

    flas(features, aspect_ratio=16/9, wrap=True)
    flas(features_2d, frozen, valid)
    # features = np.random.random((HEIGHT, WIDTH, DIM)).astype(np.float32)
    # print(features)
    frozen = np.zeros((13, 1)).astype(np.bool)
    frozen[:10, :10] = True
    # frozen = np.logical_not(frozen)
    print(frozen)

    sorting = flas(features)

    sorted_features = apply_sorting(features, sorting)
    # print(sorted_features, sorted_features.dtype, sorted_features.shape)

    image = Image.fromarray((sorted_features[:, :, :3] * 255).astype(np.uint8))
    image.save('images/image1.png', 'PNG')


if __name__ == '__main__':
    main()

