import numpy as np

from flas import flas


HEIGHT, WIDTH, DIM = 13, 13, 3


def main():
    # sort features and choose grid layout
    features_1d = np.random.random((13, DIM)).astype(np.float32)
    flas(features_1d, aspect_ratio=16/9, wrap=True)

    # use given layout
    features_2d = np.random.random((13, 13, DIM)).astype(np.float32)
    frozen = np.zeros((13, 13)).astype(np.bool)
    valid = np.ones((13, 13)).astype(np.bool)
    flas(features_2d, frozen, valid)


# Use grid class to give user fine-grained control over grid layout
class Grid:
    def __init__(self):
        self.grid = np.zeros((HEIGHT, WIDTH))

    def add(self, feature):
        ...

    def put(self, feature, x, y, frozen=False):
        self.grid[y, x] = feature

    def get(self, x, y):
        return self.grid[y, x]
