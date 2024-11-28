import numpy as np

from flas import flas, Grid


HEIGHT, WIDTH, DIM = 13, 13, 3
N_SAMPLES = HEIGHT * WIDTH


def main():
    # sort features and choose grid layout
    features_1d = np.random.random((13, DIM)).astype(np.float32)
    flas(features_1d, aspect_ratio=16/9, wrap=True)

    # use given layout
    features_2d = np.random.random((13, 13, DIM)).astype(np.float32)
    frozen = np.zeros((13, 13)).astype(np.bool)
    valid = np.ones((13, 13)).astype(np.bool)
    flas(features_2d, frozen, valid)


def use_grid():
    # create a grid
    grid = Grid()  # grid with scalable size
    grid = Grid.with_size(HEIGHT, WIDTH)  #
    grid = Grid.from_data(np.random.random((HEIGHT, WIDTH, DIM)))
    grid = Grid.from_data(np.random.random((N_SAMPLES, DIM)))
