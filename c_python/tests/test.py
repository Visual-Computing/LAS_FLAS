import numpy as np
from flas import Grid


DIM = 5


def main():
    start_features = np.random.random((10, 10, DIM))
    features = np.random.random((3, DIM))

    # grid = Grid()
    grid = Grid.from_data(start_features)
    grid.add(features)

    pos = np.arange(features.shape[0] * 2).reshape(features.shape[0], 2)
    # grid.put(features, pos+11)
    features, taken, frozen = grid.compile(16 / 9)
    print('features:', features.shape, ' taken:', taken.shape, '  frozen:', frozen.shape)
    print(np.sum(taken), np.sum(np.logical_not(taken)))


if __name__ == '__main__':
    main()
