import numpy as np
from flas import Grid


DIM = 5


def main():
    grid_features = np.random.random((5, 5, DIM))
    random_features = np.random.random((290, DIM))

    # grid = Grid()
    grid = Grid.from_data(grid_features)
    grid.add(random_features)

    # pos = np.arange(random_features.shape[0] * 2).reshape(random_features.shape[0], 2)
    # grid.put(random_features, pos+11)
    random_features, taken, frozen = grid.compile(16 / 9)
    print('random_features:', random_features.shape, ' taken:', taken.shape, '  frozen:', frozen.shape)
    print(np.sum(taken), np.sum(np.logical_not(taken)))


if __name__ == '__main__':
    main()
