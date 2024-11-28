import numpy as np
from flas import Grid


DIM = 5


def main():
    grid = Grid()
    features = np.random.random((10, DIM))
    grid.add(features)
    pos = np.arange(10*2).reshape(10, 2)
    grid.put(features, pos)
    features, taken, frozen = grid.compile(16 / 9)
    print('features:', features.shape, ' taken:', taken.shape, '  frozen:', frozen.shape)
    print(np.sum(taken), np.sum(np.logical_not(taken)))


if __name__ == '__main__':
    main()
