import numpy as np
from flas import Grid


DIM = 5


def main():
    grid = Grid()
    features = np.random.random((10, DIM))
    grid.add(features)
    pos = np.arange(10*2).reshape(10, 2)
    print(pos)
    grid.put(features, pos)
    grid.put(features, np.arange(10*2).reshape(10, 2))
    print(grid)


if __name__ == '__main__':
    main()
