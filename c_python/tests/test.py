import numpy as np
from flas import Grid


DIM = 5


def main():
    grid = Grid.with_size(10, 12)
    features = np.random.random((10, DIM))
    grid.add(features)
    grid.put(features, np.arange(10*2).reshape(10, 2))
    # grid.put(features, np.arange(10*2).reshape(10, 2))
    print(grid)


if __name__ == '__main__':
    main()
