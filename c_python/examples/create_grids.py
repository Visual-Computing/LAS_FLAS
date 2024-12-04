import numpy as np

from vc_flas import Grid, GridBuilder

"""
In order to use the FLAS implementation of vc_flas you have to create a Grid object.
A Grid contains all features in a array numpy array with shape (height, width, dim).
It also contains information about frozen fields and holes.

There are three ways of creating a Grid:
1. From a list of features
2. From a grid of features
3. From a GridBuilder
"""


def create_grid_from_features():
    # If you have features with shape (N, D)
    features = np.random.random((700, 3))

    # you can just put them in a Grid. The size of the grid will be determined automatically, depending on the
    # aspect_ratio or size argument.
    # In this case, it is possible that holes are created. These holes are always on the bottom right side.
    # As default these holes are frozen (cannot be moved by FLAS).
    # If you want FLAS to be able to move them, use freeze_holes=False.
    grid1 = Grid.from_features(features, aspect_ratio=16 / 9)
    grid2 = Grid.from_features(features, size=(35, 20), freeze_holes=True)
    print(grid1.get_size())
    print(grid2.get_size())


def create_grid_from_grid_features():
    # If you have feature_grid with shape (H, W, D)
    feature_grid = np.random.random((40, 32, 3))

    # you can put them in a Grid. The size of the grid will be (H, W) the same as the given feature_grid.
    grid = Grid.from_grid_features(feature_grid)
    print(grid.get_size())


def create_grid_with_builder():
    # If you have some features, that you want to put on a certain place (and freeze them there) and you want some other
    # features to be added somewhere to the grid, you can use the GridBuilder.
    feature_to_put_in_the_middle = np.array([0.5, 0.5, 0.5])
    features_to_add_somewhere = np.random.random((100, 3))

    # create builder with given size
    builder = GridBuilder(size=(21, 21))

    # put one feature in the middle of the map (and freeze it there).
    builder.put(feature_to_put_in_the_middle, (10, 10), frozen=True)

    # add other features somewhere into the map.
    builder.add(features_to_add_somewhere)

    grid = builder.build(freeze_holes=False)
    print(grid.get_size())


if __name__ == '__main__':
    create_grid_from_features()
    create_grid_from_grid_features()
    create_grid_with_builder()
