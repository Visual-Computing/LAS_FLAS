from typing import List

import numpy as np

from vc_flas import Grid, GridBuilder, flas

"""
The result of the FLAS algorithm is an Arrangement object.
It contains the positions of the features in the grid and can be used to create:
1. A features array with the same features as the input grid, but ordered so that similar features are close to each
   other.
2. A sorted two-dimensional array of objects associated with the features. For example, if you have a number of images
   and one feature vector for each image. An arrangement can be used to sort the images by the similarity of their
   feature vectors.
"""


def get_sorted_features():
    N, DIM = 100, 3
    colors = np.random.random((N, DIM))
    grid = Grid.from_features(colors, size=(10, 10))

    arrangement = flas(grid, wrap=True)

    sorted_colors = arrangement.get_sorted_features()
    assert sorted_colors.shape == (*grid.get_size(), DIM)

    # The grid can contain holes. Each sorted_feature[y, x] is filled with zeros.
    # This behaviour can be changed by using the hole_value argument.
    sorted_colors = arrangement.get_sorted_features(hole_value=1.0)
    print(sorted_colors.shape)

    # Interesting is also the arrangement.sorting field. The attribute arrangement.sorting[y, x] contains the index of
    # features[index] of the feature, that should be placed to [y, x] or -1 for a hole.
    print(arrangement.sorting.shape)


def working_with_labels():
    # often we have a more complicated case. Consider the following example:
    # We have 1000 image paths
    N = 1000
    all_images = ['image{}.png'.format(i) for i in range(N)]

    # and a feature embedding with 768 dimensions for each image
    DIM = 768
    all_features = np.random.random((N, DIM)).astype(np.float32)

    # now we take a subset of 120 of these images and want to sort them into a grid
    QUERY_SIZE = 120
    query_indices = np.random.default_rng().choice(a=N, size=QUERY_SIZE, replace=False, shuffle=False)
    _query_images = [all_images[index] for index in query_indices]
    query_features = all_features[query_indices]

    # we take the GridBuilder to place some images into a grid. While doing so, we define an integer for each feature,
    # that we call "label". In this case the label is the index of a feature/image in the all_features/all_images array.
    grid_builder = GridBuilder()
    grid_builder.put(
        query_features[0],
        (5, 5),
        labels=query_indices[0],  # <-- here we define the label for our first feature
    )
    grid_builder.add(
        features=query_features[1:],
        labels=query_indices[1:]  # <-- and here the labels for all other features
    )

    arrangement = flas(grid_builder.build())

    # now we can get the sorted features in two ways:
    sorted_features = arrangement.get_sorted_features()
    # or
    sorted_features_list: List[List[np.ndarray]] = arrangement.sort_by_labels(all_features)

    # we can use the same way to get a two-dimensional list of image-paths that are now sorted!
    sorted_image_paths: List[List[str]] = arrangement.sort_by_labels(all_images)


if __name__ == '__main__':
    get_sorted_features()
    working_with_labels()
