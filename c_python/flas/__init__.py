from typing import Optional, List, Tuple, Any

import numpy as np
import flas_cpp


class Grid:
    def __init__(self, features: np.ndarray, ids: np.ndarray, frozen: np.ndarray, labels: np.ndarray):
        """
        Creates a new grid.

        :param features: Numpy array with shape (n, d), where n is the number of features and d is the dimensionality
                         of each feature.
        :param ids: Numpy array with shape (h, w). Each id is the index into the features array. If id[y, x] == -1, this
                    marks a hole.
        :param frozen: Boolean array with shape (h, w). If frozen[y, x] == True, the field at (y, x) cannot be moved.
        :param labels: The external labels with shape (n,) and dtype uint32.
        """
        n, dim = features.shape
        height, width = ids.shape
        if frozen.shape != (height, width):
            raise ValueError('frozen must have shape (h, w) but got: {}'.format(frozen.shape))
        if labels.shape != (n,):
            raise ValueError('labels must have shape (n,) but got: {}'.format(labels.shape))
        if np.max(ids) >= n:
            raise ValueError('ids must be smaller than n but got id={} (n={})'.format(np.max(ids), n))

        if features.dtype != np.float32:
            raise ValueError('features must have dtype float32 but got: {}'.format(features.dtype))
        if ids.dtype != np.int32:
            raise ValueError('ids must have dtype int32 but got: {}'.format(ids.dtype))
        if frozen.dtype != np.bool:
            raise ValueError('frozen must have dtype bool but got: {}'.format(frozen.dtype))
        if not np.isdtype(labels.dtype, 'integral'):
            raise ValueError('labels must have integer dtype but got: {}'.format(labels.dtype))

        self.features = features
        self.ids = ids
        self.frozen = frozen
        self.labels = labels

    @classmethod
    def from_data(cls, features: np.ndarray, aspect_ratio: float = 1.0, freeze_holes: bool = True):
        """
        Creates a new grid from the given features.

        :param features: numpy array with shape (h, w, d) or (n, d).
        :param aspect_ratio: The preferred aspect ratio of the grid. This is only used if features has shape (n, d).
                             Otherwise, the dimensions of the given features are used.
        :param freeze_holes: Sometimes holes are needed, to create a grid with the given aspect ratio. In this case this
                             parameter defines whether holes should be frozen.
        :return:
        """
        if features.dtype != np.float32:
            features = features.astype(np.float32)

        if features.ndim == 3:
            height, width, dim = features.shape
            n = height * width
            return Grid(
                features=features.reshape(n, dim),
                ids=np.arange(n, dtype=np.int32).reshape(height, width),
                frozen=np.zeros((height, width), dtype=np.bool),
                labels=np.arange(n, dtype=np.uint32),
            )
        elif features.ndim == 2:  # 1d features case
            n, dim = features.shape
            height, width = flas_cpp.get_optimal_grid_size(n, aspect_ratio, 2, 2)

            # ids
            ids = np.empty(height * width, dtype=np.int32)
            ids[n:] = -1
            ids[:n] = np.arange(n, dtype=np.int32)

            # frozen
            frozen = np.zeros(height * width, dtype=np.bool)
            if freeze_holes:
                frozen[n:] = True  # freeze holes
            frozen = frozen.reshape(height, width)

            return Grid(
                features=features,
                ids=ids.reshape(height, width),
                frozen=frozen,
                labels=np.arange(n, dtype=np.uint32),
            )
        else:
            raise ValueError('features must have shape (h, w, d) or (n, d) but got: {}'.format(features.shape))


class GridBuilder:
    def __init__(
            self, size: Tuple[int, int] = (1, 1), aspect_ratio: float = 1.0, check_overwrite: bool = False, dim: int = 0
    ):
        self.size: Tuple[int, int] = size
        self.dim: int = dim
        self.grid_features: Optional[np.ndarray] = None
        self.grid_labels: Optional[np.ndarray] = None
        self.grid_frozen: Optional[np.ndarray] = None
        self.lazy_features: List[Tuple[np.ndarray, np.ndarray]] = []
        self.aspect_ratio = aspect_ratio
        self.check_overwrite = check_overwrite

    def __repr__(self):
        dim = self.dim
        if dim == 0:
            dim = 'unknown'
        return 'Grid(size={}, dim={})'.format(self.size, dim)

    def add(self, features: np.ndarray, labels: int | np.ndarray):
        """
        Add the given features anywhere to the grid.
        :param features: numpy array with shape (d,) or (n, d), where d is the feature dimensionality and n is the
                         number of features.
        :param labels: integer or numpy array with shape (n,), where n is the number of features. Each label at
                       labels[i] identifies one feature.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.ndim != 2:
            raise ValueError('features must have shape (n, d) but got: {}'.format(features.shape))

        if isinstance(labels, int):
            labels = np.array([labels])
        if features.shape[:-1] != labels.shape:
            raise ValueError('features and labels must have same size. features.shape != labels.shape: {} != {}'.format(
                features.shape, labels.shape))
        if not np.isdtype(labels.dtype, 'integral'):
            raise ValueError('labels must be an integer array but got: {}'.format(labels.dtype))

        self._check_newdim(features.shape[-1])
        self.lazy_features.append((features, labels))

    def put(
            self, features: np.ndarray, pos: Tuple[int, int] | np.ndarray,
            labels: int | np.ndarray, frozen: bool | np.ndarray = True
    ):
        """
        Put the given features to the given position.

        :param features: numpy array with shape (n, d) or (d,), where d is the feature dimensionality and n is the
                         number of features.
        :param pos: tuple (y, x) or numpy array with shape (n, 2) or (2,), where n is the number of positions to put
                    the features to. Should have the same n as features.
        :param labels: integer or numpy array with shape (n,), where n is the number of features. Each labels at
                       labels[i] identifies feature[i].
        :param frozen: bool or numpy array with shape (n,), defining whether the given features should be frozen.
        """
        self._check_newdim(features.shape[-1])

        features = features.reshape(-1, self.dim)
        n_features = features.shape[0]

        # check pos
        if isinstance(pos, tuple):
            pos = np.array(pos)
        old_pos_shape = pos.shape
        pos = pos.reshape(-1, 2)
        if pos.shape != (n_features, 2):
            raise ValueError('pos must have shape (n, 2) or (2,) but got: {}'.format(old_pos_shape))

        # check frozen
        if isinstance(frozen, bool):
            frozen = np.full(n_features, frozen)
        if frozen.shape != (n_features,):
            raise ValueError('frozen must have shape (n,) but got: {}'.format(frozen.shape))

        # check ids
        if np.isscalar(labels):
            labels = np.array([labels])
        if features.shape[:-1] != labels.shape:
            raise ValueError('features and labels must have same size. features.shape != labels.shape: {} != {}'.format(
                features.shape, labels.shape))
        if not np.isdtype(labels.dtype, 'integral'):
            raise ValueError('ids must be an integer array but got: {}'.format(labels.dtype))

        # resize grid
        new_size = np.max(pos, axis=0) + 1
        if self.size is not None:
            new_size = np.maximum(new_size, self.size)
        self._resize(new_size)

        # check overwrite
        if self.check_overwrite:
            if np.any(self.grid_labels[tuple(pos.T)] >= 0):
                raise ValueError('Position is already taken')

        # save features
        self.grid_labels[tuple(pos.T)] = labels
        self.grid_features[tuple(pos.T)] = features
        self.frozen[tuple(pos.T)] = frozen

    def get_features(self, pos: Tuple[int, int] | np.ndarray) -> np.ndarray:
        """
        Get the features at the given position.
        :param pos: The position to get the feature from. Can be a tuple (y, x) or a numpy array with shape (n, 2) where
                    n is the number of positions to get.
        :return: The features at the given positions
        """
        if isinstance(pos, np.ndarray):
            pos = tuple(pos.T)

        return self.grid_features[pos]

    def get_labels(self, pos: Tuple[int, int] | np.ndarray) -> np.ndarray:
        """
        Get the labels at the given position.
        :param pos: The position to get the labels from. Can be a tuple (y, x) or a numpy array with shape (n, 2) where
                    n is the number of positions to get.
        :return: The features at the given positions
        """
        if isinstance(pos, np.ndarray):
            pos = tuple(pos.T)

        return self.grid_labels[pos]

    def build(self, freeze_holes: bool = True) -> Grid:
        """
        Builds a grid consisting of three numpy arrays, which can be used for the flas algorithm.

        A grid consists of three numpy arrays: (features, labels, frozen).
                 features is a numpy array with shape (h, w, d), where d is the feature dimensionality and h and w are
                 height and width of the feature plane.
                 labels is an int numpy array with shape (h, w), where -1 indicates that the feature is a hole. Any other
                 number is the id of the feature.
                 frozen is a boolean numpy array with shape (h, w), where True indicates that the feature is frozen
                 (should not be moved).
        :param freeze_holes: If True, holes are frozen.
        """
        if self.grid_labels is None or np.sum(self.grid_labels != -1) == 0:  # only dynamic features
            n_features = self._num_lazy_features()
            height, width = flas_cpp.get_optimal_grid_size(n_features, self.aspect_ratio, 2, 2)

            # features
            features = np.concatenate([f for f, _ in self.lazy_features])
            features = features.astype(np.float32)

            # ids
            n_missing_features = height * width - n_features
            ids = np.concatenate([np.arange(n_features), np.full(n_missing_features, -1)])
            ids = ids.reshape(height, width).astype(np.int32)

            # frozen
            frozen = np.zeros((height * width), dtype=np.bool)
            if freeze_holes:
                frozen[n_features:] = freeze_holes
            frozen = frozen.reshape(height, width)

            # labels
            labels = np.concatenate([labels for _, labels in self.lazy_features])
            labels = labels.astype(np.int32)

            return Grid(features, ids, frozen, labels)
        else:
            num_static_features = np.sum(self.grid_labels != -1)
            num_lazy_features = self._num_lazy_features()
            total_num_features = num_static_features + num_lazy_features

            if total_num_features == 0:
                raise ValueError('building empty grid')

            # get dimensions to fit all lazy features
            height, width = flas_cpp.get_optimal_grid_size(
                total_num_features, self.aspect_ratio, *self.grid_labels.shape
            )

            # features
            static_indices = np.nonzero(self.grid_labels != -1)
            features = self.grid_features[static_indices]
            if self.lazy_features:
                features = [features] + [f for f, _ in self.lazy_features]
                features = np.concatenate(features)

            # ids
            ids = np.full((height, width), -1, dtype=np.int32)
            ids[static_indices] = np.arange(num_static_features)
            if self.lazy_features:
                dynamic_indices = np.nonzero(ids == -1)
                dynamic_indices = tuple(fi[:num_lazy_features] for fi in dynamic_indices)
                ids[dynamic_indices] = np.arange(num_static_features, total_num_features)

            # labels
            labels = self.grid_labels[static_indices]
            if self.lazy_features:
                labels = [labels] + [l for _, l in self.lazy_features]
                labels = np.concatenate(labels)
            labels = labels.astype(np.uint32)

            # frozen
            frozen = _embed_array(self.frozen, (height, width))
            if freeze_holes:
                hole_indices = np.nonzero(ids == -1)
                frozen[hole_indices] = True

            return Grid(features, ids, frozen, labels)

    def _check_newdim(self, new_dim):
        if self.dim == 0:
            self.dim = new_dim
            if self.size is not None:
                self._init_grids()
        else:
            if self.dim != new_dim:
                raise ValueError(
                    'Features must have same dimension as previous features but got old_dim != new_dim: {} != {}'
                    .format(self.dim, new_dim)
                )

    def _resize(self, new_size: Tuple[int, int]):
        self.size = new_size
        if self.grid_features is not None:
            self.grid_features = _embed_array(self.grid_features, self.size)
            self.grid_labels = _embed_array(self.grid_labels, self.size, fill_value=-1)
            self.frozen = _embed_array(self.frozen, self.size)
        elif self.dim != 0:
            self._init_grids()

    def _init_grids(self):
        self.grid_features = np.zeros((*self.size, self.dim), dtype=np.float32)
        self.grid_labels = np.full(self.size, -1, dtype=np.int32)
        self.frozen = np.zeros(self.size, dtype=np.bool)

    def _num_lazy_features(self) -> int:
        return sum(f.shape[0] for f, _ in self.lazy_features)


class Arrangement:
    def __init__(self, grid: Grid, sorting: np.ndarray):
        """
        Creates a new arrangement.
        :param grid: The grid that was sorted.
        :param sorting: The sorting of the grid with shape (h, w).
        """
        if sorting.shape != grid.ids.shape[:2]:
            raise ValueError('sorting must have shape {} but got: {}'.format(grid.ids, sorting.shape))

        self.grid = grid
        self.sorting = sorting

    def get_size(self) -> Tuple[int, int]:
        """
        :returns: the size of the map as tuple (height, width).
        """
        return self.sorting.shape

    def get_sorted_features(self, hole_value: float | np.ndarray = 0.0) -> np.ndarray:
        """
        Sort the features according to the given sorting. Holes are filled with the given hole_value.
        :param hole_value: Scalar or numpy array with shape (d,) defining the value to fill holes with.
        :return: A numpy array with shape (h, w, d) containing the sorted features.
        """
        dim = self.grid.features.shape[-1]
        if np.isscalar(hole_value):
            hole_value = np.full(dim, hole_value)
        hole_value = hole_value.reshape(1, dim)

        # add hole feature to end of features, so that sorting of -1 accesses this feature
        features = np.concatenate([self.grid.features, hole_value])
        return features[self.sorting.flatten()].reshape(*self.sorting.shape, dim)

    def get_sorted_labels(self) -> np.ndarray:
        """
        Returns a numpy array with shape (h, w) containing the labels of the sorted features.
        get_sorted_labels()[y, x] contains the label to the feature, that was moved to (y, x).
        If get_sorted_labels()[y, x] == -1, there is a hole.
        """
        labels = np.concatenate([self.grid.labels, [-1]])
        return labels[self.sorting.flatten()].reshape(self.sorting.shape)


def flas(grid: Grid | np.ndarray, wrap: bool = False, radius_decay: float = 0.93, max_swap_positions: int = 9,
         weight_swappable: float = 1.0, weight_non_swappable: float = 100.0, weight_hole: float = 0.01) -> Arrangement:
    """
    Sorts the given features into a 2d plane, so that similar features are close together.
    See https://github.com/Visual-Computing/LAS_FLAS for details.

    :param grid: A numpy array with shape (height, width, dims) of type float32 (otherwise it will be cast).
    :param wrap: If True, the features on the right side will be similar to features on the left side as well as
                 features on top of the plane will be similar to features on the bottom.
    :param radius_decay: How much should the filter radius decay at each iteration.
    :param max_swap_positions: Number of possible swaps to hand over to solver. Should be a square number.
    :param weight_swappable:
    :param weight_non_swappable:
    :param weight_hole:
    :return: a 2d numpy array with shape (height, width). The cell at (y, x) contains the index of the feature that
             should be at (y, x). Indices are in scanline order.
    """
    if isinstance(grid, Grid):
        pass
    elif isinstance(grid, np.ndarray):
        if grid.dtype != np.float32:
            grid = grid.astype(np.float32)
        grid = Grid.from_data(grid)
    else:
        raise TypeError('features must be a Grid or a numpy array but got: {}'.format(type(grid)))

    # TODO: return identity sorting
    if np.all(grid.frozen):
        raise ValueError('All features are frozen. Cannot sort features.')

    code, result = flas_cpp.flas(
        grid.features, grid.ids, grid.frozen, wrap, radius_decay, weight_swappable, weight_non_swappable, weight_hole,
        max_swap_positions
    )

    if code != 0:
        raise RuntimeError('FLAS failed with error code {}'.format(code))

    return Arrangement(grid, result)


def _embed_array(array: np.ndarray, new_size: Tuple[int, int], fill_value: Any = 0) -> np.ndarray:
    """
    Embeds a 3D array into a new array with a larger shape.

    Parameters:
    :param array:  Input array with shape [h, w, d].
    :param new_size: Tuple (new_h, new_w), the shape of the new array's first two dimensions.

    :return: New array with shape [new_h, new_w, d], containing the original array.
    """
    h, w = array.shape[:2]
    rest_shape = array.shape[2:]
    new_h, new_w = new_size

    if new_h < h or new_w < w:
        raise ValueError("New size must be greater than or equal to the original size.")

    # Create a new array filled with fill value
    new_array = np.full((new_h, new_w, *rest_shape), fill_value, dtype=array.dtype)

    # Embed the original array into the new array
    new_array[:h, :w] = array

    return new_array


def apply_sorting(features, sorting):
    """
    Returns the given features rearranged according to the given sorting.

    :param features: A numpy array with shape (height, width, ...). Will get rearranged depending on sorting.
    :param sorting: A numpy array with shape (height, width) of any int type. Every position defines the index of the
                    features that should take this position. For example the sorting
                        [[0, 1, 2],
                         [3, 4, 5],
                         [6, 7, 8]]
                    would not alter the features at all.
    :return: The sorted features.
    """
    height, width = sorting.shape
    dim = features.shape[-1]
    # features as 1d array. Append zero element at the end, so that sorting indices == -1 will access this element.
    features_1d = np.concatenate([features.reshape(-1, dim), np.zeros_like(features, shape=(1, dim))])
    sorted_features = features_1d[sorting.flatten()]
    return sorted_features.reshape(height, width, dim)
