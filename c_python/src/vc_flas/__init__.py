from typing import Optional, List, Tuple, Any, Sequence, Mapping, Callable

from . import metrics

import numpy as np
import flas_cpp


__version__ = "0.1.6"


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
        if frozen.dtype != bool:
            raise ValueError('frozen must have dtype bool but got: {}'.format(frozen.dtype))
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError('labels must have integer dtype but got: {}'.format(labels.dtype))

        self.features = features
        self.ids = ids
        self.frozen = frozen
        self.labels = labels

    def __repr__(self):
        return 'Grid(size={} n_holes={})'.format(self.get_size(), self.get_num_holes())

    def get_size(self) -> Tuple[int, int]:
        """
        :return: the size of the grid as tuple (height, width).
        """
        return self.ids.shape

    def get_num_features(self) -> int:
        """
        :return: the number of features in the grid. Holes do not count.
        """
        return self.features.shape[0]

    def get_num_holes(self) -> int:
        """
        :return: The number of holes in the grid.
        """
        return np.prod(self.ids.shape) - self.get_num_features()

    def get_holes(self) -> np.ndarray:
        """
        :returns: A two-dimensional numpy bool array with shape (h, w). holes[y, x] is True if the feature at position
                  (y, x) is a hole.
        """
        return np.equal(self.ids, -1)

    def get_valid(self) -> np.ndarray:
        """
        :returns: A two-dimensional numpy bool array with shape (h, w). valid[y, x] is True if the feature at position
                  (y, x) is not a hole.
        """
        return np.not_equal(self.ids, -1)

    @classmethod
    def from_grid_features(cls, features: np.ndarray, labels: np.ndarray | None = None):
        """
        Creates a new grid from the given features.

        :param features: numpy array with shape (h, w, d).
        :param labels: numpy integer array with shape (h, w) or (h*w,), where h and w are the height and width of the
                       grid.
        :return: The new grid with the given features.
        """
        if features.dtype != np.float32:
            features = features.astype(np.float32)
        if features.ndim != 3:
            raise ValueError('features must have shape (h, w, d) but got: {}'.format(features.shape))
        height, width, dim = features.shape
        n = height * width

        # labels
        if labels is None:
            labels = np.arange(n, dtype=np.int32)
        else:
            if labels.shape != (height, width) and labels.shape != (n,):
                raise ValueError('labels must have shape (h, w) or (h*w,) but got: {}'.format(labels.shape))
            if not np.issubdtype(labels.dtype, np.integer):
                raise ValueError('ids must be an integer array but got: {}'.format(labels.dtype))
            if np.any(labels < 0):
                raise ValueError('labels must be >= 0 but got label {}'.format(np.min(labels)))
            labels = labels.reshape(n)

        return Grid(
            features=features.reshape(n, dim),
            ids=np.arange(n, dtype=np.int32).reshape(height, width),
            frozen=np.zeros((height, width), dtype=bool),
            labels=labels,
        )

    @classmethod
    def from_features(
            cls, features: np.ndarray, size: Tuple[int, int] | None = None, aspect_ratio: float = 1.0,
            freeze_holes: bool = True, labels: np.ndarray | None = None
    ):
        """
        Creates a new grid from the given features.

        :param features: numpy array with shape (n, d).
        :param size: The size of the grid. If None, the size is determined automatically using aspect ratio.
        :param aspect_ratio: The preferred aspect ratio of the grid. If size is given, this is ignored.
        :param freeze_holes: Sometimes holes are needed, to create a grid with the given aspect ratio. In this case this
                             parameter defines whether holes should be frozen.
        :param labels: numpy array with shape (n,), where n is the number of features.
                       Each label at labels[i] identifies one feature. If None labels are created automatically.
        :return:
        """
        if features.ndim != 2:
            raise ValueError('features must have shape (n, d) but got: {}'.format(features.shape))

        if features.dtype != np.float32:
            features = features.astype(np.float32)

        n, dim = features.shape

        if size is not None:
            height, width = size
            if height * width < n:
                raise ValueError('size ({}, {}) is too small for {} features'.format(height, width, n))
        else:
            height, width = flas_cpp.get_optimal_grid_size(n, aspect_ratio, 2, 2)

        # ids
        ids = np.empty(height * width, dtype=np.int32)
        ids[:n] = np.arange(n, dtype=np.int32)
        ids[n:] = -1

        # frozen
        frozen = np.zeros(height * width, dtype=bool)
        if freeze_holes:
            frozen[n:] = True  # freeze holes
        frozen = frozen.reshape(height, width)

        # labels
        if labels is None:
            labels = np.arange(n, dtype=np.int32)
        else:
            if labels.shape != (n,):
                raise ValueError('labels must have shape (n,) but got: {}'.format(labels.shape))
            if not np.issubdtype(labels.dtype, np.integer):
                raise ValueError('ids must be an integer array but got: {}'.format(labels.dtype))
            if np.any(labels < 0):
                raise ValueError('labels must be >= 0 but got label {}'.format(np.min(labels)))

        return Grid(
            features=features,
            ids=ids.reshape(height, width),
            frozen=frozen,
            labels=labels,
        )


class Labeler:
    def __init__(self):
        self.next_label = 0

    def get_labels(self, features_shape: Tuple[int, int] | Tuple[int, int, int]):
        """
        Get new labels for the given features.
        :param features_shape: The shape of the features to create labels for. Either (h, w, d) or (n, d).
        :return:
        """
        if len(features_shape) == 2:
            n = features_shape[0]
        elif len(features_shape) == 3:
            n = features_shape[0] * features_shape[1]
        else:
            raise ValueError('features_shape must have shape (h, w, d) or (n, d) but got: {}'.format(features_shape))
        result = np.arange(self.next_label, self.next_label + n, dtype=np.int32)
        self.next_label += n
        return result

    def update(self, labels: int | np.ndarray):
        """
        Update internal state with labels from user.
        :param labels: The labels of the user with
        """
        self.next_label = max(np.max(labels)+1, self.next_label)

    def update_or_create(self, labels: int | np.ndarray | None, features_shape: Tuple[int, int] | Tuple[int, int, int]):
        """
        Creates new labels, if not given, and updates internal state with labels from user.

        :param features_shape: The shape of the features to create labels for. Either (h, w, d) or (n, d).
        :param labels: The labels of the user with
        :return: The given labels if not None, otherwise the created labels for the given features.
        """
        if labels is None:
            return self.get_labels(features_shape)
        else:
            self.update(labels)
            return labels


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
        self.labeler = Labeler()

    def __repr__(self):
        dim = self.dim
        if dim == 0:
            dim = 'unknown'
        return 'Grid(size={}, dim={})'.format(self.size, dim)

    def add(self, features: np.ndarray, labels: int | np.ndarray | None = None) -> np.ndarray:
        """
        Add the given features anywhere to the grid.
        :param features: numpy array with shape (d,) or (n, d), where d is the feature dimensionality and n is the
                         number of features.
        :param labels: integer or numpy array with shape (n,), where n is the number of features. Each label at
                       labels[i] identifies one feature. If None labels are created automatically.
        :returns: The labels of the added features. If given, the labels are returned. Otherwise, the labels are
                  created.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.ndim != 2:
            raise ValueError('features must have shape (n, d) but got: {}'.format(features.shape))

        if features.dtype != np.float32:
            features = features.astype(np.float32)

        labels = self.labeler.update_or_create(labels, features.shape)
        labels = self._check_valid_labels(features, labels)

        self._check_newdim(features.shape[-1])
        self.lazy_features.append((features, labels))

        return labels

    def put(
            self, features: np.ndarray, pos: Tuple[int, int] | np.ndarray,
            labels: int | np.ndarray | None = None, frozen: bool | np.ndarray = True
    ) -> np.ndarray:
        """
        Put the given features to the given position.

        :param features: numpy array with shape (n, d) or (d,), where d is the feature dimensionality and n is the
                         number of features.
        :param pos: tuple (y, x) or numpy array with shape (n, 2) or (2,), where n is the number of positions to put
                    the features to. Should have the same n as features.
        :param labels: integer or numpy array with shape (n,), where n is the number of features. Each labels at
                       labels[i] identifies feature[i]. If None labels are created automatically.
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

        # check labels
        labels = self.labeler.update_or_create(labels, features.shape)
        labels = self._check_valid_labels(features, labels)

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
        if features.dtype != np.float32:
            features = features.astype(np.float32)

        self.grid_labels[tuple(pos.T)] = labels
        self.grid_features[tuple(pos.T)] = features
        self.frozen[tuple(pos.T)] = frozen

        return labels

    @classmethod
    def _check_valid_labels(cls, features, labels):
        if np.isscalar(labels):
            labels = np.array([labels])
        if features.shape[:-1] != labels.shape:
            raise ValueError('features and labels must have same size. features.shape != labels.shape: {} != {}'.format(
                features.shape, labels.shape))
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError('ids must be an integer array but got: {}'.format(labels.dtype))
        if np.any(labels < 0):
            raise ValueError('labels must be >= 0 but got label {}'.format(np.min(labels)))
        return labels

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
                 labels is an int numpy array with shape (h, w), where -1 indicates that the feature is a hole.
                 Any other number is the id of the feature.
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
            frozen = np.zeros((height * width), dtype=bool)
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
        self.frozen = np.zeros(self.size, dtype=bool)

    def _num_lazy_features(self) -> int:
        return sum(f.shape[0] for f, _ in self.lazy_features)


class Arrangement:
    def __init__(self, grid: Grid, sorting: np.ndarray, wrap: bool):
        """
        Creates a new arrangement.
        :param grid: The grid that was sorted.
        :param sorting: The sorting of the grid with shape (h, w). sorting[y, x] contains the index of the feature that
                        should be at position (y, x), while -1 indicates a hole.
        """
        if sorting.shape != grid.ids.shape[:2]:
            raise ValueError('sorting must have shape {} but got: {}'.format(grid.ids, sorting.shape))

        self.grid = grid
        self.sorting = sorting
        self.wrap = wrap
        self.sorted_features: np.ndarray | None = None  # Use for caching

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
        if self.sorted_features is None:
            dim = self.grid.features.shape[-1]
            if np.isscalar(hole_value):
                hole_value = np.full(dim, hole_value)
            hole_value = hole_value.reshape(1, dim)

            # add hole feature to end of features, so that sorting of -1 accesses this feature
            features = np.concatenate([self.grid.features, hole_value])
            self.sorted_features = features[self.sorting.flatten()].reshape(*self.sorting.shape, dim)

        return self.sorted_features

    def get_sorted_labels(self) -> np.ndarray:
        """
        Returns a numpy array with shape (h, w) containing the labels of the sorted features.
        get_sorted_labels()[y, x] contains the label to the feature, that was moved to (y, x).
        If get_sorted_labels()[y, x] == -1, there is a hole.
        """
        labels = np.concatenate([self.grid.labels, [-1]])
        return labels[self.sorting.flatten()].reshape(self.sorting.shape)

    def sort_by_labels(
            self, label_to_obj: Sequence | Mapping[int, Any] | Callable[[int], Any], hole_value: Any = None
    ) -> List[List[Any]]:
        """
        Given a translation label -> obj, this function creates a two-dimensional list, where result[y][x] contains the
        object given by label_to_obj[get_sorted_labels()[y, x]].

        :param label_to_obj: The translation label -> obj. If the index-operator[] is implemented, it is used. If not
                             label_to_obj is called like this: label_to_obj(label)
        :param hole_value: Value to use for holes. Defaults to None.
        """
        class IndexWrapper:
            def __init__(self, lto):
                self.label_to_obj = lto

            def __getitem__(self, item):
                return self.label_to_obj(item)

        if hasattr(label_to_obj, '__getitem__'):
            pass
        elif callable(label_to_obj):
            label_to_obj = IndexWrapper(label_to_obj)
        else:
            raise TypeError('label_to_obj must be a mapping or a callable but got: {}'.format(type(label_to_obj)))

        labels = self.get_sorted_labels()
        result = []
        for line in labels:
            line_result = []
            for label in line:
                if label == -1:
                    line_result.append(hole_value)
                else:
                    line_result.append(label_to_obj[label])
            result.append(line_result)
        return result

    def get_holes(self) -> np.ndarray:
        """
        :return: a two-dimensional numpy bool array with shape (h, w) containing True for holes and False for valid
                 positions.
        """
        return np.equal(self.sorting, -1)

    def get_valid(self) -> np.ndarray:
        """
        :return: a two-dimensional numpy bool array with shape (h, w) containing True for valid positions and False for
                 holes.
        """
        return np.not_equal(self.sorting, -1)

    def get_distance_preservation_quality(self) -> float:
        sorted_features = self.get_sorted_features()
        return metrics.distance_preservation_quality(sorted_features, self.get_valid(), self.wrap)

    def get_mean_neighbor_distance(self) -> float:
        sorted_features = self.get_sorted_features()
        valid = self.get_valid()
        return metrics.mean_neighbor_distance(sorted_features, valid, self.wrap)

    def get_distance_ratio_to_optimum(self) -> float:
        sorted_features = self.get_sorted_features()
        valid = self.get_valid()
        return metrics.distance_ratio_to_optimum(sorted_features, valid, self.wrap)


def flas(
        grid: Grid | np.ndarray, wrap: bool = False, radius_decay: float = 0.93,
        callback: Callable[[float], bool] | None = None, optimize_narrow_grids: int = 1, max_swap_positions: int = 9,
        weight_swappable: float = 1.0, weight_non_swappable: float = 100.0, weight_hole: float = 0.01, seed: int = -1
) -> Arrangement:
    """
    Sorts the given features into a 2d plane, so that similar features are close together.
    See https://github.com/Visual-Computing/LAS_FLAS for details.

    :param grid: A numpy array with shape (height, width, dims) of type float32 (otherwise it will be cast).
    :param wrap: If True, the features on the right side will be similar to features on the left side as well as
                 features on top of the plane will be similar to features on the bottom.
    :param radius_decay: How much should the filter radius decay at each iteration.
    :param callback: A callback that accepts a float and returns a boolean. The float is the current progress of the
                     algorithm between 0.0 and 1.0. If True is returned, the algorithm stops.
    :param optimize_narrow_grids: Four narrow grids it can be useful to force rows to be more similar. You can choose
                                  one of the following options:
                                  0: no optimization
                                  1: optimization for aspect_ratios < 0.1 (ten times more rows than columns)
                                  2: always optimize
    :param max_swap_positions: Number of possible swaps to hand over to solver. Should be a square number.
    :param weight_swappable:
    :param weight_non_swappable:
    :param weight_hole:
    :param seed: The random seed to initialize pseudo random numbers. If -1, the seed is not initialized.
    :return: a 2d numpy array with shape (height, width). The cell at (y, x) contains the index of the feature that
             should be at (y, x). Indices are in scanline order.
    """
    if isinstance(grid, Grid):
        pass
    elif isinstance(grid, np.ndarray):
        if grid.ndim == 2:
            grid = Grid.from_features(grid)
        elif grid.ndim == 3:
            grid = Grid.from_grid_features(grid)
        else:
            raise ValueError('numpy grid must have shape (h, w, d) or (n, d) but got: {}'.format(grid.shape))
    else:
        raise TypeError('features must be a Grid or a numpy array but got: {}'.format(type(grid)))

    size = grid.get_size()
    if size[0] < 2 or size[1] < 2:
        raise ValueError('Grid must have at least size (2, 2), but got: {}'.format(size))

    if radius_decay >= 1.0:
        raise ValueError('radius_decay must be smaller than 1.0 but got: {}'.format(radius_decay))

    # TODO: return identity sorting
    if np.all(grid.frozen):
        raise ValueError('All features are frozen. Cannot sort features.')

    if callback is None:
        code, result = flas_cpp.flas_no_callback(
            grid.features, grid.ids, grid.frozen, wrap, radius_decay, weight_swappable, weight_non_swappable,
            weight_hole, max_swap_positions, seed, optimize_narrow_grids
        )
    else:
        code, result = flas_cpp.flas(
            grid.features, grid.ids, grid.frozen, wrap, radius_decay, weight_swappable, weight_non_swappable,
            weight_hole, max_swap_positions, seed, optimize_narrow_grids, callback
        )

    if code != 0:
        raise RuntimeError('FLAS failed with error code {}'.format(code))

    return Arrangement(grid, result, wrap)


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


__all__ = ['GridBuilder', 'Grid', 'Arrangement', 'flas', 'metrics',  'Labeler']
