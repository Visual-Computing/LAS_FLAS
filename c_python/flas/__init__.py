from typing import Optional, List, Tuple, Any

import numpy as np
import flas_cpp


class Grid:
    def __init__(self, aspect_ratio: float = 1.0, check_overwrite: bool = False):
        self.size: Optional[Tuple[int, int]] = None
        self.dim: Optional[int] = None
        self.grid_features: Optional[np.ndarray] = None
        self.grid_ids: Optional[np.ndarray] = None
        self.grid_frozen: Optional[np.ndarray] = None
        self.lazy_features: List[Tuple[np.ndarray, np.ndarray]] = []
        self.aspect_ratio = aspect_ratio
        self.check_overwrite = check_overwrite

    def __repr__(self):
        return 'Grid(size={}, dim={})'.format(self.size, self.dim)

    @classmethod
    def from_data(cls, features: np.ndarray, ids: np.ndarray):
        """
        Creates a grid with the given features.
        :param features: A numpy array with shape (h, w, d) or (n, d).
        :param ids: A numpy array with shape (h, w) or (n,). Each id at ids[y, x] corresponds to feature[y, x] for later
                    identification.
        """
        if features.shape[:-1] != ids.shape:
            raise ValueError('features and ids must have same size. features.shape != ids.shape: {} != {}'.format(
                features.shape, ids.shape))
        if not np.isdtype(ids.dtype, 'integral'):
            raise ValueError('ids must be an integer array but got: {}'.format(ids.dtype))

        if features.ndim == 2:
            grid = Grid()
            grid.dim = features.shape[-1]
            grid.lazy_features.append((features, ids))
        elif features.ndim == 3:
            grid = Grid()
            grid.dim = features.shape[-1]
            grid._resize(features.shape[:2])
            grid.grid_features = features
            grid.grid_ids = ids
            grid.frozen = np.zeros(features.shape[:2], dtype=np.bool)
        else:
            raise ValueError('features must have shape (h, w, d) or (n, d) but got: {}'.format(features.shape))
        return grid

    def add(self, features: np.ndarray, ids: int | np.ndarray):
        """
        Add the given features anywhere to the grid.
        :param features: numpy array with shape (d,) or (n, d), where d is the feature dimensionality and n is the
                         number of features.
        :param ids: integer or numpy array with shape (n,), where n is the number of features. Each id at ids[i]
                    identifies on feature.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.ndim != 2:
            raise ValueError('features must have shape (n, d) but got: {}'.format(features.shape))

        if isinstance(ids, int):
            ids = np.array([ids])
        if features.shape[:-1] != ids.shape:
            raise ValueError('features and ids must have same size. features.shape != ids.shape: {} != {}'.format(
                features.shape, ids.shape))
        if not np.isdtype(ids.dtype, 'integral'):
            raise ValueError('ids must be an integer array but got: {}'.format(ids.dtype))

        self._check_newdim(features.shape[-1])
        self.lazy_features.append((features, ids))

    def put(
            self, features: np.ndarray, pos: Tuple[int, int] | np.ndarray,
            ids: int | np.ndarray, frozen: bool | np.ndarray = True
    ):
        """
        Put the given features to the given position.

        :param features: numpy array with shape (n, d) or (d,), where d is the feature dimensionality and n is the
                         number of features.
        :param pos: tuple (y, x) or numpy array with shape (n, 2) or (2,), where n is the number of positions to put
                    the features to. Should have the same n as features.
        :param ids: integer or numpy array with shape (n,), where n is the number of features. Each id at ids[i]
                    identifies feature[i].
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
        if isinstance(ids, int):
            ids = np.array([ids])
        if features.shape[:-1] != ids.shape:
            raise ValueError('features and ids must have same size. features.shape != ids.shape: {} != {}'.format(
                features.shape, ids.shape))
        if not np.isdtype(ids.dtype, 'integral'):
            raise ValueError('ids must be an integer array but got: {}'.format(ids.dtype))

        # resize grid
        new_size = np.max(pos, axis=0) + 1
        if self.size is not None:
            new_size = np.maximum(new_size, self.size)
        self._resize(new_size)

        # check overwrite
        if self.check_overwrite:
            if np.any(self.grid_ids[tuple(pos.T)] >= 0):
                raise ValueError('Position is already taken')

        # save features
        self.grid_ids[tuple(pos.T)] = ids
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

    def get_ids(self, pos: Tuple[int, int] | np.ndarray) -> np.ndarray:
        """
        Get the ids at the given position.
        :param pos: The position to get the feature from. Can be a tuple (y, x) or a numpy array with shape (n, 2) where
                    n is the number of positions to get.
        :return: The features at the given positions
        """
        if isinstance(pos, np.ndarray):
            pos = tuple(pos.T)

        return self.grid_ids[pos]

    def compile(self):
        """
        Compiles this grid into numpy arrays, which can be used for the flas algorithm.

        :return: A tuple with three numpy arrays: (features, ids, frozen).
                 features is a numpy array with shape (h, w, d), where d is the feature dimensionality and h and w are
                 height and width of the feature plane.
                 ids is an int numpy array with shape (h, w), where -1 indicates that the feature is a hole. Any other
                 number is the id of the feature.
                 frozen is a boolean numpy array with shape (h, w), where True indicates that the feature is frozen
                 (should not be moved).
        """
        if self.grid_ids is None or np.sum(self.grid_ids != -1) == 0:  # only dynamic features
            n_features = self._num_lazy_features()
            height, width = flas_cpp.get_optimal_grid_size(n_features, self.aspect_ratio, 2, 2)

            # features
            features = [f for f, _ in self.lazy_features]
            n_missing_features = height * width - n_features
            if n_missing_features > 0:
                padding_features = np.zeros((n_missing_features, self.dim), dtype=np.float32)
                features = features + [padding_features]
            features = np.concatenate(features)
            features = features.reshape(height, width, self.dim)

            # ids
            ids = [ids for _, ids in self.lazy_features]
            if n_missing_features > 0:
                padding_ids = np.full((n_missing_features,), -1, dtype=np.int32)
                ids = ids + [padding_ids]
            ids = np.concatenate(ids)
            ids = ids.reshape(height, width)
            ids = ids.astype(np.int32)

            # frozen
            frozen = np.zeros((height, width), dtype=np.bool)

            return features, ids, frozen
        else:
            num_static_features = np.sum(self.grid_ids != -1)
            num_lazy_features = self._num_lazy_features()
            total_num_features = num_static_features + num_lazy_features

            # get width / height to fit all lazy features
            height, width = flas_cpp.get_optimal_grid_size(
                total_num_features, self.aspect_ratio, *self.grid_ids.shape
            )

            # scale grids to new size
            new_grid = _embed_array(self.grid_features, (height, width))
            new_grid_ids = _embed_array(self.grid_ids, (height, width), fill_value=-1)
            new_frozen = _embed_array(self.frozen, (height, width))

            # apply lazy features
            if self.lazy_features:
                features = [f for f, _ in self.lazy_features]
                ids = [ids for _, ids in self.lazy_features]
                free_indices = np.where(new_grid_ids == -1)
                free_indices = tuple(fi[:num_lazy_features] for fi in free_indices)
                new_grid[free_indices] = np.concatenate(features)
                new_grid_ids[free_indices] = np.concatenate(ids)

            new_grid_ids = new_grid_ids.astype(np.int32)

            return new_grid, new_grid_ids, new_frozen

    def _check_newdim(self, new_dim):
        if self.dim is None:
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
            self.grid_ids = _embed_array(self.grid_ids, self.size, fill_value=-1)
            self.frozen = _embed_array(self.frozen, self.size)
        elif self.dim is not None:
            self._init_grids()

    def _init_grids(self):
        self.grid_features = np.zeros((*self.size, self.dim), dtype=np.float32)
        self.grid_ids = np.full(self.size, -1, dtype=np.bool)
        self.frozen = np.zeros(self.size, dtype=np.bool)

    def _num_lazy_features(self) -> int:
        return sum(f.shape[0] for f, _ in self.lazy_features)


def flas(features: Grid | np.ndarray, wrap: bool = False, radius_decay: float = 0.93, max_swap_positions: int = 9,
         weight_swappable: float = 1.0, weight_non_swappable: float = 100.0, weight_hole: float = 0.01) -> np.ndarray:
    """
    Sorts the given features into a 2d plane, so that similar features are close together.
    See https://github.com/Visual-Computing/LAS_FLAS for details.

    :param features: A numpy array with shape (height, width, dims) of type float32 (otherwise it will be cast).
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
    if isinstance(features, Grid):
        features, ids, frozen = features.compile()
    elif isinstance(features, np.ndarray):
        if features.dtype != np.float32:
            features = features.astype(np.float32)
        if features.ndim == 3:
            ids = np.ones(features.shape[:2], dtype=np.bool)
            frozen = np.zeros(features.shape[:2], dtype=np.bool)
        elif features.ndim == 2:  # 1d features case
            grid = Grid.from_data(features, np.arange(features.shape[0]))
            features, ids, frozen = grid.compile()
        else:
            raise ValueError('features must have shape (h, w, d) or (n, d) but got: {}'.format(features.shape))
    else:
        raise TypeError('features must be a Grid or a numpy array but got: {}'.format(type(features)))

    # TODO: this is wrong, when using ids
    if np.all(frozen):
        # if all features are frozen, we return identity sorting.
        return np.arange(np.prod(features.shape[:2])).reshape(features.shape[:2])

    code, result = flas_cpp.flas_2d_features(
        features, ids, frozen, wrap, radius_decay, weight_swappable, weight_non_swappable, weight_hole,
        max_swap_positions
    )

    if code != 0:
        raise RuntimeError('FLAS failed with error code {}'.format(code))

    return result


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
