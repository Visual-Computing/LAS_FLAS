from typing import Optional, List, Tuple
import warnings

import numpy as np
import flas_c_py


def flas(features: np.ndarray, frozen: Optional[np.ndarray] = None, aspect_ratio: float = 1.0, wrap: bool = False,
         freeze_holes: bool = True, radius_decay: float = 0.93, max_swap_positions: int = 9,
         weight_swappable: float = 1.0, weight_non_swappable: float = 100.0, weight_hole: float = 0.01) -> np.ndarray:
    """
    Sorts the given features into a 2d plane, so that similar features are close together.
    See https://github.com/Visual-Computing/LAS_FLAS for details.

    :param features: A numpy array with shape (height, width, dims) of type float32 (otherwise it will be cast).
    :param frozen: A numpy array with shape (height, width) of type bool. If None, all features are assumed to be not
                   frozen. Frozen features will not be moved.
    :param aspect_ratio: The desired aspect ratio of the plane. Only used, if features are given in 1d.
    :param wrap: If True, the features on the right side will be similar to features on the left side as well as
                 features on top of the plane will be similar to features on the bottom.
    :param freeze_holes: If True, holes in the plane will be frozen.
    :param radius_decay: How much should the filter radius decay at each iteration.
    :param max_swap_positions: Number of possible swaps to hand over to solver. Should be a square number.
    :param weight_swappable:
    :param weight_non_swappable:
    :param weight_hole:
    :return: a 2d numpy array with shape (height, width). The cell at (y, x) contains the index of the feature that
             should be at (y, x). Indices are in scanline order.
    """
    code = 1
    result = None

    if isinstance(features, Grid):  # TODO: grid case
        raise NotImplementedError()
    elif len(features.shape) == 2:  # 1d features case
        if features.dtype != np.float32:
            features = features.astype(np.float32)
        if frozen is not None:
            warnings.warn(
                'frozen will be ignored, because features are not given in grid. features.shape={}'
                .format(features.shape)
            )
        code, result = flas_c_py.flas_1d_features(
            features, aspect_ratio, freeze_holes, wrap, radius_decay, weight_swappable, weight_non_swappable,
            weight_hole, max_swap_positions
        )
    elif len(features.shape) == 3:  # 2d features case
        if features.dtype != np.float32:
            features = features.astype(np.float32)
        if frozen is None:
            frozen = np.zeros(features.shape[:2], dtype=np.bool)
        else:
            if frozen.shape != features.shape[:2]:
                raise ValueError(
                    'frozen must have same size as features {} but got {}'.format(features.shape[:2], frozen.shape)
                )
            if frozen.dtype != np.bool:
                raise TypeError('frozen must be of type np.bool but got "{}"'.format(frozen.dtype))
            if np.all(frozen):
                # if all features are frozen, we return identity sorting.
                return np.arange(np.prod(features.shape[:2])).reshape(features.shape[:2])
        code, result = flas_c_py.flas_2d_features(
            features, frozen, wrap, radius_decay, weight_swappable, weight_non_swappable, weight_hole,
            max_swap_positions
        )
    else:
        features_info = type(features).__name__
        if isinstance(features, np.ndarray):
            features_info = 'numpy array with shape {}'.format(features.shape)
        raise ValueError(
            'features must be Grid or a numpy array with shape (height, width, dims) or (n, dims) but got {}'
            .format(features_info)
        )

    if code != 0:
        raise RuntimeError('FLAS failed with error code {}'.format(code))
    return result


class Grid:
    def __init__(self):
        self.size: Optional[Tuple[int, int]] = None
        self.dim: Optional[int] = None
        self.grid: Optional[np.ndarray] = None
        self.grid_taken: Optional[np.ndarray] = None
        self.grid_frozen: Optional[np.ndarray] = None
        self.lazy_features: List[np.ndarray] = []

    def __repr__(self):
        return 'Grid(size={}, dim={})'.format(self.size, self.dim)

    @classmethod
    def from_data(cls, features: np.ndarray):
        """
        Creates a grid with the given features.
        :param features: A numpy array with shape (h, w, d) or (n, d).
        """
        if features.ndim == 2:
            grid = Grid()
            grid.dim = features.shape[-1]
            grid.lazy_features.append(features)
        elif features.ndim == 3:
            grid = Grid()
            grid.dim = features.shape[-1]
            grid._resize(features.shape[:2])
            grid.grid = features
            grid.grid_taken = np.ones(features.shape[:2], dtype=np.bool)
            grid.frozen = np.zeros(features.shape[:2], dtype=np.bool)
        else:
            raise ValueError('features must have shape (h, w, d) or (n, d) but got: {}'.format(features.shape))
        return grid

    def add(self, features: np.ndarray):
        """
        Add the given features anywhere to the grid.
        :param features: numpy array with shape (d,) or (n, d), where d is the feature dimensionality and n is the
                         number of features.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.ndim != 2:
            raise ValueError('Features must have shape (n, d) but got: {}'.format(features.shape))

        self._check_newdim(features.shape[-1])
        self.lazy_features.append(features)

    def put(self, features: np.ndarray, pos: Tuple[int, int] | np.ndarray, frozen: bool | np.ndarray = True):
        """
        Put the given features to the given position.

        :param features: numpy array with shape (n, d) or (d,), where d is the feature dimensionality and n is the
                         number of features.
        :param pos: tuple (y, x) or numpy array with shape (n, 2) or (2,), where n is the number of positions to put
                    the features to. Should have the same n as features.
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

        new_size = np.max(pos, axis=0) + 1
        if self.size is not None:
            new_size = np.maximum(new_size, self.size)
        self._resize(new_size)

        # check fields already taken
        if np.any(self.grid_taken[tuple(pos.T)]):
            raise ValueError('Position is already taken')
        self.grid_taken[tuple(pos.T)] = True
        self.grid[tuple(pos.T)] = features
        self.frozen[tuple(pos.T)] = frozen

    def get(self, pos: Tuple[int, int] | np.ndarray) -> np.ndarray:
        """
        Get the features at the given position.
        :param pos: The position to get the feature from. Can be a tuple (y, x) or a numpy array with shape (n, 2) where
                    n is the number of positions to get.
        :return:
        """
        if isinstance(pos, np.ndarray):
            pos = tuple(pos.T)

        return self.grid[pos]

    def compile(self, aspect_ratio: float):
        """
        Returns a numpy array with shape (h, w, d), if there are features with a defined position.
        Returns a numpy array with shape (n, d), if there are no features with a defined position.

        :param aspect_ratio: The aspect ratio
        :return: A tuple with three numpy arrays: (features, taken, frozen).
                 features is a numpy array with shape (h, w, d) or (n, d), where d is the feature dimensionality, h and
                 w are height and width of the feature plane and n is the number of features.
                 taken is a boolean numpy array with shape (h, w) or (n,), where True indicates that the feature is
                 valid (not a hole).
                 frozen is a boolean numpy array with shape (h, w) or (n,), where True indicates that the feature is
                 frozen (should not be swapped).
        """
        if self.grid is None or np.sum(self.grid_taken) == 0:  # only dynamic features
            features = np.concatenate(self.lazy_features)
            n_features = features.shape[0]
            taken = np.zeros(n_features, dtype=np.bool)
            taken[:n_features] = True
            return features, taken, np.zeros(n_features, dtype=np.bool)
        else:
            num_static_features = np.sum(self.grid_taken)
            num_lazy_features = self._num_lazy_features()
            total_num_features = num_static_features + num_lazy_features

            # get width / height to fit all lazy features
            height, width = get_optimal_grid_size(total_num_features, aspect_ratio, *self.grid_taken.shape)

            # scale grids to new size
            new_grid = _embed_array(self.grid, (height, width))
            new_grid_taken = _embed_array(self.grid_taken, (height, width))
            new_frozen = _embed_array(self.frozen, (height, width))

            # apply lazy features
            free_indices = np.where(np.logical_not(new_grid_taken))
            free_indices = tuple(fi[:num_lazy_features] for fi in free_indices)
            new_grid[free_indices] = np.concatenate(self.lazy_features)
            new_grid_taken[free_indices] = True

            return new_grid, new_grid_taken, new_frozen

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
        if self.grid is not None:
            self.grid = _embed_array(self.grid, self.size)
            self.grid_taken = _embed_array(self.grid_taken, self.size)
            self.frozen = _embed_array(self.frozen, self.size)
        elif self.dim is not None:
            self._init_grids()

    def _init_grids(self):
        self.grid = np.zeros((*self.size, self.dim), dtype=np.float32)
        self.grid_taken = np.zeros(self.size, dtype=np.bool)
        self.frozen = np.zeros(self.size, dtype=np.bool)

    def _num_lazy_features(self) -> int:
        return sum(f.shape[0] for f in self.lazy_features)


def _embed_array(array: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
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

    # Create a new array filled with zeros
    new_array = np.zeros((new_h, new_w, *rest_shape), dtype=array.dtype)

    # Embed the original array into the new array
    new_array[:h, :w] = array

    return new_array


def get_optimal_grid_size(total_num_features, aspect_ratio, min_height, min_width):
    while total_num_features > min_height * min_width:
        if min_width / min_height < aspect_ratio:
            min_width += 1
        else:
            min_height += 1
    return min_height, min_width


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
