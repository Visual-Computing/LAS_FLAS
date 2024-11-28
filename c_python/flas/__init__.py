from typing import Optional
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
    pass


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
