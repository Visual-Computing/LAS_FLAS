from typing import Optional

import numpy as np
import flas_c_py


def flas(features: np.ndarray, frozen: Optional[np.ndarray] = None, wrap: bool = False,
         radius_decay: float = 0.93, max_swap_positions: int = 9,
         weight_swappable: float = 1.0, weight_non_swappable: float = 100.0, weight_hole: float = 0.01) -> np.ndarray:
    """
    Sorts the given features into a 2d plane, so that similar features are close together.
    See https://github.com/Visual-Computing/LAS_FLAS for details.

    :param features: A numpy array with shape (height, width, dims) of type float32 (otherwise it will be cast).
    :param frozen: A numpy array with shape (height, width) of type bool. If None, all features are assumed to be not
                   frozen. Frozen features will not be moved.
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
    if len(features.shape) != 3:
        raise ValueError('features must have shape (height, width, dims) but got {}'.format(features.shape))

    if features.dtype != np.float32:
        features = features.astype(np.float32)

    if frozen is None:
        frozen = np.zeros(features.shape[:2], dtype=np.bool)
    else:
        if frozen.shape != features.shape[:2]:
            raise ValueError(
                'in_use must have same size as features {} but got {}'.format(features.shape[:2], frozen.shape)
            )
        if frozen.dtype != np.bool:
            raise TypeError('in_use must be of type np.bool but got "{}"'.format(frozen.dtype))
        if np.all(frozen):
            # if all features are frozen, we return identity sorting.
            return np.arange(np.prod(features.shape[:2])).reshape(features.shape[:2])

    success, result = flas_c_py.flas(
        features, frozen, wrap, radius_decay, weight_swappable, weight_non_swappable, weight_hole, max_swap_positions
    )
    if success != 0:
        raise RuntimeError('FLAS failed with error code {}'.format(success))
    return result


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
    height, width = features.shape[:2]
    return features.reshape(height * width, -1)[sorting.flatten()].reshape(features.shape)
