"""
Do miscellaneous auxiliary tasks.

Author: Nikolay Lysenko
"""


import numpy as np


def shift_horizontally(arr: np.ndarray, shift: int) -> np.ndarray:
    """
    Shift 2D array along horizontal axis.

    Number of columns of output array is the same as that of input array,
    because non-fitting values are removed and gaps are padded with zeros.

    :param arr:
        2D array
    :param shift:
        signed value of shift, positive for shift to the right
        and negative for shift to the left
    :return:
        shifted array
    """
    if shift == 0:
        return arr
    elif shift > 0:
        return np.hstack((np.zeros((arr.shape[0], shift)), arr[:, :-shift]))
    else:
        return np.hstack((arr[:, -shift:], np.zeros((arr.shape[0], -shift))))


def shift_vertically(arr: np.ndarray, shift: int) -> np.ndarray:
    """
    Shift 2D array along vertical axis.

    Length of output array is the same as length of input array,
    because non-fitting values are removed and gaps are padded with zeros.

    :param arr:
        2D array
    :param shift:
        signed value of shift, positive for upward shift
        and negative for downward shift
    :return:
        shifted array
    """
    return shift_horizontally(arr.T, shift).T
