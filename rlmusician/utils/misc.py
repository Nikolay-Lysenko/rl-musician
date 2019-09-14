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


def apply_rolling_aggregation(
        arr: np.ndarray, max_lag: int, fn_name: str, lags_only: bool = True
) -> np.ndarray:
    """
    Compute rolling statistics of array.

    :param arr:
        2D array with horizontal axis representing time
    :param max_lag:
        maximum lag to include
    :param fn_name:
        name of aggregation function, one of 'min', 'mean', and 'max'
    :param lags_only:
        if it is `True`, the current value is excluded from computation;
        if it is `False`, rolling window includes the current value
    :return:
        array of rolling statistics
    """
    range_start = 1 if lags_only else 0
    lagged_rolls = [
        shift_horizontally(arr, i).reshape(arr.shape + (1,))
        for i in range(range_start, max_lag + 1)
    ]
    lagged_roll = np.concatenate(lagged_rolls, axis=-1)
    name_to_fn = {
        'min': lambda x: x.min(axis=-1),
        'mean': lambda x: x.mean(axis=-1),
        'max': lambda x: x.max(axis=-1)
    }
    rolling_stats_roll = name_to_fn[fn_name](lagged_roll)
    return rolling_stats_roll
