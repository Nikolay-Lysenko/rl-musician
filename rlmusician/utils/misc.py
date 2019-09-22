"""
Do miscellaneous auxiliary tasks.

Author: Nikolay Lysenko
"""


import multiprocessing as mp
from typing import Any, Callable, List, Optional, Tuple

import numpy as np


def map_in_parallel(
        fn: Callable,
        args: List[Tuple[Any, ...]],
        n_processes: Optional[int] = None
) -> List[Any]:
    """
    Apply function to each tuple of arguments from given list in parallel.

    This function contains boilerplate code that is needed for correct work
    of `pytest-cov`. Usage of `mp.Pool` as context manager is not alternative
    to this function, because:
    1) not all covered lines of code may be marked as covered;
    2) some files with names like'.coverage.hostname.*' may be not deleted.

    See more: https://github.com/pytest-dev/pytest-cov/issues/250

    :param fn:
        function
    :param args:
        list of tuples of arguments
    :param n_processes:
        number of child processes
    :return:
        results of applying the function to the arguments
    """
    pool = mp.Pool(n_processes)
    try:
        results = pool.starmap(fn, args)
    finally:
        pool.close()
        pool.join()
    return results


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
    result = np.zeros_like(arr)
    if shift > 0:
        result[:, shift:] = arr[:, :-shift]
    else:
        result[:, :shift] = arr[:, -shift:]
    return result


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
    if shift == 0:
        return arr
    result = np.zeros_like(arr)
    if shift > 0:
        result[:shift, :] = arr[-shift:, :]
    else:
        result[shift:, :] = arr[:-shift, :]
    return result


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
