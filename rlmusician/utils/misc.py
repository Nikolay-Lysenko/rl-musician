"""
Do miscellaneous auxiliary tasks.

Author: Nikolay Lysenko
"""


import multiprocessing as mp
from typing import Any, Callable, List, Optional, Tuple


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
    2) some files with names like '.coverage.hostname.*' may be not deleted.

    See more: https://github.com/pytest-dev/pytest-cov/issues/250.

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


def convert_to_base(number: int, base: int) -> List[int]:
    """
    Convert number to its representation in a given system.

    :param number:
        integer number
    :param base:
        positive integer number to be used as base
    :return:
        list where each element represents a digit in a given system
    """
    digits = []
    if number == 0:
        return [0]
    while number > 0:
        remainder = number % base
        digits.append(remainder)
        number //= base
    digits = digits[::-1]
    return digits
