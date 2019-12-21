"""
Do miscellaneous auxiliary tasks.

Author: Nikolay Lysenko
"""


import multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional, Tuple


def map_in_parallel(
        fn: Callable,
        args: List[Tuple[Any, ...]],
        pool_kwargs: Optional[Dict[str, Any]] = None
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
    :param pool_kwargs:
        parameters of pool such as number of processes and maximum number of
        tasks for a worker before it is replaced with a new one
    :return:
        results of applying the function to the arguments
    """
    pool_kwargs = pool_kwargs or {}
    pool_kwargs['processes'] = pool_kwargs.get('n_processes')
    pool_kwargs['maxtasksperchild'] = pool_kwargs.get('max_tasks_per_child')
    old_keys = ['n_processes', 'max_tasks_per_child']
    pool_kwargs = {k: v for k, v in pool_kwargs.items() if k not in old_keys}
    pool = mp.Pool(**pool_kwargs)
    try:
        results = pool.starmap(fn, args)
    finally:
        pool.close()
        pool.join()
    return results


def convert_to_base(
        number: int, base: int, min_length: Optional[int] = None
) -> List[int]:
    """
    Convert number to its representation in a given system.

    :param number:
        positive integer number to be converted
    :param base:
        positive integer number to be used as base
    :param min_length:
        if result length is less than it, zero padding is added to the left
    :return:
        list where each element represents a digit in a given system
    """
    digits = []
    if number == 0:
        digits = [0]
    while number > 0:
        remainder = number % base
        digits.append(remainder)
        number //= base
    if min_length is not None:
        padding = [0 for _ in range(max(min_length - len(digits), 0))]
        digits.extend(padding)
    digits = digits[::-1]
    return digits
