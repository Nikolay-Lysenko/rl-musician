"""
Do miscellaneous auxiliary tasks.

Author: Nikolay Lysenko
"""


import multiprocessing as mp
from copy import deepcopy
from typing import Any, Callable, Dict, Iterator, List, Optional


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


def imap_in_parallel(
        fn: Callable,
        args: Iterator[Any],
        pool_kwargs: Optional[Dict[str, Any]] = None
) -> Iterator[Any]:
    """
    Apply function to each argument from given iterable in parallel.

    This function contains boilerplate code that is needed for correct work
    of `pytest-cov`. Usage of `mp.Pool` as context manager is not alternative
    to this function, because:
    1) not all covered lines of code may be marked as covered;
    2) some files with names like '.coverage.hostname.*' may be not deleted.

    See more: https://github.com/pytest-dev/pytest-cov/issues/250.

    :param fn:
        function
    :param args:
        generator of arguments
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
        results = pool.imap(fn, args)
    finally:
        pool.close()
        pool.join()
    return results


def generate_deep_copies(something: Any, n_copies: int) -> Iterator[Any]:
    """
    Generate deep copies of an object.

    :param something:
        object to be copied
    :param n_copies:
        number of copies to be generated
    :return:
        deep copies
    """
    for _ in range(n_copies):
        yield deepcopy(something)


def rolling_aggregate(
        values: List[float],
        aggregation_fn: Callable[[List[float]], float],
        window_size: int
) -> List[float]:
    """
    Compute rolling aggregate.

    :param values:
        list of values to be aggregated
    :param aggregation_fn:
        aggregation function
    :param window_size:
        size of rolling window
    :return:
        list of rolling aggregates
    """
    window = []
    results = []
    for value in values:
        if len(window) == window_size:
            window.pop(0)
        window.append(value)
        results.append(aggregation_fn(window))
    return results
