"""
Test `rlmusician.utils.misc` module.

Author: Nikolay Lysenko
"""


from typing import Callable, List, Optional

import pytest

from rlmusician.utils.misc import convert_to_base, rolling_aggregate


@pytest.mark.parametrize(
    "number, base, min_length, expected",
    [
        (12, 2, None, [1, 1, 0, 0]),
        (0, 5, None, [0]),
        (0, 5, 2, [0, 0]),
        (12, 8, 6, [0, 0, 0, 0, 1, 4]),
    ]
)
def test_convert_to_base(
        number: int, base: int, min_length: Optional, expected: List[int]
) -> None:
    """Test `convert_to_base` function."""
    result = convert_to_base(number, base, min_length)
    assert result == expected


@pytest.mark.parametrize(
    "values, aggregation_fn, window_size, expected",
    [
        ([0, 5, 2, 1, -3, 6, 4, 7], min, 3, [0, 0, 0, 1, -3, -3, -3, 4]),
    ]
)
def test_rolling_aggregate(
        values: List[float], aggregation_fn: Callable[[List[float]], float],
        window_size: int, expected: List[float]
) -> None:
    """Test `rolling_aggregate` function."""
    result = rolling_aggregate(values, aggregation_fn, window_size)
    assert result == expected
