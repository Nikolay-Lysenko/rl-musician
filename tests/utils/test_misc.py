"""
Test `rlmusician.utils.misc` module.

Author: Nikolay Lysenko
"""


import numpy as np
import pytest

from rlmusician.utils import (
    apply_rolling_aggregation, shift_vertically
)


@pytest.mark.parametrize(
    "arr, max_lag, fn_name, lags_only, expected",
    [
        (
            # `arr`
            np.array([
                [1, 0, 1, 1],
                [0, 1, 1, 0],
                [0, 1, 0, 1],
            ]),
            # `max_lag`
            2,
            # `fn_name`
            'mean',
            # `lags_only`
            True,
            # `expected`
            np.array([
                [0.0, 0.5, 0.5, 0.5],
                [0.0, 0.0, 0.5, 1.0],
                [0.0, 0.0, 0.5, 0.5],
            ])
        ),
        (
            # `arr`
            np.array([
                [1, 0, 1, 1],
                [0, 1, 1, 0],
                [0, 1, 0, 1],
            ]),
            # `max_lag`
            2,
            # `fn_name`
            'mean',
            # `lags_only`
            False,
            # `expected`
            np.array([
                [1/3, 1/3, 2/3, 2/3],
                [0.0, 1/3, 2/3, 2/3],
                [0.0, 1/3, 1/3, 2/3],
            ])
        ),
    ]
)
def test_apply_rolling_aggregation(
        arr: np.ndarray, max_lag: int, fn_name: str, lags_only: bool,
        expected: np.ndarray
) -> None:
    """Test `apply_rolling_aggregation` function."""
    result = apply_rolling_aggregation(arr, max_lag, fn_name, lags_only)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "arr, shift, expected",
    [
        (
            # `arr`
            np.array([
                [1, 0, 1],
                [0, 1, 0],
                [0, 0, 1],
            ]),
            # `shift`
            0,
            # `expected`
            np.array([
                [1, 0, 1],
                [0, 1, 0],
                [0, 0, 1],
            ])
        ),
        (
            # `arr`
            np.array([
                [1, 0, 1],
                [0, 1, 0],
                [0, 0, 1],
            ]),
            # `shift`
            1,
            # `expected`
            np.array([
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 0],
            ])
        ),
        (
            # `arr`
            np.array([
                [1, 0, 1],
                [0, 1, 0],
                [0, 0, 1],
            ]),
            # `shift`
            -2,
            # `expected`
            np.array([
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 1],
            ])
        ),
    ]
)
def test_shift_vertically(
        arr: np.ndarray, shift: int, expected: np.ndarray
) -> None:
    """Test `shift_vertically` function."""
    result = shift_vertically(arr, shift)
    np.testing.assert_equal(result, expected)
