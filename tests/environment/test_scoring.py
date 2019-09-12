"""
Test `rlmusician.environment.scoring` module.

Author: Nikolay Lysenko
"""


from typing import Dict

import numpy as np
import pytest

from rlmusician.environment.scoring import (
    score_horizontal_variance,
    score_vertical_variance,
    score_consonances
)


@pytest.mark.parametrize(
    "roll, expected",
    [
        (
            # `roll`
            np.array([
                [0, 0],
                [1, 1]
            ]),
            # `expected`
            0
        ),
        (
            # `roll`
            np.array([
                [1, 0],
                [0, 1]
            ]),
            # `expected`
            0.25
        ),
    ]
)
def test_score_horizontal_variance(roll: np.ndarray, expected: float) -> None:
    """Test `score_horizontal_variance` function."""
    result = score_horizontal_variance(roll)
    assert result == expected


@pytest.mark.parametrize(
    "roll, expected",
    [
        (
            # `roll`
            np.array([
                [1, 0],
                [1, 0]
            ]),
            # `expected`
            0
        ),
        (
            # `roll`
            np.array([
                [1, 0],
                [0, 1]
            ]),
            # `expected`
            0.25
        ),
    ]
)
def test_score_vertical_variance(roll: np.ndarray, expected: float) -> None:
    """Test `score_vertical_variance` function."""
    result = score_vertical_variance(roll)
    assert result == expected


@pytest.mark.parametrize(
    "roll, interval_consonances, distance_weights, expected",
    [
        (
            # `roll`
            np.array([
                [0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [1, 1, 1, 0, 0]
            ]),
            # `interval_consonances`
            {1: -1, 2: -0.5, 3: 0},
            # `distance_weights`
            {0: 1, 1: 1, 2: 0.5, 3: 0.25},
            # `expected`
            -3.75
        ),
        (
            # `roll`
            np.array([
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 0, 0, 0, 1]
            ]),
            # `interval_consonances`
            {1: -1, 2: -0.5, 3: 0},
            # `distance_weights`
            {0: 1, 1: 1, 2: 0.5, 3: 0.25},
            # `expected`
            -2.25
        ),
        (
            # `roll`
            np.array([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0]
            ]),
            # `interval_consonances`
            {1: -1, 2: -0.5, 3: 0},
            # `distance_weights`
            {0: 1, 1: 1, 2: 0.5, 3: 0.25},
            # `expected`
            0
        ),
    ]
)
def test_score_consonances(
        roll: np.ndarray, interval_consonances: Dict[int, float],
        distance_weights: Dict[int, float], expected: float
) -> None:
    """Test `score_consonances` function."""
    result = score_consonances(roll, interval_consonances, distance_weights)
    assert result == expected
