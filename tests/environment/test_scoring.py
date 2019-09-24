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
    score_absence_of_long_sounds,
    score_noncyclicity,
    score_consonances,
    score_conjunct_motion
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
    "roll, max_n_time_steps, expected",
    [
        (
            # `roll`,
            np.array([
                [1, 1, 1, 1],
                [0, 1, 0, 1],
                [0, 0, 0, 0],
            ]),
            # `max_n_time_steps`
            3,
            # expected`
            -1
        ),
        (
            # `roll`,
            np.array([
                [1, 1, 1, 1],
                [0, 1, 0, 1],
                [1, 1, 0, 0],
            ]),
            # `max_n_time_steps`
            2,
            # expected`
            -2
        ),
        (
            # `roll`,
            np.array([
                [1, 1, 1, 1],
                [0, 1, 0, 1],
                [1, 1, 0, 0],
            ]),
            # `max_n_time_steps`
            1,
            # expected`
            -4
        ),
    ]
)
def test_score_absence_of_long_sounds(
        roll: np.ndarray, max_n_time_steps: int, expected: int
) -> None:
    """Test `score_absence_of_long_sounds` function."""
    result = score_absence_of_long_sounds(roll, max_n_time_steps)
    assert result == expected


@pytest.mark.parametrize(
    "roll, max_n_time_steps, max_share, expected",
    [
        (
            # `roll`
            np.array([
                [1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1],
            ]),
            # `max_n_time_steps`
            1,
            # `max_share`
            0.5,
            # `expected`
            1
        ),
        (
            # `roll`
            np.array([
                [1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1],
            ]),
            # `max_n_time_steps`
            2,
            # `max_share`
            0.5,
            # `expected`
            1 / 3
        ),
        (
            # `roll`
            np.array([
                [1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1],
                [0, 1, 1, 0, 0, 1],
            ]),
            # `max_n_time_steps`
            3,
            # `max_share`
            1,
            # `expected`
            5 / 18
        ),
    ]
)
def test_score_noncyclicity(
        roll: np.ndarray, max_n_time_steps: int, max_share: float,
        expected: float
) -> None:
    """Test `score_noncyclicity` function."""
    result = score_noncyclicity(roll, max_n_time_steps, max_share)
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
        (
            # `roll`
            np.array([
                [0, 1, 0, 1, 0],
                [0, 1, 1, 0, 0],
                [1, 0, 0, 1, 0],
                [0, 1, 0, 0, 1]
            ]),
            # `interval_consonances`
            {1: -1, 2: -0.5, 3: 0},
            # `distance_weights`
            {0: 1, 1: 1, 2: 0.5, 3: 0.25},
            # `expected`
            -11.75
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


@pytest.mark.parametrize(
    "roll, max_n_semitones, max_n_time_steps, expected",
    [
        (
            # `roll`
            np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]),
            # `max_n_semitones`
            1,
            # `max_n_time_steps`
            1,
            # `expected`
            3
        ),
        (
            # `roll`
            np.array([
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 1, 0, 1],
            ]),
            # `max_n_semitones`
            1,
            # `max_n_time_steps`
            1,
            # `expected`
            1
        ),
        (
            # `roll`
            np.array([
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]),
            # `max_n_semitones`
            1,
            # `max_n_time_steps`
            1,
            # `expected`
            2
        ),
        (
            # `roll`
            np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]),
            # `max_n_semitones`
            1,
            # `max_n_time_steps`
            2,
            # `expected`
            2
        ),
        (
            # `roll`
            np.array([
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 1, 0, 0],
                [0, 0, 1, 1],
            ]),
            # `max_n_semitones`
            2,
            # `max_n_time_steps`
            1,
            # `expected`
            3
        ),
        (
            # `roll`
            np.array([
                [1, 1, 1, 1],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 1],
            ]),
            # `max_n_semitones`
            3,
            # `max_n_time_steps`
            3,
            # `expected`
            5
        ),
    ]
)
def test_score_conjunct_motion(
        roll: np.ndarray, max_n_semitones: int, max_n_time_steps: int,
        expected: int
) -> None:
    """Test `score_conjunct_motion` function."""
    result = score_conjunct_motion(roll, max_n_semitones, max_n_time_steps)
    assert result == expected
