"""
Test `rlmusician.environment.rules` module.

Author: Nikolay Lysenko
"""


from typing import List, Optional

import pytest

from rlmusician.environment.piece import LineElement
from rlmusician.environment.rules import (
    check_absence_of_pitch_class_clashes,
    check_resolution_of_submediant_and_leading_tone,
    check_step_motion_to_final_pitch,
)


@pytest.mark.parametrize(
    "sonority, expected",
    [
        (
            # `sonority
            [
                LineElement(10, 4, 5, True, [-2, -1, 0, 1, 2]),
                LineElement(12, 5, 6, False, [-2, -1, 0, 1, 2])
            ],
            # `expected`
            True
        ),
        (
            # `sonority
            [
                LineElement(3, 0, 1, True, [0, 1, 2]),
                LineElement(15, 7, 1, True, [-2, -1, 0])
            ],
            # `expected`
            False
        ),
    ]
)
def test_check_absence_of_pitch_class_clashes(
        sonority: List[LineElement], expected: bool
) -> None:
    """Test `check_absence_of_pitch_class_clashes` function."""
    result = check_absence_of_pitch_class_clashes(sonority)
    assert result == expected


@pytest.mark.parametrize(
    "line, measure, movement, expected",
    [
        (
            # `line`
            [
                LineElement(10, 4, 5, True, [-2, -1, 0, 1, 2]),
                LineElement(12, 5, 6, False, [-2, -1, 0, 1, 2]),
                LineElement(14, 6, 7, False, [-2, -1, 0, 1]),
                None,
                None
            ],
            # `measure`
            2,
            # `movement`
            1,
            # `expected`
            True
        ),
        (
            # `line`
            [
                LineElement(10, 4, 5, True, [-2, -1, 0, 1, 2]),
                LineElement(12, 5, 6, False, [-2, -1, 0, 1, 2]),
                LineElement(14, 6, 7, False, [-2, -1, 0, 1]),
                None,
                None
            ],
            # `measure`
            2,
            # `movement`
            -1,
            # `expected`
            False
        ),
        (
            # `line`
            [
                LineElement(15, 7, 1, True, [-2, -1, 0]),
                LineElement(14, 6, 7, False, [-2, -1, 0, 1]),
                LineElement(12, 5, 6, False, [-2, -1, 0, 1, 2]),
                None,
                None
            ],
            # `measure`
            2,
            # `movement`
            -1,
            # `expected`
            True
        ),
        (
            # `line`
            [
                LineElement(15, 7, 1, True, [-2, -1, 0]),
                LineElement(14, 6, 7, False, [-2, -1, 0, 1]),
                LineElement(12, 5, 6, False, [-2, -1, 0, 1, 2]),
                None,
                None
            ],
            # `measure`
            2,
            # `movement`
            2,
            # `expected`
            False
        ),
    ]
)
def test_check_resolution_of_submediant_and_leading_tone(
        line: List[Optional['LineElement']], measure: int, movement: int,
        expected: bool
) -> None:
    """Test `check_resolution_of_submediant_and_leading_tone` function."""
    result = check_resolution_of_submediant_and_leading_tone(
        line, measure, movement
    )
    assert result == expected


@pytest.mark.parametrize(
    "line, line_elements, measure, movement, prohibit_rearticulation, "
    "expected",
    [
        (
            # `line`
            [
                LineElement(10, 4, 5, True, [-2, -1, 0, 1, 2]),
                LineElement(8, 3, 4, False, [-2, -1, 0, 1, 2]),
                LineElement(7, 2, 3, True, [-2, -1, 0, 1, 2]),
                None,
                LineElement(3, 0, 1, True, [0, 1, 2]),
            ],
            # `line_elements`
            [
                LineElement(3, 0, 1, True, [0, 1, 2]),
                LineElement(5, 1, 2, False, [-1, 0, 1, 2]),
                LineElement(7, 2, 3, True, [-2, -1, 0, 1, 2]),
                LineElement(8, 3, 4, False, [-2, -1, 0, 1, 2]),
                LineElement(10, 4, 5, True, [-2, -1, 0, 1, 2]),
                LineElement(12, 5, 6, False, [-2, -1, 0, 1, 2]),
                LineElement(14, 6, 7, False, [-2, -1, 0, 1]),
                LineElement(15, 7, 1, True, [-2, -1, 0]),
            ],
            # `measure`
            2,
            # `movement`
            -2,
            # `prohibit_rearticulation`
            True,
            # `expected`
            False
        ),
        (
            # `line`
            [
                LineElement(10, 4, 5, True, [-2, -1, 0, 1, 2]),
                LineElement(8, 3, 4, False, [-2, -1, 0, 1, 2]),
                LineElement(7, 2, 3, True, [-2, -1, 0, 1, 2]),
                None,
                LineElement(3, 0, 1, True, [0, 1, 2]),
            ],
            # `line_elements`
            [
                LineElement(3, 0, 1, True, [0, 1, 2]),
                LineElement(5, 1, 2, False, [-1, 0, 1, 2]),
                LineElement(7, 2, 3, True, [-2, -1, 0, 1, 2]),
                LineElement(8, 3, 4, False, [-2, -1, 0, 1, 2]),
                LineElement(10, 4, 5, True, [-2, -1, 0, 1, 2]),
                LineElement(12, 5, 6, False, [-2, -1, 0, 1, 2]),
                LineElement(14, 6, 7, False, [-2, -1, 0, 1]),
                LineElement(15, 7, 1, True, [-2, -1, 0]),
            ],
            # `measure`
            2,
            # `movement`
            -2,
            # `prohibit_rearticulation`
            False,
            # `expected`
            True
        ),
    ]
)
def test_check_step_motion_to_final_pitch(
        line: List[Optional[LineElement]], line_elements: List[LineElement],
        measure: int, movement: int, prohibit_rearticulation: bool,
        expected: bool
) -> None:
    """Test `check_step_motion_to_final_pitch` function."""
    result = check_step_motion_to_final_pitch(
        line, line_elements, measure, movement, prohibit_rearticulation
    )
    assert result == expected
