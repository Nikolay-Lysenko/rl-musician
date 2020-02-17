"""
Test `rlmusician.environment.rules` module.

Author: Nikolay Lysenko
"""


from typing import List, Optional

import pytest

from rlmusician.environment.piece import LineElement
from rlmusician.environment.rules import check_step_motion_to_final_pitch


@pytest.mark.parametrize(
    "line, line_elements, measure, movement, prohibit_rearticulation, "
    "expected",
    [
        (
            # `line`
            [
                LineElement(10, 4, True, [-2, -1, 0, 1, 2]),
                LineElement(8, 3, False, [-2, -1, 0, 1, 2]),
                LineElement(7, 2, True, [-2, -1, 0, 1, 2]),
                None,
                LineElement(3, 0, True, [0, 1, 2]),
            ],
            # `line_elements`
            [
                LineElement(3, 0, True, [0, 1, 2]),
                LineElement(5, 1, False, [-1, 0, 1, 2]),
                LineElement(7, 2, True, [-2, -1, 0, 1, 2]),
                LineElement(8, 3, False, [-2, -1, 0, 1, 2]),
                LineElement(10, 4, True, [-2, -1, 0, 1, 2]),
                LineElement(12, 5, False, [-2, -1, 0, 1, 2]),
                LineElement(14, 6, False, [-2, -1, 0, 1]),
                LineElement(15, 7, True, [-2, -1, 0]),
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
                LineElement(10, 4, True, [-2, -1, 0, 1, 2]),
                LineElement(8, 3, False, [-2, -1, 0, 1, 2]),
                LineElement(7, 2, True, [-2, -1, 0, 1, 2]),
                None,
                LineElement(3, 0, True, [0, 1, 2]),
            ],
            # `line_elements`
            [
                LineElement(3, 0, True, [0, 1, 2]),
                LineElement(5, 1, False, [-1, 0, 1, 2]),
                LineElement(7, 2, True, [-2, -1, 0, 1, 2]),
                LineElement(8, 3, False, [-2, -1, 0, 1, 2]),
                LineElement(10, 4, True, [-2, -1, 0, 1, 2]),
                LineElement(12, 5, False, [-2, -1, 0, 1, 2]),
                LineElement(14, 6, False, [-2, -1, 0, 1]),
                LineElement(15, 7, True, [-2, -1, 0]),
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
