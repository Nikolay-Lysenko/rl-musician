"""
Test `rlmusician.environment.rules` module.

Author: Nikolay Lysenko
"""


from typing import List

import pytest

from rlmusician.environment.piece import LineElement
from rlmusician.environment.rules import (
    check_absence_of_large_intervals,
    check_absence_of_lines_crossing,
    check_absence_of_monotonous_long_motion,
    check_absence_of_overlapping_motion,
    check_absence_of_skip_series,
    check_absence_of_stalled_pitches,
    check_consonance_on_strong_beat,
    check_resolution_of_submediant_and_leading_tone,
    check_resolution_of_suspended_dissonance,
    check_stability_of_rearticulated_pitch,
    check_step_motion_from_dissonance,
    check_step_motion_to_dissonance,
    check_step_motion_to_final_pitch,
    check_that_skip_is_followed_by_opposite_step_motion,
    check_validity_of_rhythmic_pattern,
)
from rlmusician.utils import ScaleElement


@pytest.mark.parametrize(
    "counterpoint_continuation, cantus_firmus_elements, max_n_semitones, "
    "expected",
    [
        (
            LineElement(ScaleElement('C1', 3, 2, 1, True), 16, 20),
            [LineElement(ScaleElement('C2', 15, 9, 1, True), 16, 24)],
            7,
            False
        ),
        (
            LineElement(ScaleElement('C1', 3, 2, 1, True), 16, 20),
            [LineElement(ScaleElement('C2', 15, 9, 1, True), 16, 24)],
            16,
            True
        ),
    ]
)
def test_check_absence_of_large_intervals(
        counterpoint_continuation: LineElement,
        cantus_firmus_elements: List[LineElement],
        max_n_semitones: int,
        expected: bool
) -> None:
    """Test `check_absence_of_large_intervals` function."""
    result = check_absence_of_large_intervals(
        counterpoint_continuation, cantus_firmus_elements, max_n_semitones
    )
    assert result == expected


@pytest.mark.parametrize(
    "counterpoint_continuation, cantus_firmus_elements, "
    "is_counterpoint_above, prohibit_unisons, expected",
    [
        (
            LineElement(ScaleElement('C1', 3, 2, 1, True), 16, 20),
            [LineElement(ScaleElement('C1', 3, 2, 1, True), 16, 24)],
            True,
            True,
            False
        ),
        (
            LineElement(ScaleElement('D1', 5, 3, 2, False), 16, 20),
            [LineElement(ScaleElement('C1', 3, 2, 1, True), 16, 24)],
            True,
            True,
            True
        ),
        (
            LineElement(ScaleElement('D1', 5, 3, 2, False), 16, 20),
            [LineElement(ScaleElement('C1', 3, 2, 1, True), 16, 24)],
            False,
            True,
            False
        ),
    ]
)
def test_check_absence_of_lines_crossing(
        counterpoint_continuation: LineElement,
        cantus_firmus_elements: List[LineElement],
        is_counterpoint_above: bool, prohibit_unisons: bool, expected: bool
) -> None:
    """Test `check_absence_of_lines_crossing` function."""
    result = check_absence_of_lines_crossing(
        counterpoint_continuation, cantus_firmus_elements,
        is_counterpoint_above, prohibit_unisons
    )
    assert result == expected


@pytest.mark.parametrize(
    "counterpoint_continuation, current_motion_start_element, "
    "max_distance_in_semitones, expected",
    [
        (
            LineElement(ScaleElement('C1', 3, 2, 1, True), 24, 28),
            LineElement(ScaleElement('C2', 15, 9, 1, True), 8, 10),
            9,
            False
        ),
        (
            LineElement(ScaleElement('C1', 3, 2, 1, True), 24, 28),
            LineElement(ScaleElement('C2', 15, 9, 1, True), 8, 10),
            12,
            True
        ),
    ]
)
def test_check_absence_of_monotonous_long_motion(
        counterpoint_continuation: LineElement,
        current_motion_start_element: LineElement,
        max_distance_in_semitones: int,
        expected: bool
) -> None:
    """Test `check_absence_of_monotonous_long_motion` function."""
    result = check_absence_of_monotonous_long_motion(
        counterpoint_continuation,
        current_motion_start_element,
        max_distance_in_semitones
    )
    assert result == expected


@pytest.mark.parametrize(
    "counterpoint_continuation, previous_cantus_firmus_element, "
    "is_counterpoint_above, expected",
    [
        (
            LineElement(ScaleElement('C1', 3, 2, 1, True), 24, 28),
            LineElement(ScaleElement('C2', 15, 9, 1, True), 16, 24),
            True,
            False
        ),
        (
            LineElement(ScaleElement('C1', 3, 2, 1, True), 24, 28),
            LineElement(ScaleElement('C2', 15, 9, 1, True), 16, 24),
            False,
            True
        ),
        (
            LineElement(ScaleElement('C1', 3, 2, 1, True), 24, 28),
            LineElement(ScaleElement('C1', 3, 2, 1, True), 16, 24),
            True,
            False
        ),
        (
            LineElement(ScaleElement('C1', 3, 2, 1, True), 24, 28),
            LineElement(ScaleElement('C1', 3, 2, 1, True), 16, 24),
            False,
            False
        ),
    ]
)
def test_check_absence_of_overlapping_motion(
        counterpoint_continuation: LineElement,
        previous_cantus_firmus_element: LineElement,
        is_counterpoint_above: bool, expected: bool
) -> None:
    """Test `check_absence_of_overlapping_motion` function."""
    result = check_absence_of_overlapping_motion(
        counterpoint_continuation, previous_cantus_firmus_element,
        is_counterpoint_above
    )
    assert result == expected


@pytest.mark.parametrize(
    "movement, past_movements, max_n_skips, expected",
    [
        (0, [2, 2], 2, True),
        (3, [3, -2], 2, False),
        (2, [-3, 2], 3, True),
        (4, [], 1, True),
        (4, [], 0, False),
        (-2, [2, 0, 1, 2], 2, True),
    ]
)
def test_check_absence_of_skip_series(
        movement: int, past_movements: List[int], max_n_skips: int,
        expected: bool
) -> None:
    """Test `check_absence_of_skip_series` function."""
    result = check_absence_of_skip_series(
        movement, past_movements, max_n_skips
    )
    assert result == expected


@pytest.mark.parametrize(
    "movement, past_movements, max_n_repetitions, expected",
    [
        (1, [1, 2, 0], 2, True),
        (0, [], 2, True),
        (0, [0], 2, False),
        (0, [0, 0, 0], 5, True),
        (0, [0, 1, 0, 1], 2, True)
    ]
)
def test_check_absence_of_stalled_pitches(
        movement: int, past_movements: List[int], max_n_repetitions: int,
        expected: bool
) -> None:
    """Test `check_absence_of_stalled_pitches` function."""
    result = check_absence_of_stalled_pitches(
        movement, past_movements, max_n_repetitions
    )
    assert result == expected


@pytest.mark.parametrize(
    "counterpoint_continuation, cantus_firmus_elements, expected",
    [
        (
            LineElement(ScaleElement('C2', 15, 9, 1, True), 16, 20),
            [LineElement(ScaleElement('B1', 14, 8, 7, False), 16, 24)],
            False
        ),
        (
            LineElement(ScaleElement('C2', 15, 9, 1, True), 18, 20),
            [LineElement(ScaleElement('B1', 14, 8, 7, False), 16, 24)],
            True
        ),
        (
            LineElement(ScaleElement('C2', 15, 9, 1, True), 20, 28),
            [
                LineElement(ScaleElement('A1', 12, 7, 6, False), 16, 24),
                LineElement(ScaleElement('B1', 14, 8, 7, False), 24, 32)
            ],
            True
        ),
    ]
)
def test_check_consonance_on_strong_beat(
        counterpoint_continuation: LineElement,
        cantus_firmus_elements: List[LineElement],
        expected: bool
) -> None:
    """Test `check_consonance_on_strong_beat` function."""
    result = check_consonance_on_strong_beat(
        counterpoint_continuation, cantus_firmus_elements
    )
    assert result == expected


@pytest.mark.parametrize(
    "line, movement, expected",
    [
        (
            [
                LineElement(ScaleElement('G1', 10, 6, 5, True), 4, 8)
            ],
            1,
            True
        ),
        (
            [
                LineElement(ScaleElement('G1', 10, 6, 5, True), 4, 8),
                LineElement(ScaleElement('A1', 12, 7, 6, False), 8, 12),
                LineElement(ScaleElement('B1', 14, 8, 7, False), 12, 16),
            ],
            1,
            True
        ),
        (
            [
                LineElement(ScaleElement('G1', 10, 6, 5, True), 4, 8),
                LineElement(ScaleElement('A1', 12, 7, 6, False), 8, 12),
                LineElement(ScaleElement('B1', 14, 8, 7, False), 12, 16),
            ],
            -1,
            False
        ),
        (
            [
                LineElement(ScaleElement('C2', 15, 9, 1, True), 4, 8),
                LineElement(ScaleElement('B1', 14, 8, 7, False), 8, 12),
                LineElement(ScaleElement('A1', 12, 7, 6, False), 12, 16),
            ],
            -1,
            True
        ),
        (
            [
                LineElement(ScaleElement('C2', 15, 7, 1, True), 4, 8),
                LineElement(ScaleElement('B1', 14, 6, 7, False), 8, 12),
                LineElement(ScaleElement('A1', 12, 5, 6, False), 12, 16),
            ],
            1,
            False
        ),
        (
            [
                LineElement(ScaleElement('C2', 15, 7, 1, True), 8, 12),
                LineElement(ScaleElement('C2', 15, 7, 1, True), 12, 16),
            ],
            1,
            True
        ),
    ]
)
def test_check_resolution_of_submediant_and_leading_tone(
        line: List[LineElement], movement: int, expected: bool
) -> None:
    """Test `check_resolution_of_submediant_and_leading_tone` function."""
    result = check_resolution_of_submediant_and_leading_tone(
        line, movement
    )
    assert result == expected


@pytest.mark.parametrize(
    "line, movement, counterpoint_continuation, cantus_firmus_elements, "
    "is_last_element_consonant, expected",
    [
        (
            [
                LineElement(ScaleElement('C1', 3, 2, 1, True), 4, 8),
                LineElement(ScaleElement('D1', 5, 3, 2, False), 8, 12),
            ],
            0,
            LineElement(ScaleElement('D1', 5, 3, 2, False), 12, 16),
            [LineElement(ScaleElement('C1', 3, 2, 1, True), 8, 16)],
            False,
            True
        ),
        (
            [
                LineElement(ScaleElement('C1', 3, 2, 1, True), 4, 8),
                LineElement(ScaleElement('D1', 5, 3, 2, False), 8, 12),
                LineElement(ScaleElement('D1', 5, 3, 2, False), 12, 20),
            ],
            2,
            LineElement(ScaleElement('F1', 8, 5, 4, False), 20, 24),
            [LineElement(ScaleElement('D1', 5, 3, 2, False), 16, 24)],
            True,
            True
        ),
        (
            [
                LineElement(ScaleElement('C1', 3, 2, 1, True), 4, 8),
                LineElement(ScaleElement('D1', 5, 3, 2, False), 8, 12),
                LineElement(ScaleElement('D1', 5, 3, 2, False), 12, 20),
            ],
            2,
            LineElement(ScaleElement('F1', 8, 5, 4, False), 20, 24),
            [LineElement(ScaleElement('C1', 3, 2, 1, True), 16, 24)],
            False,
            False
        ),
        (
            [
                LineElement(ScaleElement('C1', 3, 2, 1, True), 4, 8),
                LineElement(ScaleElement('D1', 5, 3, 2, False), 8, 12),
                LineElement(ScaleElement('D1', 5, 3, 2, False), 12, 20),
            ],
            -1,
            LineElement(ScaleElement('C1', 3, 2, 1, False), 20, 24),
            [LineElement(ScaleElement('C1', 3, 2, 1, True), 16, 24)],
            False,
            True
        ),
    ]
)
def test_check_resolution_of_suspended_dissonance(
        line: List[LineElement],
        movement: int,
        counterpoint_continuation: LineElement,
        cantus_firmus_elements: List[LineElement],
        is_last_element_consonant: bool,
        expected: bool
) -> None:
    """Test `check_resolution_of_suspended_dissonance` function."""
    result = check_resolution_of_suspended_dissonance(
        line, movement, counterpoint_continuation, cantus_firmus_elements,
        is_last_element_consonant
    )
    assert result == expected


@pytest.mark.parametrize(
    "counterpoint_continuation, movement, expected",
    [
        (LineElement(ScaleElement('D1', 5, 3, 2, False), 4, 8), 1, True),
        (LineElement(ScaleElement('D1', 5, 3, 2, False), 4, 8), 0, False),
        (LineElement(ScaleElement('C1', 3, 2, 1, True), 4, 8), 0, True),
    ]
)
def test_check_stability_of_rearticulated_pitch(
        counterpoint_continuation: LineElement, movement: int, expected: bool
) -> None:
    """Test `check_stability_of_rearticulated_pitch` function."""
    result = check_stability_of_rearticulated_pitch(
        counterpoint_continuation, movement
    )
    assert result == expected


@pytest.mark.parametrize(
    "movement, is_last_element_consonant, expected",
    [
        (2, True, True),
        (-1, True, True),
        (2, False, False),
        (-1, False, True),
    ]
)
def test_check_step_motion_from_dissonance(
        movement: int, is_last_element_consonant: bool, expected: bool
) -> None:
    """Test `check_step_motion_from_dissonance` function."""
    result = check_step_motion_from_dissonance(
        movement, is_last_element_consonant
    )
    assert result == expected


@pytest.mark.parametrize(
    "counterpoint_continuation, cantus_firmus_elements, movement, expected",
    [
        (
            LineElement(ScaleElement('C2', 15, 9, 1, True), 20, 28),
            [
                LineElement(ScaleElement('A1', 12, 7, 6, False), 16, 24),
                LineElement(ScaleElement('B1', 14, 8, 7, False), 24, 32)
            ],
            2,
            True
        ),
        (
            LineElement(ScaleElement('C2', 15, 9, 1, True), 20, 28),
            [LineElement(ScaleElement('B1', 14, 8, 7, False), 16, 24)],
            2,
            False
        ),
        (
            LineElement(ScaleElement('C2', 15, 9, 1, True), 20, 28),
            [LineElement(ScaleElement('B1', 14, 8, 7, False), 16, 24)],
            -1,
            True
        ),
    ]
)
def test_check_step_motion_to_dissonance(
        counterpoint_continuation: LineElement,
        cantus_firmus_elements: List[LineElement],
        movement: int,
        expected: bool
) -> None:
    """Test `check_step_motion_to_dissonance` function."""
    result = check_step_motion_to_dissonance(
        counterpoint_continuation, cantus_firmus_elements, movement
    )
    assert result == expected


@pytest.mark.parametrize(
    "counterpoint_continuation, counterpoint_end, piece_duration, "
    "prohibit_rearticulation, expected",
    [
        (
            LineElement(ScaleElement('D1', 5, 3, 2, False), 28, 32),
            ScaleElement('C1', 3, 2, 1, True),
            40,
            True,
            True
        ),
        (
            LineElement(ScaleElement('C1', 3, 2, 1, True), 28, 32),
            ScaleElement('C1', 3, 2, 1, True),
            40,
            True,
            False
        ),
        (
            LineElement(ScaleElement('C1', 3, 2, 1, True), 28, 32),
            ScaleElement('C1', 3, 2, 1, True),
            40,
            False,
            True
        ),
        (
            LineElement(ScaleElement('G1', 10, 6, 5, True), 24, 26),
            ScaleElement('C1', 3, 2, 1, True),
            40,
            False,
            True
        ),
        (
            LineElement(ScaleElement('A1', 11, 7, 6, False), 24, 26),
            ScaleElement('C1', 3, 2, 1, True),
            40,
            False,
            False
        ),
    ]
)
def test_check_step_motion_to_final_pitch(
        counterpoint_continuation: LineElement, counterpoint_end: ScaleElement,
        piece_duration: int, prohibit_rearticulation: bool, expected: bool
) -> None:
    """Test `check_step_motion_to_final_pitch` function."""
    result = check_step_motion_to_final_pitch(
        counterpoint_continuation, counterpoint_end, piece_duration,
        prohibit_rearticulation
    )
    assert result == expected


@pytest.mark.parametrize(
    "movement, past_movements, min_n_scale_degrees, expected",
    [
        (2, [1, 1, 2], 3, True),
        (2, [1, 1, 2], 2, False),
        (-1, [2], 2, True),
        (3, [], 2, True),
        (1, [-3], 3, True),
        (-1, [-3], 3, False),
    ]
)
def test_check_that_skip_is_followed_by_opposite_step_motion(
        movement: int, past_movements: List[int], min_n_scale_degrees: int,
        expected: bool
) -> None:
    """Test `check_that_skip_is_followed_by_opposite_step_motion` function."""
    result = check_that_skip_is_followed_by_opposite_step_motion(
        movement, past_movements, min_n_scale_degrees
    )
    assert result == expected


@pytest.mark.parametrize(
    "durations, expected",
    [
        ([4, 2, 1], True),
        ([], True),
        ([4, 1, 1, 1, 1], False),
        ([1, 2], False)
    ]
)
def test_check_validity_of_rhythmic_pattern(
        durations: List[int], expected: bool
) -> None:
    """Test `check_validity_of_rhythmic_pattern` function."""
    result = check_validity_of_rhythmic_pattern(durations)
    assert result == expected
