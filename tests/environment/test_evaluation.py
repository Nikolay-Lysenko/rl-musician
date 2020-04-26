"""
Test `rlmusician.environment.evaluation` module.

Author: Nikolay Lysenko
"""


from typing import List, Optional, Tuple

import pytest

from rlmusician.environment.evaluation import (
    evaluate_absence_of_looped_fragments,
    evaluate_climax_explicity,
    evaluate_entropy,
    evaluate_number_of_skips,
)
from rlmusician.environment.piece import Piece


@pytest.mark.parametrize(
    "piece, steps, min_size, max_size, expected",
    [
        (
            # `piece`
            Piece(
                tonic='C',
                scale_type='major',
                cantus_firmus=['C4', 'D4', 'C4', 'D4', 'C4'],
                counterpoint_specifications={
                    'start_note': 'E4',
                    'end_note': 'E4',
                    'lowest_note': 'G3',
                    'highest_note': 'G4',
                    'start_pause_in_eights': 4,
                    'max_skip_in_degrees': 2,
                },
                rules={
                    'names': ['rearticulation_stability'],
                    'params': {}
                },
                rendering_params={}
            ),
            # `steps`,
            [(0, 4), (0, 4), (0, 4), (0, 4), (0, 4), (0, 4)],
            # `min_size`
            8,
            # `max_size`
            None,
            # `expected`
            -5
        ),
        (
            # `piece`
            Piece(
                tonic='C',
                scale_type='major',
                cantus_firmus=['C4', 'D4', 'E4', 'D4', 'C4'],
                counterpoint_specifications={
                    'start_note': 'E4',
                    'end_note': 'E4',
                    'lowest_note': 'G3',
                    'highest_note': 'G4',
                    'start_pause_in_eights': 4,
                    'max_skip_in_degrees': 2,
                },
                rules={
                    'names': ['rearticulation_stability'],
                    'params': {}
                },
                rendering_params={}
            ),
            # `steps`,
            [(0, 4), (0, 4), (0, 4), (0, 4), (0, 4), (0, 4)],
            # `min_size`
            8,
            # `max_size`
            None,
            # `expected`
            0
        ),
    ]
)
def test_evaluate_absence_of_looped_fragments(
        piece: Piece, steps: List[Tuple[int, int]],
        min_size: Optional[int], max_size: Optional[int], expected: float
) -> None:
    """Test `evaluate_absence_of_looped_fragments` function."""
    for movement, duration in steps:
        piece.add_line_element(movement, duration)
    result = evaluate_absence_of_looped_fragments(piece, min_size, max_size)
    assert result == expected


@pytest.mark.parametrize(
    "piece, steps, shortage_penalty, duplication_penalty, expected",
    [
        (
            # `piece`
            Piece(
                tonic='C',
                scale_type='major',
                cantus_firmus=['C4', 'D4', 'E4', 'D4', 'C4'],
                counterpoint_specifications={
                    'start_note': 'E4',
                    'end_note': 'E4',
                    'lowest_note': 'G3',
                    'highest_note': 'E4',
                    'start_pause_in_eights': 4,
                    'max_skip_in_degrees': 2,
                },
                rules={
                    'names': ['rearticulation_stability'],
                    'params': {}
                },
                rendering_params={}
            ),
            # `steps`,
            [(-1, 4), (-1, 4), (0, 4), (-1, 4), (1, 4), (1, 4)],
            # `shortage_penalty`
            0.3,
            # `duplication_penalty`
            0.5,
            # `expected`
            0.5
        ),
        (
            # `piece`
            Piece(
                tonic='C',
                scale_type='major',
                cantus_firmus=['C4', 'D4', 'E4', 'D4', 'C4'],
                counterpoint_specifications={
                    'start_note': 'E4',
                    'end_note': 'E4',
                    'lowest_note': 'G3',
                    'highest_note': 'G4',
                    'start_pause_in_eights': 4,
                    'max_skip_in_degrees': 2,
                },
                rules={
                    'names': ['rearticulation_stability'],
                    'params': {}
                },
                rendering_params={}
            ),
            # `steps`,
            [(1, 4), (-1, 4), (0, 4), (0, 4), (0, 4), (0, 4)],
            # `shortage_penalty`
            0.3,
            # `duplication_penalty`
            0.5,
            # `expected`
            0.7
        ),
    ]
)
def test_evaluate_climax_explicity(
        piece: Piece, steps: List[Tuple[int, int]],
        shortage_penalty: float, duplication_penalty: float, expected: float
) -> None:
    """Test `evaluate_climax_explicity` function."""
    for movement, duration in steps:
        piece.add_line_element(movement, duration)
    result = evaluate_climax_explicity(
        piece, shortage_penalty, duplication_penalty
    )
    assert round(result, 4) == expected


@pytest.mark.parametrize(
    "piece, steps, expected",
    [
        (
            # `piece`
            Piece(
                tonic='C',
                scale_type='major',
                cantus_firmus=['C4', 'D4', 'E4', 'D4', 'C4'],
                counterpoint_specifications={
                    'start_note': 'E4',
                    'end_note': 'G3',
                    'lowest_note': 'G3',
                    'highest_note': 'G4',
                    'start_pause_in_eights': 4,
                    'max_skip_in_degrees': 3,
                },
                rules={
                    'names': ['rearticulation_stability'],
                    'params': {}
                },
                rendering_params={}
            ),
            # `steps`,
            [(1, 4), (1, 4), (-3, 4), (-1, 4), (-1, 4), (-1, 4)],
            # `expected`
            1
        ),
        (
            # `piece`
            Piece(
                tonic='C',
                scale_type='major',
                cantus_firmus=['C4', 'D4', 'E4', 'D4', 'C4'],
                counterpoint_specifications={
                    'start_note': 'E4',
                    'end_note': 'E4',
                    'lowest_note': 'G3',
                    'highest_note': 'G4',
                    'start_pause_in_eights': 4,
                    'max_skip_in_degrees': 3,
                },
                rules={
                    'names': ['rearticulation_stability'],
                    'params': {}
                },
                rendering_params={}
            ),
            # `steps`,
            [(1, 4), (1, 4), (-3, 4), (-1, 4), (-1, 4), (-1, 2), (-1, 2)],
            # `expected`
            0.9826
        ),
    ]
)
def test_evaluate_entropy(
        piece: Piece, steps: List[Tuple[int, int]], expected: float
) -> None:
    """Test `evaluate_entropy` function."""
    for movement, duration in steps:
        piece.add_line_element(movement, duration)
    result = evaluate_entropy(piece)
    assert round(result, 4) == expected


@pytest.mark.parametrize(
    "piece, steps, min_n_skips, max_n_skips, expected",
    [
        (
            # `piece`
            Piece(
                tonic='C',
                scale_type='major',
                cantus_firmus=['C4', 'D4', 'E4', 'D4', 'C4'],
                counterpoint_specifications={
                    'start_note': 'E4',
                    'end_note': 'E4',
                    'lowest_note': 'G3',
                    'highest_note': 'G4',
                    'start_pause_in_eights': 4,
                    'max_skip_in_degrees': 3,
                },
                rules={
                    'names': ['rearticulation_stability'],
                    'params': {}
                },
                rendering_params={}
            ),
            # `steps`,
            [(1, 4), (1, 4), (-3, 4), (-1, 4), (-1, 4), (-1, 4)],
            # `min_n_skips`
            1,
            # `max_n_skips`
            2,
            # `expected`
            1
        ),
    ]
)
def test_evaluate_number_of_skips(
        piece: Piece, steps: List[Tuple[int, int]],
        min_n_skips: int, max_n_skips: int, expected: float
) -> None:
    """Test `evaluate_number_of_skips` function."""
    for movement, duration in steps:
        piece.add_line_element(movement, duration)
    result = evaluate_number_of_skips(piece, min_n_skips, max_n_skips)
    assert result == expected
