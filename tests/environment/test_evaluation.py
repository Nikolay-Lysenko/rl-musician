"""
Test `rlmusician.environment.evaluation` module.

Author: Nikolay Lysenko
"""


from typing import List

import pytest

from rlmusician.environment.evaluation import (
    evaluate_absence_of_unisons, evaluate_lines_correlation
)
from rlmusician.environment.piece import Piece


@pytest.mark.parametrize(
    "piece, all_movements, expected",
    [
        (
            # `piece`
            Piece(
                tonic='C',
                scale='major',
                n_measures=5,
                max_skip=2,
                line_specifications=[
                    {
                        'lowest_note': 'G3',
                        'highest_note': 'G4',
                        'start_note': 'C4',
                        'end_note': 'G4'
                    },
                    {
                        'lowest_note': 'G4',
                        'highest_note': 'G5',
                        'start_note': 'C5',
                        'end_note': 'G5'
                    },
                    {
                        'lowest_note': 'G5',
                        'highest_note': 'G6',
                        'start_note': 'G6',
                        'end_note': 'C6'
                    },
                ],
                rendering_params={}
            ),
            # `all_movements`,
            [
                [1, 1, -1],
                [1, 1, -1],
                [1, 1, -1],
            ],
            # `expected`
            -5 / 9
        ),
    ]
)
def test_evaluate_absence_of_unisons(
        piece: Piece, all_movements: List[List[int]], expected: float
) -> None:
    """Test `evaluate_absence_of_unisons` function."""
    for movements in all_movements:
        piece.add_measure(movements)
    result = evaluate_absence_of_unisons(piece)
    assert result == expected


@pytest.mark.parametrize(
    "piece, all_movements, expected",
    [
        (
            # `piece`
            Piece(
                tonic='C',
                scale='major',
                n_measures=5,
                max_skip=2,
                line_specifications=[
                    {
                        'lowest_note': 'G3',
                        'highest_note': 'G4',
                        'start_note': 'C4',
                        'end_note': 'G4'
                    },
                    {
                        'lowest_note': 'G4',
                        'highest_note': 'G5',
                        'start_note': 'C5',
                        'end_note': 'G5'
                    },
                    {
                        'lowest_note': 'G5',
                        'highest_note': 'G6',
                        'start_note': 'G6',
                        'end_note': 'C6'
                    },
                ],
                rendering_params={}
            ),
            # `all_movements`,
            [
                [1, 1, -1],
                [1, 1, -1],
                [1, 1, -1],
            ],
            # `expected`
            0.6621
        ),
    ]
)
def test_evaluate_lines_correlation(
        piece: Piece, all_movements: List[List[int]], expected: float
) -> None:
    """Test `evaluate_lines_correlation` function."""
    for movements in all_movements:
        piece.add_measure(movements)
    result = evaluate_lines_correlation(piece)
    assert round(result, 4) == expected
