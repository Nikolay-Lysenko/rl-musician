"""
Test `rlmusician.environment.evaluation` module.

Author: Nikolay Lysenko
"""


from typing import List

import pytest

from rlmusician.environment.evaluation import (
    evaluate_autocorrelation,
    evaluate_entropy,
    evaluate_absence_of_pitch_class_clashes,
    evaluate_independence_of_motion,
    evaluate_lines_correlation
)
from rlmusician.environment.piece import Piece


@pytest.mark.parametrize(
    "piece, all_movements, max_lag, expected",
    [
        (
            # `piece`
            Piece(
                tonic='C',
                scale='major',
                n_measures=9,
                max_skip=2,
                line_specifications=[
                    {
                        'lowest_note': 'C4',
                        'highest_note': 'E4',
                        'start_note': 'C4',
                        'end_note': 'C4'
                    }
                ],
                rendering_params={}
            ),
            # `all_movements`,
            [[1], [-1], [1], [-1], [1], [-1], [1]],
            # `max_lag`
            3,
            # `expected`
            0
        ),
        (
            # `piece`
            Piece(
                tonic='C',
                scale='major',
                n_measures=4,
                max_skip=2,
                line_specifications=[
                    {
                        'lowest_note': 'C4',
                        'highest_note': 'E4',
                        'start_note': 'E4',
                        'end_note': 'C4'
                    }
                ],
                rendering_params={}
            ),
            # `all_movements`,
            [[0], [-2]],
            # `max_lag`
            2,
            # `expected`
            1
        ),
    ]
)
def test_evaluate_autocorrelation(
        piece: Piece, all_movements: List[List[int]], max_lag: int,
        expected: float
) -> None:
    """Test `evaluate_autocorrelation` function."""
    for movements in all_movements:
        piece.add_measure(movements)
    result = evaluate_autocorrelation(piece, max_lag)
    assert round(result, 4) == expected


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
                        'lowest_note': 'C4',
                        'highest_note': 'G4',
                        'start_note': 'C4',
                        'end_note': 'G4'
                    }
                ],
                rendering_params={}
            ),
            # `all_movements`,
            [[1], [1], [1]],
            # `expected`
            1
        ),
        (
            # `piece`
            Piece(
                tonic='C',
                scale='major',
                n_measures=5,
                max_skip=2,
                line_specifications=[
                    {
                        'lowest_note': 'C5',
                        'highest_note': 'G5',
                        'start_note': 'C5',
                        'end_note': 'C5'
                    }
                ],
                rendering_params={}
            ),
            # `all_movements`,
            [[0], [0], [0]],
            # `expected`
            0
        ),
    ]
)
def test_evaluate_entropy(
        piece: Piece, all_movements: List[List[int]], expected: float
) -> None:
    """Test `evaluate_entropy` function."""
    for movements in all_movements:
        piece.add_measure(movements)
    result = evaluate_entropy(piece)
    assert round(result, 4) == expected


@pytest.mark.parametrize(
    "piece, all_movements, pure_clash_coef, expected",
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
            # `pure_clash_coef`
            2,
            # `expected`
            -5 / 9
        ),
        (
            # `piece`
            Piece(
                tonic='C',
                scale='major',
                n_measures=4,
                max_skip=2,
                line_specifications=[
                    {
                        'lowest_note': 'B3',
                        'highest_note': 'B4',
                        'start_note': 'G4',
                        'end_note': 'G4'
                    },
                    {
                        'lowest_note': 'G4',
                        'highest_note': 'G5',
                        'start_note': 'C5',
                        'end_note': 'G4'
                    },
                ],
                rendering_params={}
            ),
            # `all_movements`,
            [
                [0, -1],
                [1, -1],
            ],
            # `pure_clash_coef`
            3,
            # `expected`
            -1.5
        ),
    ]
)
def test_evaluate_absence_of_pitch_class_clashes(
        piece: Piece, all_movements: List[List[int]], pure_clash_coef: float,
        expected: float
) -> None:
    """Test `evaluate_absence_of_pitch_class_clashes` function."""
    for movements in all_movements:
        piece.add_measure(movements)
    result = evaluate_absence_of_pitch_class_clashes(piece, pure_clash_coef)
    assert result == expected


@pytest.mark.parametrize(
    "piece, all_movements, "
    "parallel_coef, similar_coef, oblique_coef, contrary_coef, expected",
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
                        'lowest_note': 'C4',
                        'highest_note': 'C5',
                        'start_note': 'G4',
                        'end_note': 'G4'
                    },
                    {
                        'lowest_note': 'G4',
                        'highest_note': 'G5',
                        'start_note': 'C5',
                        'end_note': 'C5'
                    },
                ],
                rendering_params={}
            ),
            # `all_movements`,
            [
                [-1, 1],
                [-1, -1],
                [2, 1]
            ],
            # `parallel_coef`
            -1,
            # `similar_coef`
            -0.5,
            # `oblique_coef`
            0,
            # `contrary_coef`
            1,
            # `expected`
            -0.125
        ),
    ]
)
def test_evaluate_independence_of_motion(
        piece: Piece, all_movements: List[List[int]], parallel_coef: float,
        similar_coef: float, oblique_coef: float, contrary_coef: float,
        expected: float
) -> None:
    """Test `evaluate_independence_of_motion` function."""
    for movements in all_movements:
        piece.add_measure(movements)
    result = evaluate_independence_of_motion(
        piece, parallel_coef, similar_coef, oblique_coef, contrary_coef
    )
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
                        'end_note': 'C5'
                    },
                ],
                rendering_params={}
            ),
            # `all_movements`,
            [
                [2, 0],
                [0, 0],
                [2, 0],
            ],
            # `expected`
            0
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
