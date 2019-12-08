"""
Test `rlmusician.environment.piece` module.

Author: Nikolay Lysenko
"""


from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from rlmusician.environment import Piece


class TestPianoRollEnv:
    """Tests for `Piece` class."""

    @pytest.mark.parametrize(
        "tonic, scale, n_measures, max_skip, line_specifications, match",
        [
            (
                # `tonic`
                'C',
                # `scale`
                'major',
                # `n_measures`
                10,
                # `max_skip`
                2,
                # `line_specifications`
                [
                    {
                        'lowest_note': 'G4',
                        'highest_note': 'G3',
                        'start_note': 'C4',
                        'end_note': 'C4'
                    }
                ],
                # `match`
                "No pitches from .*"
            ),
            (
                # `tonic`
                'C',
                # `scale`
                'major',
                # `n_measures`
                10,
                # `max_skip`
                2,
                # `line_specifications`
                [
                    {
                        'lowest_note': 'G3',
                        'highest_note': 'G4',
                        'start_note': 'C#4',
                        'end_note': 'C4'
                    }
                ],
                # `match`
                ".* does not belong to .*"
            ),
            (
                # `tonic`
                'C',
                # `scale`
                'major',
                # `n_measures`
                10,
                # `max_skip`
                2,
                # `line_specifications`
                [
                    {
                        'lowest_note': 'G3',
                        'highest_note': 'G4',
                        'start_note': 'C4',
                        'end_note': 'D4'
                    }
                ],
                # `match`
                ".* is not a tonic triad member for .*"
            ),
        ]
    )
    def test_improper_initialization(
            self, tonic: str, scale: str, n_measures: int, max_skip: int,
            line_specifications: List[Dict[str, Any]], match: str
    ) -> None:
        """Test that initialization with invalid values is prohibited."""
        with pytest.raises(ValueError, match=match):
            _ = Piece(
                tonic, scale, n_measures, max_skip, line_specifications, {}
            )

    @pytest.mark.parametrize(
        "tonic, scale, n_measures, max_skip, line_specifications, rng, roll",
        [
            (
                # `tonic`
                'D',
                # `scale`
                'major',
                # `n_measures`
                5,
                # `max_skip`
                2,
                # `line_specifications`
                [
                    {
                        'lowest_note': 'G3',
                        'highest_note': 'G4',
                        'start_note': 'F#4',
                        'end_note': 'D4'
                    },
                    {
                        'lowest_note': 'G3',
                        'highest_note': 'G4',
                        'start_note': 'A3',
                        'end_note': 'A3'
                    },
                ],
                # `rng`,
                (34, 46),
                # `roll`
                np.array([
                    [0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ])
            ),
        ]
    )
    def test_initialization(
            self, tonic: str, scale: str, n_measures: int, max_skip: int,
            line_specifications: List[Dict[str, Any]], rng: Tuple[int, int],
            roll: np.ndarray
    ) -> None:
        """Test that initialization with valid values works as expected."""
        piece = Piece(
            tonic, scale, n_measures, max_skip, line_specifications, {}
        )
        assert len(piece.lines) == len(line_specifications)
        assert (piece.lowest_row_to_show, piece.highest_row_to_show) == rng
        np.testing.assert_equal(piece.piano_roll, roll)

    @pytest.mark.parametrize(
        "tonic, scale, n_measures, max_skip, line_specifications, "
        "previous_movements, candidate_movements, expected",
        [
            (
                # `tonic`
                'C',
                # `scale`
                'major',
                # `n_measures`
                10,
                # `max_skip`
                2,
                # `line_specifications`
                [
                    {
                        'lowest_note': 'G3',
                        'highest_note': 'G4',
                        'start_note': 'C4',
                        'end_note': 'C4'
                    },
                    {
                        'lowest_note': 'E4',
                        'highest_note': 'E5',
                        'start_note': 'C5',
                        'end_note': 'G4'
                    },
                    {
                        'lowest_note': 'G4',
                        'highest_note': 'G5',
                        'start_note': 'G5',
                        'end_note': 'C5'
                    },
                ],
                # `previous_movements`
                [],
                # `candidate_movements`
                [
                    [1, 1, -1],
                    [1, 1, 1],
                    [1, -2, -1],
                    [2, 1, -1],
                ],
                # `expected`
                [True, False, False, False]
            ),
            (
                # `tonic`
                'C',
                # `scale`
                'major',
                # `n_measures`
                6,
                # `max_skip`
                2,
                # `line_specifications`
                [
                    {
                        'lowest_note': 'G3',
                        'highest_note': 'G4',
                        'start_note': 'C4',
                        'end_note': 'C4'
                    },
                    {
                        'lowest_note': 'E4',
                        'highest_note': 'E5',
                        'start_note': 'C5',
                        'end_note': 'G4'
                    },
                    {
                        'lowest_note': 'G4',
                        'highest_note': 'G5',
                        'start_note': 'G5',
                        'end_note': 'C5'
                    },
                ],
                # `previous_movements`
                [
                    [2, -1, 0],
                    [1, -1, -1],
                ],
                # `candidate_movements`
                [
                    [-1, -1, -1],
                    [-1, -1, 1],
                    [-1, 2, -1],
                    [1, -1, -1],
                ],
                # `expected`
                [True, False, False, False]
            ),
        ]
    )
    def test_check_movements(
            self, tonic: str, scale: str, n_measures: int, max_skip: int,
            line_specifications: List[Dict[str, Any]],
            previous_movements: List[List[int]],
            candidate_movements: List[List[int]],
            expected: List[bool]
    ) -> None:
        """Test `check_movements` method."""
        piece = Piece(
            tonic, scale, n_measures, max_skip, line_specifications, {}
        )
        for movements in previous_movements:
            piece.add_measure(movements)
        result = [
            piece.check_movements(movements)
            for movements in candidate_movements
        ]
        assert result == expected

    @pytest.mark.parametrize(
        "tonic, scale, n_measures, max_skip, line_specifications, "
        "movements, expected_positions, expected_roll",
        [
            (
                # `tonic`
                'C',
                # `scale`
                'major',
                # `n_measures`
                10,
                # `max_skip`
                2,
                # `line_specifications`
                [
                    {
                        'lowest_note': 'G3',
                        'highest_note': 'G4',
                        'start_note': 'C4',
                        'end_note': 'C4'
                    },
                    {
                        'lowest_note': 'G4',
                        'highest_note': 'G5',
                        'start_note': 'G5',
                        'end_note': 'C5'
                    },
                ],
                # `movements`
                [
                    [1, -1],
                    [-1, -1],
                    [2, -2],
                    [2, -1],
                    [-2, -1],
                    [0, -1],
                    [-2, 1],
                    [1, 1],
                    [-1, 1]
                ],
                # `expected_positions`
                [
                    [3, 4, 3, 5, 7, 5, 5, 3, 4, 3],
                    [7, 6, 5, 3, 2, 1, 0, 1, 2, 3],
                ],
                # `expected_roll`
                np.array([
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ])
            ),
        ]
    )
    def test_add_measure(
            self, tonic: str, scale: str, n_measures: int, max_skip: int,
            line_specifications: List[Dict[str, Any]],
            movements: List[List[int]],
            expected_positions: List[int], expected_roll: np.ndarray
    ) -> None:
        """Test `add_measure` method."""
        piece = Piece(
            tonic, scale, n_measures, max_skip, line_specifications, {}
        )
        for mov in movements:
            piece.add_measure(mov)
        relative_positions = [
            [element.relative_position for element in line]
            for line in piece.lines
        ]
        assert relative_positions == expected_positions
        np.testing.assert_equal(piece.piano_roll, expected_roll)
