"""
Test `rlmusician.environment.piece` module.

Author: Nikolay Lysenko
"""


from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from rlmusician.environment import Piece
from rlmusician.environment.piece import LineElement
from rlmusician.utils import ScaleElement


class TestPiece:
    """Tests for `Piece` class."""

    @pytest.mark.parametrize(
        "tonic, scale_type, cantus_firmus, counterpoint_specifications, "
        "rules, match",
        [
            (
                # `tonic`
                'C',
                # `scale_type`
                'major',
                # `cantus_firmus`
                ['C4', 'D4', 'E4', 'D4', 'C4'],
                # `counterpoint_specifications`
                {
                    'start_note': 'E4',
                    'end_note': 'E4',
                    'lowest_note': 'G4',
                    'highest_note': 'G3',
                    'start_pause_in_eighths': 4,
                    'max_skip_in_degrees': 2,
                },
                # `rules`
                {
                    'names': ['rearticulation_stability'],
                    'params': {}
                },
                # `match`
                "Lowest note and highest note are in wrong order: "
            ),
            (
                # `tonic`
                'C',
                # `scale_type`
                'major',
                # `cantus_firmus`
                ['D4', 'D4', 'E4', 'D4', 'C4'],
                # `counterpoint_specifications`
                {
                    'start_note': 'E4',
                    'end_note': 'E4',
                    'lowest_note': 'G3',
                    'highest_note': 'G4',
                    'start_pause_in_eighths': 4,
                    'max_skip_in_degrees': 2,
                },
                # `rules`
                {
                    'names': ['rearticulation_stability'],
                    'params': {}
                },
                # `match`
                "cantus firmus can not start with it"
            ),
            (
                # `tonic`
                'C',
                # `scale_type`
                'major',
                # `cantus_firmus`
                ['C4', 'D4', 'E4', 'D4', 'C4'],
                # `counterpoint_specifications`
                {
                    'start_note': 'E4',
                    'end_note': 'D4',
                    'lowest_note': 'G3',
                    'highest_note': 'G4',
                    'start_pause_in_eighths': 4,
                    'max_skip_in_degrees': 2,
                },
                # `rules`
                {
                    'names': ['rearticulation_stability'],
                    'params': {}
                },
                # `match`
                "counterpoint line can not end with it"
            ),
            (
                # `tonic`
                'C',
                # `scale_type`
                'major',
                # `cantus_firmus`
                ['C4', 'D#4', 'E4', 'D4', 'C4'],
                # `counterpoint_specifications`
                {
                    'start_note': 'E4',
                    'end_note': 'E4',
                    'lowest_note': 'G3',
                    'highest_note': 'G4',
                    'start_pause_in_eighths': 4,
                    'max_skip_in_degrees': 2,
                },
                # `rules`
                {
                    'names': ['rearticulation_stability'],
                    'params': {}
                },
                # `match`
                "is not from"
            ),
            (
                # `tonic`
                'C',
                # `scale_type`
                'major',
                # `cantus_firmus`
                ['C4', 'D4', 'E4', 'D4', 'C4'],
                # `counterpoint_specifications`
                {
                    'start_note': 'E4',
                    'end_note': 'E#4',
                    'lowest_note': 'G3',
                    'highest_note': 'G4',
                    'start_pause_in_eighths': 4,
                    'max_skip_in_degrees': 2,
                },
                # `rules`
                {
                    'names': ['rearticulation_stability'],
                    'params': {}
                },
                # `match`
                "is not from"
            ),
        ]
    )
    def test_improper_initialization(
            self, tonic: str, scale_type: str, cantus_firmus: List[str],
            counterpoint_specifications: Dict[str, Any], rules: Dict[str, Any],
            match: str
    ) -> None:
        """Test that initialization with invalid values is prohibited."""
        with pytest.raises(ValueError, match=match):
            _ = Piece(
                tonic, scale_type, cantus_firmus, counterpoint_specifications,
                rules, rendering_params={}
            )

    @pytest.mark.parametrize(
        "tonic, scale_type, cantus_firmus, counterpoint_specifications, "
        "rules, rng, roll",
        [
            (
                # `tonic`
                'D',
                # `scale_type`
                'major',
                # `cantus_firmus`,
                ['A3', 'G3', 'F#3', 'B3', 'A3'],
                # `counterpoint_specifications`
                {
                    'lowest_note': 'G3',
                    'highest_note': 'G4',
                    'start_note': 'F#4',
                    'end_note': 'D4',
                    'start_pause_in_eighths': 4,
                    'max_skip_in_degrees': 2,
                },
                # `rules`
                {
                    'names': ['rearticulation_stability'],
                    'params': {}
                },
                # `rng`,
                (33, 46),
                # `roll`
                np.array([
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ])
            ),
        ]
    )
    def test_initialization(
            self, tonic: str, scale_type: str, cantus_firmus: List[str],
            counterpoint_specifications: Dict[str, Any], rules: Dict[str, Any],
            rng: Tuple[int, int], roll: np.ndarray
    ) -> None:
        """Test that initialization with valid values works as expected."""
        piece = Piece(
            tonic, scale_type, cantus_firmus, counterpoint_specifications,
            rules, rendering_params={}
        )
        assert piece.current_measure_durations == []
        assert (piece.lowest_row_to_show, piece.highest_row_to_show) == rng
        np.testing.assert_equal(piece.piano_roll, roll)

    @pytest.mark.parametrize(
        "tonic, scale_type, cantus_firmus, counterpoint_specifications, "
        "rules, previous_steps, candidate_steps, expected",
        [
            (
                # `tonic`
                'C',
                # `scale_type`
                'major',
                # `cantus_firmus`
                ['C4', 'D4', 'E4', 'D4', 'C4'],
                # `counterpoint_specifications`
                {
                    'start_note': 'E4',
                    'end_note': 'E4',
                    'lowest_note': 'G3',
                    'highest_note': 'G4',
                    'start_pause_in_eighths': 4,
                    'max_skip_in_degrees': 2,
                },
                # `rules`
                {
                    'names': ['rearticulation_stability'],
                    'params': {}
                },
                # `previous_steps`
                [(1, 4), (-1, 4), (1, 4), (-1, 4), (1, 4)],
                # `candidate_steps`
                [(-3, 4), (2, 4), (-1, 8), (0, 4), (-1, 4)],
                # `expected`
                [False, False, False, False, True]
            ),
            (
                # `tonic`
                'C',
                # `scale_type`
                'major',
                # `cantus_firmus`
                ['C4', 'D4', 'D4', 'D4', 'C4'],
                # `counterpoint_specifications`
                {
                    'start_note': 'E4',
                    'end_note': 'E4',
                    'lowest_note': 'D4',
                    'highest_note': 'G4',
                    'start_pause_in_eighths': 4,
                    'max_skip_in_degrees': 2,
                },
                # `rules`
                {
                    'names': ['consonance_on_strong_beat'],
                    'params': {}
                },
                # `previous_steps`
                [(-1, 4), (0, 4)],
                # `candidate_steps`
                [(3, 2), (-2, 4), (0, 18), (1, 4), (0, 4)],
                # `expected`
                [False, False, False, False, True]
            ),
        ]
    )
    def test_check_validity(
            self, tonic: str, scale_type: str, cantus_firmus: List[str],
            counterpoint_specifications: Dict[str, Any], rules: Dict[str, Any],
            previous_steps: List[Tuple[int, int]],
            candidate_steps: List[Tuple[int, int]],
            expected: List[bool]
    ) -> None:
        """Test `check_validity` method."""
        piece = Piece(
            tonic, scale_type, cantus_firmus, counterpoint_specifications,
            rules, rendering_params={}
        )
        for movement, duration in previous_steps:
            piece.add_line_element(movement, duration)
        result = [
            piece.check_validity(movement, duration)
            for movement, duration in candidate_steps
        ]
        assert result == expected

    @pytest.mark.parametrize(
        "tonic, scale_type, cantus_firmus, counterpoint_specifications, "
        "rules, steps, expected_positions, "
        "expected_current_measure_durations, expected_current_motion_start, "
        "expected_is_last_element_consonant, expected_roll",
        [
            (
                # `tonic`
                'C',
                # `scale_type`
                'major',
                # `cantus_firmus`
                ['C4', 'D4', 'E4', 'D4', 'C4'],
                # `counterpoint_specifications`
                {
                    'start_note': 'E4',
                    'end_note': 'E4',
                    'lowest_note': 'G3',
                    'highest_note': 'G4',
                    'start_pause_in_eighths': 4,
                    'max_skip_in_degrees': 2,
                },
                # `rules`
                {
                    'names': ['rearticulation_stability'],
                    'params': {}
                },
                # `steps`
                [(-2, 4), (-2, 4)],
                # `expected_positions`
                [43, 39, 36],
                # `expected_current_measure_durations`
                [],
                # `expected_current_motion_start`
                LineElement(ScaleElement('E4', 43, 25, 3, True), 4, 8),
                # `expected_is_last_element_consonant`
                False,
                # `expected_roll`
                np.array([
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ])
            ),
            (
                # `tonic`
                'C',
                # `scale_type`
                'major',
                # `cantus_firmus`
                ['C4', 'D4', 'E4', 'D4', 'C4'],
                # `counterpoint_specifications`
                {
                    'start_note': 'E4',
                    'end_note': 'E4',
                    'lowest_note': 'G3',
                    'highest_note': 'G4',
                    'start_pause_in_eighths': 4,
                    'max_skip_in_degrees': 2,
                },
                # `rules`
                {
                    'names': ['rearticulation_stability'],
                    'params': {}
                },
                # `steps`
                [(-2, 4), (-2, 4), (-1, 4)],
                # `expected_positions`
                [43, 39, 36, 34],
                # `expected_current_measure_durations`
                [4],
                # `expected_current_motion_start`
                LineElement(ScaleElement('E4', 43, 25, 3, True), 4, 8),
                # `expected_is_last_element_consonant`
                True,
                # `expected_roll`
                np.array([
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ])
            ),
            (
                # `tonic`
                'C',
                # `scale_type`
                'major',
                # `cantus_firmus`
                ['C4', 'D4', 'E4', 'D4', 'C4'],
                # `counterpoint_specifications`
                {
                    'start_note': 'E4',
                    'end_note': 'E4',
                    'lowest_note': 'G3',
                    'highest_note': 'G4',
                    'start_pause_in_eighths': 4,
                    'max_skip_in_degrees': 2,
                },
                # `rules`
                {
                    'names': ['rearticulation_stability'],
                    'params': {}
                },
                # `steps`
                [(-2, 4), (-2, 4), (-1, 4), (2, 8), (1, 2), (0, 1)],
                # `expected_positions`
                [43, 39, 36, 34, 38, 39, 39],
                # `expected_current_measure_durations`
                [4, 2, 1],
                # `expected_current_motion_start`
                LineElement(ScaleElement('G3', 34, 20, 5, True), 16, 20),
                # `expected_is_last_element_consonant`
                False,
                # `expected_roll`
                np.array([
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ])
            ),
            (
                # `tonic`
                'C',
                # `scale_type`
                'major',
                # `cantus_firmus`
                ['C4', 'D4', 'E4', 'D4', 'C4'],
                # `counterpoint_specifications`
                {
                    'start_note': 'E4',
                    'end_note': 'E4',
                    'lowest_note': 'G3',
                    'highest_note': 'G4',
                    'start_pause_in_eighths': 4,
                    'max_skip_in_degrees': 2,
                },
                # `rules`
                {
                    'names': ['rearticulation_stability'],
                    'params': {}
                },
                # `steps`
                [(1, 4), (-2, 8)],
                # `expected_positions`
                [43, 44, 41],
                # `expected_current_measure_durations`
                [4],
                # `expected_current_motion_start`
                LineElement(ScaleElement('F4', 44, 26, 4, False), 8, 12),
                # `expected_is_last_element_consonant`
                False,
                # `expected_roll`
                np.array([
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ])
            ),
        ]
    )
    def test_add_line_element(
            self, tonic: str, scale_type: str, cantus_firmus: List[str],
            counterpoint_specifications: Dict[str, Any], rules: Dict[str, Any],
            steps: List[Tuple[int, int]], expected_positions: List[int],
            expected_current_measure_durations: List[int],
            expected_current_motion_start: LineElement,
            expected_is_last_element_consonant: bool, expected_roll: np.ndarray
    ) -> None:
        """Test `add_line_element` method."""
        piece = Piece(
            tonic, scale_type, cantus_firmus, counterpoint_specifications,
            rules, rendering_params={}
        )
        for movement, duration in steps:
            piece.add_line_element(movement, duration)
        positions = [
            x.scale_element.position_in_semitones for x in piece.counterpoint
        ]
        assert positions == expected_positions
        assert piece.current_measure_durations == expected_current_measure_durations
        assert piece.current_motion_start_element == expected_current_motion_start
        assert piece.is_last_element_consonant == expected_is_last_element_consonant
        np.testing.assert_equal(piece.piano_roll, expected_roll)

    @pytest.mark.parametrize(
        "tonic, scale_type, cantus_firmus, counterpoint_specifications, "
        "rules, steps, expected_roll",
        [
            (
                # `tonic`
                'C',
                # `scale_type`
                'major',
                # `cantus_firmus`
                ['C4', 'D4', 'E4', 'D4', 'C4'],
                # `counterpoint_specifications`
                {
                    'start_note': 'E4',
                    'end_note': 'E4',
                    'lowest_note': 'G3',
                    'highest_note': 'G4',
                    'start_pause_in_eighths': 4,
                    'max_skip_in_degrees': 2,
                },
                # `rules`
                {
                    'names': ['rearticulation_stability'],
                    'params': {}
                },
                # `steps`
                [(-2, 4), (-2, 4), (-1, 4), (2, 8), (1, 2), (0, 1)],
                # `expected_roll`
                np.array([
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ])
            ),
        ]
    )
    def test_reset(
            self, tonic: str, scale_type: str, cantus_firmus: List[str],
            counterpoint_specifications: Dict[str, Any], rules: Dict[str, Any],
            steps: List[Tuple[int, int]], expected_roll: np.ndarray
    ) -> None:
        """Test `reset` method."""
        piece = Piece(
            tonic, scale_type, cantus_firmus, counterpoint_specifications,
            rules, rendering_params={}
        )
        for movement, duration in steps:
            piece.add_line_element(movement, duration)
        piece.reset()
        assert piece.past_movements == []
        assert piece.current_time_in_eighths == 8
        np.testing.assert_equal(piece.piano_roll, expected_roll)
