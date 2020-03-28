"""
Test `rlmusician.environment.piece` module.

Author: Nikolay Lysenko
"""


from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from rlmusician.environment import Piece


class TestPiece:
    """Tests for `Piece` class."""

    @pytest.mark.parametrize(
        "tonic, scale_type, n_measures, max_skip, line_specifications, "
        "voice_leading_rules, harmony_rules, match",
        [
            (
                # `tonic`
                'C',
                # `scale_type`
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
                # `voice_leading_rules`
                {
                    'names': [
                        'rearticulation',
                        'destination_of_skip',
                        'turn_after_skip',
                        'VI_VII_resolution',
                        'step_motion_to_end'
                    ],
                    'params': {
                        'turn_after_skip': {
                            'min_n_scale_degrees': 3
                        },
                        'step_motion_to_end': {
                            'prohibit_rearticulation': False
                        }
                    }
                },
                # `harmony_rules`
                {'names': [], 'params': {}},
                # `match`
                "No pitches from .*"
            ),
            (
                # `tonic`
                'C',
                # `scale_type`
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
                # `voice_leading_rules`
                {
                    'names': [
                        'rearticulation',
                        'destination_of_skip',
                        'turn_after_skip',
                        'VI_VII_resolution',
                        'step_motion_to_end'
                    ],
                    'params': {
                        'turn_after_skip': {
                            'min_n_scale_degrees': 3
                        },
                        'step_motion_to_end': {
                            'prohibit_rearticulation': False
                        }
                    }
                },
                # `harmony_rules`
                {'names': [], 'params': {}},
                # `match`
                ".* does not belong to .*"
            ),
            (
                # `tonic`
                'C',
                # `scale_type`
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
                # `voice_leading_rules`
                {
                    'names': [
                        'rearticulation',
                        'destination_of_skip',
                        'turn_after_skip',
                        'VI_VII_resolution',
                        'step_motion_to_end'
                    ],
                    'params': {
                        'turn_after_skip': {
                            'min_n_scale_degrees': 3
                        },
                        'step_motion_to_end': {
                            'prohibit_rearticulation': False
                        }
                    }
                },
                # `harmony_rules`
                {'names': [], 'params': {}},
                # `match`
                ".* is not a tonic triad member for .*"
            ),
        ]
    )
    def test_improper_initialization(
            self, tonic: str, scale_type: str, n_measures: int, max_skip: int,
            line_specifications: List[Dict[str, Any]],
            voice_leading_rules: Dict[str, Any], harmony_rules: Dict[str, Any],
            match: str
    ) -> None:
        """Test that initialization with invalid values is prohibited."""
        with pytest.raises(ValueError, match=match):
            _ = Piece(
                tonic, scale_type, n_measures, max_skip, line_specifications,
                voice_leading_rules, harmony_rules, rendering_params={}
            )

    @pytest.mark.parametrize(
        "tonic, scale_type, n_measures, max_skip, line_specifications, "
        "voice_leading_rules, harmony_rules, rng, roll",
        [
            (
                # `tonic`
                'D',
                # `scale_type`
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
                # `voice_leading_rules`
                {
                    'names': [
                        'rearticulation',
                        'destination_of_skip',
                        'turn_after_skip',
                        'VI_VII_resolution',
                        'step_motion_to_end'
                    ],
                    'params': {
                        'turn_after_skip': {
                            'min_n_scale_degrees': 3
                        },
                        'step_motion_to_end': {
                            'prohibit_rearticulation': False
                        }
                    }
                },
                # `harmony_rules`
                {
                    'names': [
                        'consonance',
                        'absence_of_large_intervals'
                    ],
                    'params': {
                        'absence_of_large_intervals': {
                            'max_n_semitones': 16
                        }
                    }
                },
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
            self, tonic: str, scale_type: str, n_measures: int, max_skip: int,
            line_specifications: List[Dict[str, Any]],
            voice_leading_rules: Dict[str, Any], harmony_rules: Dict[str, Any],
            rng: Tuple[int, int], roll: np.ndarray
    ) -> None:
        """Test that initialization with valid values works as expected."""
        piece = Piece(
            tonic, scale_type, n_measures, max_skip, line_specifications,
            voice_leading_rules, harmony_rules, rendering_params={}
        )
        assert len(piece.lines) == len(line_specifications)
        assert (piece.lowest_row_to_show, piece.highest_row_to_show) == rng
        np.testing.assert_equal(piece.piano_roll, roll)

    @pytest.mark.parametrize(
        "tonic, scale_type, n_measures, max_skip, line_specifications, "
        "voice_leading_rules, harmony_rules, "
        "previous_movements, candidate_movements, expected",
        [
            (
                # `tonic`
                'C',
                # `scale_type`
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
                # `voice_leading_rules`
                {
                    'names': [
                        'rearticulation',
                        'destination_of_skip',
                        'turn_after_skip',
                        'VI_VII_resolution',
                        'step_motion_to_end'
                    ],
                    'params': {
                        'turn_after_skip': {
                            'min_n_scale_degrees': 3
                        },
                        'step_motion_to_end': {
                            'prohibit_rearticulation': False
                        }
                    }
                },
                # `harmony_rules`
                {
                    'names': [
                        'consonance',
                        'absence_of_large_intervals'
                    ],
                    'params': {
                        'absence_of_large_intervals': {
                            'max_n_semitones': 16
                        }
                    }
                },
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
                # `scale_type`
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
                # `voice_leading_rules`
                {
                    'names': [
                        'rearticulation',
                        'destination_of_skip',
                        'turn_after_skip',
                        'VI_VII_resolution',
                        'step_motion_to_end'
                    ],
                    'params': {
                        'turn_after_skip': {
                            'min_n_scale_degrees': 3
                        },
                        'step_motion_to_end': {
                            'prohibit_rearticulation': False
                        }
                    }
                },
                # `harmony_rules`
                {
                    'names': [
                        'consonance',
                        'absence_of_large_intervals'
                    ],
                    'params': {
                        'absence_of_large_intervals': {
                            'max_n_semitones': 16
                        }
                    }
                },
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
            (
                # `tonic`
                'C',
                # `scale_type`
                'major',
                # `n_measures`
                10,
                # `max_skip`
                5,
                # `line_specifications`
                [
                    {
                        'lowest_note': 'G3',
                        'highest_note': 'G5',
                        'start_note': 'G4',
                        'end_note': 'C4'
                    },
                    {
                        'lowest_note': 'C4',
                        'highest_note': 'C6',
                        'start_note': 'C5',
                        'end_note': 'G4'
                    },
                ],
                # `voice_leading_rules`
                {
                    'names': [
                        'rearticulation',
                        'destination_of_skip',
                        'turn_after_skip',
                        'VI_VII_resolution',
                        'step_motion_to_end'
                    ],
                    'params': {
                        'turn_after_skip': {
                            'min_n_scale_degrees': 3
                        },
                        'step_motion_to_end': {
                            'prohibit_rearticulation': False
                        }
                    }
                },
                # `harmony_rules`
                {
                    'names': [
                        'consonance',
                        'absence_of_large_intervals'
                    ],
                    'params': {
                        'absence_of_large_intervals': {
                            'max_n_semitones': 16
                        }
                    }
                },
                # `previous_movements`
                [[0, 4]],
                # `candidate_movements`
                [
                    [-1, -1],
                    [1, -2],
                    [1, 1],
                    [0, -1],
                ],
                # `expected`
                [True, False, False, False]
            ),
        ]
    )
    def test_check_movements(
            self, tonic: str, scale_type: str, n_measures: int, max_skip: int,
            line_specifications: List[Dict[str, Any]],
            voice_leading_rules: Dict[str, Any], harmony_rules: Dict[str, Any],
            previous_movements: List[List[int]],
            candidate_movements: List[List[int]],
            expected: List[bool]
    ) -> None:
        """Test `check_movements` method."""
        piece = Piece(
            tonic, scale_type, n_measures, max_skip, line_specifications,
            voice_leading_rules, harmony_rules, rendering_params={}
        )
        for movements in previous_movements:
            piece.add_measure(movements)
        result = [
            piece.check_movements(movements)
            for movements in candidate_movements
        ]
        assert result == expected

    @pytest.mark.parametrize(
        "tonic, scale_type, n_measures, max_skip, line_specifications, "
        "voice_leading_rules, harmony_rules, "
        "movements, expected_positions, expected_roll",
        [
            (
                # `tonic`
                'C',
                # `scale_type`
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
                # `voice_leading_rules`
                {
                    'names': [
                        'rearticulation',
                        'destination_of_skip',
                        'turn_after_skip',
                        'VI_VII_resolution',
                        'step_motion_to_end'
                    ],
                    'params': {
                        'turn_after_skip': {
                            'min_n_scale_degrees': 3
                        },
                        'step_motion_to_end': {
                            'prohibit_rearticulation': False
                        }
                    }
                },
                # `harmony_rules`
                {
                    'names': [
                        'consonance',
                        'absence_of_large_intervals'
                    ],
                    'params': {
                        'absence_of_large_intervals': {
                            'max_n_semitones': 16
                        }
                    }
                },
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
            self, tonic: str, scale_type: str, n_measures: int, max_skip: int,
            line_specifications: List[Dict[str, Any]],
            voice_leading_rules: Dict[str, Any], harmony_rules: Dict[str, Any],
            movements: List[List[int]],
            expected_positions: List[int], expected_roll: np.ndarray
    ) -> None:
        """Test `add_measure` method."""
        piece = Piece(
            tonic, scale_type, n_measures, max_skip, line_specifications,
            voice_leading_rules, harmony_rules, rendering_params={}
        )
        for movement in movements:
            piece.add_measure(movement)
        relative_positions = [
            [element.relative_position for element in line]
            for line in piece.lines
        ]
        assert relative_positions == expected_positions
        np.testing.assert_equal(piece.piano_roll, expected_roll)

    @pytest.mark.parametrize(
        "tonic, scale_type, n_measures, max_skip, line_specifications, "
        "voice_leading_rules, harmony_rules, "
        "movements, expected_passed_movements, expected_roll",
        [
            (
                # `tonic`
                'C',
                # `scale_type`
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
                # `voice_leading_rules`
                {
                    'names': [
                        'rearticulation',
                        'destination_of_skip',
                        'turn_after_skip',
                        'VI_VII_resolution',
                        'step_motion_to_end'
                    ],
                    'params': {
                        'turn_after_skip': {
                            'min_n_scale_degrees': 3
                        },
                        'step_motion_to_end': {
                            'prohibit_rearticulation': False
                        }
                    }
                },
                # `harmony_rules`
                {
                    'names': [
                        'consonance',
                        'absence_of_large_intervals'
                    ],
                    'params': {
                        'absence_of_large_intervals': {
                            'max_n_semitones': 16
                        }
                    }
                },
                # `movements`
                [
                    [1, -1],
                    [-1, -1],
                    [2, -2],
                    [2, -1],
                    [-2, -1],
                    [0, -1],
                    [-2, 1],
                ],
                # `expected_passed_movements`
                [[], []],
                # `expected_roll`
                np.array([
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ])
            ),
        ]
    )
    def test_reset(
            self, tonic: str, scale_type: str, n_measures: int, max_skip: int,
            line_specifications: List[Dict[str, Any]],
            voice_leading_rules: Dict[str, Any], harmony_rules: Dict[str, Any],
            movements: List[List[int]],
            expected_passed_movements: List[int], expected_roll: np.ndarray
    ) -> None:
        """Test `reset` method."""
        piece = Piece(
            tonic, scale_type, n_measures, max_skip, line_specifications,
            voice_leading_rules, harmony_rules, rendering_params={}
        )
        for movement in movements:
            piece.add_measure(movement)
        piece.reset()
        assert piece.passed_movements == expected_passed_movements
        np.testing.assert_equal(piece.piano_roll, expected_roll)

