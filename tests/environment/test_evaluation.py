"""
Test `rlmusician.environment.evaluation` module.

Author: Nikolay Lysenko
"""


from typing import List

import pytest

from rlmusician.environment.evaluation import (
    evaluate_absence_of_looped_pitches,
    evaluate_absence_of_looped_fragments,
    evaluate_entropy,
    evaluate_absence_of_pitch_class_clashes,
    evaluate_motion_by_types,
    evaluate_lines_correlation,
    evaluate_climax_explicity,
    evaluate_number_of_skips,
    evaluate_absence_of_downward_skips,
)
from rlmusician.environment.piece import Piece


@pytest.mark.parametrize(
    "piece, all_movements, max_n_repetitions, expected",
    [
        (
            # `piece`
            Piece(
                tonic='C',
                scale_type='major',
                n_measures=6,
                max_skip=2,
                line_specifications=[
                    {
                        'lowest_note': 'G3',
                        'highest_note': 'G4',
                        'start_note': 'C4',
                        'end_note': 'C4'
                    }
                ],
                voice_leading_rules={
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
                harmony_rules={
                    'names': [],
                    'params': {}
                },
                rendering_params={}
            ),
            # `all_movements`,
            [[2], [0], [0], [-2]],
            # `max_n_repetitions`
            2,
            # `expected`
            -1
        ),
        (
            # `piece`
            Piece(
                tonic='C',
                scale_type='major',
                n_measures=6,
                max_skip=2,
                line_specifications=[
                    {
                        'lowest_note': 'G3',
                        'highest_note': 'G4',
                        'start_note': 'C4',
                        'end_note': 'C4'
                    }
                ],
                voice_leading_rules={
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
                harmony_rules={
                    'names': [],
                    'params': {}
                },
                rendering_params={}
            ),
            # `all_movements`,
            [[2], [0], [0], [-2]],
            # `max_n_repetitions`
            3,
            # `expected`
            0
        ),
    ]
)
def test_evaluate_absence_of_looped_pitches(
        piece: Piece, all_movements: List[List[int]],
        max_n_repetitions: int, expected: float
) -> None:
    """Test `evaluate_absence_of_looped_pitches` function."""
    for movements in all_movements:
        piece.add_measure(movements)
    result = evaluate_absence_of_looped_pitches(piece, max_n_repetitions)
    assert result == expected


@pytest.mark.parametrize(
    "piece, all_movements, min_size, max_size, expected",
    [
        (
            # `piece`
            Piece(
                tonic='C',
                scale_type='major',
                n_measures=7,
                max_skip=2,
                line_specifications=[
                    {
                        'lowest_note': 'G3',
                        'highest_note': 'G4',
                        'start_note': 'C4',
                        'end_note': 'C4'
                    }
                ],
                voice_leading_rules={
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
                harmony_rules={
                    'names': [],
                    'params': {}
                },
                rendering_params={}
            ),
            # `all_movements`,
            [[1], [-1], [1], [-1], [1]],
            # `min_size`
            2,
            # `max_size`
            2,
            # `expected`
            -4
        ),
        (
            # `piece`
            Piece(
                tonic='C',
                scale_type='major',
                n_measures=9,
                max_skip=2,
                line_specifications=[
                    {
                        'lowest_note': 'G3',
                        'highest_note': 'G4',
                        'start_note': 'C4',
                        'end_note': 'C4'
                    }
                ],
                voice_leading_rules={
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
                harmony_rules={
                    'names': [],
                    'params': {}
                },
                rendering_params={}
            ),
            # `all_movements`,
            [[1], [-1], [1], [-1], [1], [-1], [1]],
            # `min_size`
            2,
            # `max_size`
            4,
            # `expected`
            -8
        ),
        (
            # `piece`
            Piece(
                tonic='C',
                scale_type='major',
                n_measures=9,
                max_skip=2,
                line_specifications=[
                    {
                        'lowest_note': 'G3',
                        'highest_note': 'G4',
                        'start_note': 'C4',
                        'end_note': 'C4'
                    }
                ],
                voice_leading_rules={
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
                harmony_rules={
                    'names': [],
                    'params': {}
                },
                rendering_params={}
            ),
            # `all_movements`,
            [[1], [1], [1], [1], [-1], [-1], [-1]],
            # `min_size`
            2,
            # `max_size`
            4,
            # `expected`
            0
        ),
    ]
)
def test_evaluate_absence_of_looped_fragments(
        piece: Piece, all_movements: List[List[int]],
        min_size: int, max_size: int, expected: float
) -> None:
    """Test `evaluate_absence_of_looped_fragments` function."""
    for movements in all_movements:
        piece.add_measure(movements)
    result = evaluate_absence_of_looped_fragments(piece, min_size, max_size)
    assert result == expected


@pytest.mark.parametrize(
    "piece, all_movements, expected",
    [
        (
            # `piece`
            Piece(
                tonic='C',
                scale_type='major',
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
                voice_leading_rules={
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
                harmony_rules={
                    'names': [],
                    'params': {}
                },
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
                scale_type='major',
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
                voice_leading_rules={
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
                harmony_rules={
                    'names': [],
                    'params': {}
                },
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
                scale_type='major',
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
                voice_leading_rules={
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
                harmony_rules={
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
                scale_type='major',
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
                voice_leading_rules={
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
                harmony_rules={
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
                scale_type='major',
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
                voice_leading_rules={
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
                harmony_rules={
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
            -0.1,
            # `contrary_coef`
            1,
            # `expected`
            -0.15
        ),
    ]
)
def test_evaluate_motion_by_types(
        piece: Piece, all_movements: List[List[int]], parallel_coef: float,
        similar_coef: float, oblique_coef: float, contrary_coef: float,
        expected: float
) -> None:
    """Test `evaluate_motion_by_types` function."""
    for movements in all_movements:
        piece.add_measure(movements)
    result = evaluate_motion_by_types(
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
                scale_type='major',
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
                voice_leading_rules={
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
                harmony_rules={
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
                scale_type='major',
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
                voice_leading_rules={
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
                harmony_rules={
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


@pytest.mark.parametrize(
    "piece, all_movements, shortage_penalty, duplication_penalty, expected",
    [
        (
            # `piece`
            Piece(
                tonic='C',
                scale_type='major',
                n_measures=5,
                max_skip=2,
                line_specifications=[
                    {
                        'lowest_note': 'G3',
                        'highest_note': 'G4',
                        'start_note': 'G4',
                        'end_note': 'C4'
                    },
                    {
                        'lowest_note': 'G4',
                        'highest_note': 'G5',
                        'start_note': 'C5',
                        'end_note': 'C5'
                    },
                ],
                voice_leading_rules={
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
                harmony_rules={
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
                rendering_params={}
            ),
            # `all_movements`,
            [
                [-1, 1],
                [-1, -1],
                [-1, 1],
            ],
            # `shortage_penalty`
            0.3,
            # `duplication_penalty`
            0.5,
            # `expected`
            0.3
        )
    ]
)
def test_evaluate_climax_explicity(
        piece: Piece, all_movements: List[List[int]],
        shortage_penalty: float, duplication_penalty: float, expected: float
) -> None:
    """Test `evaluate_climax_explicity` function."""
    for movements in all_movements:
        piece.add_measure(movements)
    result = evaluate_climax_explicity(
        piece, shortage_penalty, duplication_penalty
    )
    assert round(result, 4) == expected


@pytest.mark.parametrize(
    "piece, all_movements, min_n_skips, max_n_skips, expected",
    [
        (
            # `piece`
            Piece(
                tonic='C',
                scale_type='major',
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
                voice_leading_rules={
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
                harmony_rules={
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
                rendering_params={}
            ),
            # `all_movements`,
            [
                [2, 0],
                [0, 0],
                [2, 0],
            ],
            # `min_n_skips`
            1,
            # `max_n_skips`
            2,
            # `expected`
            0.5
        ),
    ]
)
def test_evaluate_number_of_skips(
        piece: Piece, all_movements: List[List[int]],
        min_n_skips: int, max_n_skips: int, expected: float
) -> None:
    """Test `evaluate_number_of_skips` function."""
    for movements in all_movements:
        piece.add_measure(movements)
    result = evaluate_number_of_skips(piece, min_n_skips, max_n_skips)
    assert result == expected


@pytest.mark.parametrize(
    "piece, all_movements, size_penalty_power, expected",
    [
        (
            # `piece`
            Piece(
                tonic='C',
                scale_type='major',
                n_measures=5,
                max_skip=2,
                line_specifications=[
                    {
                        'lowest_note': 'G3',
                        'highest_note': 'G4',
                        'start_note': 'G4',
                        'end_note': 'C4'
                    },
                    {
                        'lowest_note': 'G4',
                        'highest_note': 'G5',
                        'start_note': 'C5',
                        'end_note': 'C5'
                    },
                ],
                voice_leading_rules={
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
                harmony_rules={
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
                rendering_params={}
            ),
            # `all_movements`,
            [
                [-2, 0],
                [0, 0],
                [-2, 0],
            ],
            # `size_penalty_power`
            2,
            # `expected`
            -8
        ),
    ]
)
def test_evaluate_absence_of_downward_skips(
        piece: Piece, all_movements: List[List[int]],
        size_penalty_power: float, expected: float
) -> None:
    """Test `evaluate_absence_of_downward_skips` function."""
    for movements in all_movements:
        piece.add_measure(movements)
    result = evaluate_absence_of_downward_skips(piece)
    assert result == expected
