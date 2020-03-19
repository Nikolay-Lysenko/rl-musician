"""
Test `rlmusician.utils.io` module.

Author: Nikolay Lysenko
"""


from typing import List

import pytest

from rlmusician.environment import Piece
from rlmusician.utils import (
    create_events_from_piece, create_midi_from_piece, create_wav_from_events
)


@pytest.mark.parametrize(
    "piece, all_movements, measure_in_seconds, volume, expected",
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
            # `measure_in_seconds`
            1,
            # `volume`
            0.2,
            # `expected`
            'default_timbre\t0\t1\tC5\t0.2\t0\t\n'
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
            [[0], [0]],
            # `measure_in_seconds`
            1,
            # `volume`
            0.2,
            # `expected`
            'default_timbre\t0\t1\tC5\t0.2\t0\t\n'
        ),
    ]
)
def test_create_events_from_piece(
        path_to_tmp_file: str, piece: Piece, all_movements: List[List[int]],
        measure_in_seconds: int, volume: float, expected: str
) -> None:
    """Test `create_events_from_piece` function."""
    for movements in all_movements:
        piece.add_measure(movements)
    create_events_from_piece(
        piece,
        path_to_tmp_file,
        measure_in_seconds=measure_in_seconds,
        timbre='default_timbre',
        volume=volume
    )
    with open(path_to_tmp_file) as in_file:
        header = in_file.readline()
        result = in_file.readline()
        assert result == expected


@pytest.mark.parametrize(
    "piece, all_movements",
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
            [[0], [0], [0]]
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
            [[0], [0]]
        ),
    ]
)
def test_create_midi_from_piece(
        path_to_tmp_file: str, piece: Piece, all_movements: List[List[int]]
) -> None:
    """Test `create_midi_from_piece` function."""
    for movements in all_movements:
        piece.add_measure(movements)
    create_midi_from_piece(
        piece,
        path_to_tmp_file,
        measure_in_seconds=1,
        instrument=0,
        velocity=100
    )


@pytest.mark.parametrize(
    "tsv_content",
    [
        (
            [
                "timbre\tstart_time\tduration\tfrequency\tvolume\tlocation\teffects",
                "default_timbre\t1\t1\tA0\t1\t0\t",
                'default_timbre\t2\t1\t1\t1\t0\t[{"name": "tremolo", "frequency": 1}]'
            ]
        )
    ]
)
def test_create_wav_from_events(
        path_to_tmp_file: str, path_to_another_tmp_file: str,
        tsv_content: List[str]
) -> None:
    """Test `create_wav_from_events` function."""
    with open(path_to_tmp_file, 'w') as tmp_tsv_file:
        for line in tsv_content:
            tmp_tsv_file.write(line + '\n')
    create_wav_from_events(path_to_tmp_file, path_to_another_tmp_file)
