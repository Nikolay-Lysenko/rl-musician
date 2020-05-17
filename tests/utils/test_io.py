"""
Test `rlmusician.utils.io` module.

Author: Nikolay Lysenko
"""


from typing import Dict, List, Tuple

import pretty_midi
import pytest

from rlmusician.environment import Piece
from rlmusician.utils import (
    create_events_from_piece, create_midi_from_piece, create_wav_from_events
)


@pytest.mark.parametrize(
    "piece, all_steps, instrument_number, note_number, expected",
    [
        (
            # `piece`
            Piece(
                tonic='C',
                scale_type='major',
                cantus_firmus=['C4', 'D4', 'E4', 'D4', 'C4'],
                counterpoint_specifications={
                    'start_note': 'G4',
                    'end_note': 'C5',
                    'lowest_note': 'C4',
                    'highest_note': 'C6',
                    'start_pause_in_eighths': 4,
                    'max_skip_in_degrees': 2
                },
                rules={
                    'names': ['rearticulation_stability'],
                    'params': {}
                },
                rendering_params={}
            ),
            # `all_steps`,
            [(2, 4), [2, 8], [-1, 1]],
            # `instrument_number`
            1,
            # `note_number`
            3,
            # `expected`
            {'pitch': 72, 'start': 2.5, 'end': 2.625}
        ),
    ]
)
def test_create_midi_from_piece(
        path_to_tmp_file: str, piece: Piece, all_steps: List[Tuple[int, int]],
        instrument_number: int, note_number: int, expected: Dict[str, float]
) -> None:
    """Test `create_midi_from_piece` function."""
    for movement, duration in all_steps:
        piece.add_line_element(movement, duration)
    create_midi_from_piece(
        piece,
        path_to_tmp_file,
        measure_in_seconds=1,
        cantus_firmus_instrument=0,
        counterpoint_instrument=0,
        velocity=100
    )
    midi_data = pretty_midi.PrettyMIDI(path_to_tmp_file)
    instrument = midi_data.instruments[instrument_number]
    midi_note = instrument.notes[note_number]
    result = {
        'pitch': midi_note.pitch,
        'start': midi_note.start,
        'end': midi_note.end
    }
    assert result == expected


@pytest.mark.parametrize(
    "piece, all_steps, measure_in_seconds, volume, row_number, expected",
    [
        (
            # `piece`
            Piece(
                tonic='C',
                scale_type='major',
                cantus_firmus=['C4', 'D4', 'E4', 'D4', 'C4'],
                counterpoint_specifications={
                    'start_note': 'G4',
                    'end_note': 'C5',
                    'lowest_note': 'C4',
                    'highest_note': 'C6',
                    'start_pause_in_eighths': 4,
                    'max_skip_in_degrees': 2
                },
                rules={
                    'names': ['rearticulation_stability'],
                    'params': {}
                },
                rendering_params={}
            ),
            # `all_steps`,
            [(2, 4), [-2, 8], [0, 1]],
            # `measure_in_seconds`
            1,
            # `volume`
            0.2,
            # `row_number`
            2,
            # `expected`
            'default_timbre\t0.5\t0.5\tG4\t0.2\t0\t\n'
        ),
        (
            # `piece`
            Piece(
                tonic='C',
                scale_type='major',
                cantus_firmus=['C4', 'D4', 'E4', 'D4', 'C4'],
                counterpoint_specifications={
                    'start_note': 'G4',
                    'end_note': 'C5',
                    'lowest_note': 'C4',
                    'highest_note': 'C6',
                    'start_pause_in_eighths': 4,
                    'max_skip_in_degrees': 2
                },
                rules={
                    'names': ['rearticulation_stability'],
                    'params': {}
                },
                rendering_params={}
            ),
            # `all_steps`,
            [(2, 4), [2, 8], [-1, 1]],
            # `measure_in_seconds`
            1,
            # `volume`
            0.2,
            # `row_number`
            5,
            # `expected`
            'default_timbre\t1.5\t1.0\tD5\t0.2\t0\t\n'
        ),
    ]
)
def test_create_events_from_piece(
        path_to_tmp_file: str, piece: Piece, all_steps: List[Tuple[int, int]],
        measure_in_seconds: int, volume: float, row_number: int, expected: str
) -> None:
    """Test `create_events_from_piece` function."""
    for movement, duration in all_steps:
        piece.add_line_element(movement, duration)
    create_events_from_piece(
        piece,
        path_to_tmp_file,
        measure_in_seconds=measure_in_seconds,
        cantus_firmus_timbre='default_timbre',
        counterpoint_timbre='default_timbre',
        volume=volume
    )
    with open(path_to_tmp_file) as in_file:
        for i in range(row_number):
            in_file.readline()
        result = in_file.readline()
        assert result == expected


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
