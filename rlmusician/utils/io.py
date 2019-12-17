"""
Read from some formats and write to some formats.

Author: Nikolay Lysenko
"""


from pkg_resources import resource_filename

import numpy as np
import pypianoroll

from sinethesizer.io import (
    convert_tsv_to_timeline, create_timbres_registry, write_timeline_to_wav
)


def create_midi_from_piano_roll(
        roll: np.ndarray, midi_path: str, lowest_note: str, tempo: int,
        instrument: int, velocity: float
) -> None:
    """
    Create MIDI file from array with piano roll.

    :param roll:
        piano roll
    :param midi_path:
        path where resulting MIDI file is going to be saved
    :param lowest_note:
        note that corresponds to the lowest row of piano roll
    :param tempo:
        number of piano roll's time steps per minute
    :param instrument:
        ID (number) of instrument according to General MIDI specification
    :param velocity:
        one common velocity for all notes
    :return:
        None
    """
    notes_order = {
        'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6,
        'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
    }

    n_rows_below = (
        len(notes_order) * int(lowest_note[-1]) + notes_order[lowest_note[:-1]]
    )
    n_pypianoroll_pitches = 128
    n_rows_above = n_pypianoroll_pitches - n_rows_below - roll.shape[0]

    roll = np.hstack((roll, np.zeros((roll.shape[0], 1))))
    resized_roll = np.hstack((
        np.zeros((roll.shape[1], n_rows_below)),
        roll.T,
        np.zeros((roll.shape[1], n_rows_above))
    ))

    track = pypianoroll.Track(velocity * resized_roll, instrument)
    multitrack = pypianoroll.Multitrack(
        tracks=[track],
        tempo=tempo,
        beat_resolution=1
    )
    pypianoroll.write(multitrack, midi_path)


def create_wav_from_events(events_path: str, output_path: str) -> None:
    """
    Create WAV file.

    :param events_path:
        path to TSV file with track represented as `sinethesizer` events
    :param output_path:
        path where resulting WAV file is going to be saved
    :return:
        None
    """
    presets_path = resource_filename(
        'rlmusician',
        'configs/sinethesizer_presets.yml'
    )
    settings = {
        'frame_rate': 44100,
        'trailing_silence': 2,
        'max_channel_delay': 0.02,
        'timbres_registry': create_timbres_registry(presets_path)
    }
    timeline = convert_tsv_to_timeline(events_path, settings)
    write_timeline_to_wav(output_path, timeline, settings['frame_rate'])
