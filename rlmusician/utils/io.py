"""
Support reading and writing operations for some formats.

Author: Nikolay Lysenko
"""


from typing import List

import numpy as np
from sinethesizer.io.utils import get_list_of_notes


def find_involved_notes(lowest_note: str, n_notes: int) -> List[str]:
    """
    Find all involved consecutive notes.

    :param lowest_note:
        the lowest involved note
    :param n_notes:
        number of involved notes
    :return:
        requested number of consecutive notes starting from the given note
    """
    all_notes = get_list_of_notes()
    position = 0
    for position, note in enumerate(all_notes):
        if note == lowest_note:
            break
    involved_notes = all_notes[position:(position + n_notes)]
    return involved_notes


def convert_roll_to_events(
        roll: np.ndarray, roll_notes: List[str], timbre: str,
        step_in_seconds: float, volume: float,
        location: int = 0, effects: str = ''
) -> List[str]:
    """
    Convert piano roll to `sinethesizer` events.

    :param roll:
        piano roll to be converted
    :param roll_notes:
        notes that are corresponding to rows of piano roll
    :param timbre:
        timbre to be used by `sinethesizer`
    :param step_in_seconds:
        duration of one time step of piano roll in seconds
    :param volume:
        relative volume of sound to be played by `sinethesizer`
    :param location:
        position of imaginary sound source
    :param effects:
        sound effects to be applied to the resulting event
    :return:
        `sinethesizer` events
    """
    results = []
    for i in range(roll.shape[0]):
        frequency = roll_notes[i]
        start_time = None
        for j in range(roll.shape[1]):
            if j == 1 and start_time is None:
                start_time = j * step_in_seconds
            if j == 0 and start_time is not None:
                duration = j * step_in_seconds - start_time
                results.append((
                    timbre, start_time, duration, frequency, volume,
                    location, effects
                ))
                start_time = None
    results = sorted(results, key=lambda x: (x[1], x[3], x[2]))
    results = ['\t'.join([str(y) for y in x]) for x in results]
    return results


def write_to_sinethesizer_format(
        roll: np.ndarray, file_path: str, lowest_note: str, timbre: str,
        step_in_seconds: float, volume: float,
        location: int = 0, effects: str = ''
) -> None:
    """
    Convert piano roll to TSV file for `sinethesizer`.

    :param roll:
        piano roll to be converted
    :param file_path:
        path to a file where result is going to be saved
    :param lowest_note:
        note that corresponds to the lowest row of piano roll
    :param timbre:
        timbre to be used by `sinethesizer`
    :param step_in_seconds:
        duration of one time step of piano roll in seconds
    :param volume:
        relative volume of sound to be played by `sinethesizer`
    :param location:
        position of imaginary sound source
    :param effects:
        sound effects to be applied to the resulting event
    :return:
        None
    """
    roll_notes = find_involved_notes(lowest_note, roll.shape[0])
    events = convert_roll_to_events(
        roll, roll_notes, timbre, step_in_seconds, volume, location, effects
    )
    columns = [
        'timbre', 'start_time', 'duration', 'frequency', 'volume',
        'location', 'effects'
    ]
    header = '\t'.join(columns)
    results = [header] + events
    with open(file_path, 'w') as out_file:
        for line in results:
            out_file.write(line)
