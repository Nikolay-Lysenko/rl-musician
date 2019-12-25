"""
Help to work with notions from music theory.

Author: Nikolay Lysenko
"""


from typing import List

from sinethesizer.io.utils import get_note_to_position_mapping


NOTE_TO_POSITION = get_note_to_position_mapping()


def get_positions_by_relation_to_tonic(
        tonic: str, pattern: List[bool]
) -> List[int]:
    """
    Return positions of all pitches that are in a particular relation to tonic.

    :param tonic:
        tonic pitch class represented by letter (like C or A#)
    :param pattern:
        12-element list containing flags whether a corresponding pitch class
        (starting from tonic) must be selected
    :return:
        sorted list of selected positions
    """
    tonic_position = NOTE_TO_POSITION[tonic + '1']
    selected_positions = [
        x for x in range(len(NOTE_TO_POSITION))
        if pattern[(x - tonic_position) % len(pattern)]
    ]
    return selected_positions


def get_positions_from_scale(tonic: str, scale: str) -> List[int]:
    """
    Return positions of all pitches that belong to a specified scale.

    :param tonic:
        tonic pitch class represented by letter (like C or A#)
    :param scale:
        scale (currently, 'major', 'natural_minor', and 'harmonic_minor' are
        supported)
    :return:
        sorted list of all pitch positions from the specified scale
    """
    patterns = {
        'major': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        'natural_minor': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
        'harmonic_minor': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1],
    }
    pattern = patterns[scale]
    positions_from_scale = get_positions_by_relation_to_tonic(tonic, pattern)
    return positions_from_scale


def get_tonic_triad_positions(tonic: str, scale: str) -> List[int]:
    """
    Return positions of all pitches from tonic triad pitch classes.

    :param tonic:
        tonic pitch class represented by letter (like C or A#)
    :param scale:
        scale (currently, 'major', 'natural_minor', and 'harmonic_minor' are
        supported)
    :return:
        sorted list of positions from tonic triad
    """
    patterns = {
        'major': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        'natural_minor': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        'harmonic_minor': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    }
    pattern = patterns[scale]
    tonic_triad_positions = get_positions_by_relation_to_tonic(tonic, pattern)
    return tonic_triad_positions


def slice_positions(
        positions: List[int], lowest_note: str, highest_note: str,
) -> List[int]:
    """
    Keep only positions that are within a specified range.

    :param positions:
        list of pitch positions
    :param lowest_note:
        lowest note (inclusively) represented like C3 or A#4
    :param highest_note:
        highest note (inclusively) represented like C3 or A#4
    :return:
        list of positions from a specified range
    """
    lowest_position = NOTE_TO_POSITION[lowest_note]
    highest_position = NOTE_TO_POSITION[highest_note]
    sliced_positions = [
        x for x in positions
        if lowest_position <= x <= highest_position
    ]
    return sliced_positions
