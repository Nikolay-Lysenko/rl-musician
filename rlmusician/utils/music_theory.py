"""
Help to work with notions from music theory.

Author: Nikolay Lysenko
"""


from typing import List, NamedTuple

from sinethesizer.io.utils import get_note_to_position_mapping


NOTE_TO_POSITION = get_note_to_position_mapping()


class ScaleElement(NamedTuple):
    """A pitch from a scale."""

    absolute_position: int
    degree: int


def create_scale(tonic: str, scale_type: str) -> List[ScaleElement]:
    """
    Create scale.

    :param tonic:
        tonic pitch class represented by letter (like C or A#)
    :param scale_type:
        type of scale (currently, 'major', 'natural_minor', and
        'harmonic_minor' are supported)
    :return:
        scale
    """
    patterns = {
        'major': [1, 0, 2, 0, 3, 4, 0, 5, 0, 6, 0, 7],
        'natural_minor': [1, 0, 2, 3, 0, 4, 0, 5, 6, 0, 7, 0],
        'harmonic_minor': [1, 0, 2, 3, 0, 4, 0, 5, 6, 0, 0, 7],
    }
    pattern = patterns[scale_type]
    tonic_position = NOTE_TO_POSITION[tonic + '1']
    scale = []
    for x in range(len(NOTE_TO_POSITION)):
        degree = pattern[(x - tonic_position) % len(pattern)]
        if degree > 0:
            scale.append(ScaleElement(absolute_position=x, degree=degree))
    return scale


def slice_scale(
        scale: List[ScaleElement], lowest_note: str, highest_note: str
) -> List[ScaleElement]:
    """
    Keep only pitches that are within a specified range.

    :param scale:
        list of pitches
    :param lowest_note:
        lowest note (inclusively) represented like C3 or A#4
    :param highest_note:
        highest note (inclusively) represented like C3 or A#4
    :return:
        list of pitches from a specified range
    """
    lowest_position = NOTE_TO_POSITION[lowest_note]
    highest_position = NOTE_TO_POSITION[highest_note]
    sliced_scale = [
        x for x in scale
        if lowest_position <= x.absolute_position <= highest_position
    ]
    return sliced_scale
