"""
Help to work with notions from music theory.

Author: Nikolay Lysenko
"""


from typing import List, NamedTuple

from sinethesizer.io.utils import get_note_to_position_mapping


NOTE_TO_POSITION = get_note_to_position_mapping()
TONIC_TRIAD_DEGREES = (1, 3, 5)


class ScaleElement(NamedTuple):
    """A pitch from a diatonic scale."""

    note: str
    position_in_semitones: int
    position_in_degrees: int
    degree: int
    is_from_tonic_triad: bool


class Scale:
    """A diatonic scale."""

    def __init__(self, tonic: str, scale_type: str):
        """
        Initialize an instance.

        :param tonic:
            tonic pitch class represented by letter (like C or A#)
        :param scale_type:
            type of scale (currently, 'major', 'natural_minor', and
            'harmonic_minor' are supported)
        """
        self.tonic = tonic
        self.scale_type = scale_type

        self.elements = self.__create_elements()
        self.note_to_element = {
            element.note: element for element in self.elements
        }
        self.position_in_semitones_to_element = {
            element.position_in_semitones: element for element in self.elements
        }
        self.position_in_degrees_to_element = {
            element.position_in_degrees: element for element in self.elements
        }

    def __create_elements(self) -> List[ScaleElement]:
        """Create sorted list of scale elements."""
        patterns = {
            'major': [
                1, None, 2, None, 3, 4, None, 5, None, 6, None, 7
            ],
            'natural_minor': [
                1, None, 2, 3, None, 4, None, 5, 6, None, 7, None
            ],
            'harmonic_minor': [
                1, None, 2, 3, None, 4, None, 5, 6, None, None, 7
            ],
        }
        pattern = patterns[self.scale_type]
        tonic_position = NOTE_TO_POSITION[self.tonic + '1']
        elements = []
        position_in_degrees = 0
        for note, position_in_semitones in NOTE_TO_POSITION.items():
            remainder = (position_in_semitones - tonic_position) % len(pattern)
            degree = pattern[remainder]
            if degree is not None:
                element = ScaleElement(
                    note=note,
                    position_in_semitones=position_in_semitones,
                    position_in_degrees=position_in_degrees,
                    degree=degree,
                    is_from_tonic_triad=(degree in TONIC_TRIAD_DEGREES)
                )
                elements.append(element)
                position_in_degrees += 1
        return elements

    def get_element_by_note(self, note: str) -> ScaleElement:
        """Get scale element by its note (like 'C4' or 'A#5')."""
        try:
            return self.note_to_element[note]
        except KeyError:
            raise ValueError(
                f"Note {note} is not from {self.tonic}-{self.scale_type}."
            )

    def get_element_by_position_in_semitones(self, pos: int) -> ScaleElement:
        """Get scale element by its position in semitones."""
        try:
            return self.position_in_semitones_to_element[pos]
        except KeyError:
            raise ValueError(
                f"Position {pos} is not from {self.tonic}-{self.scale_type}."
            )

    def get_element_by_position_in_degrees(self, pos: int) -> ScaleElement:
        """Get scale element by its position in scale degrees."""
        try:
            return self.position_in_degrees_to_element[pos]
        except KeyError:
            raise ValueError(
                f"Position {pos} is out of available positions range."
            )


def check_consonance(
        first: ScaleElement, second: ScaleElement,
        is_perfect_fourth_consonant: bool = False
) -> bool:
    """
    Check whether an interval between two pitches is consonant.

    :param first:
        first pitch
    :param second:
        second pitch
    :param is_perfect_fourth_consonant:
        indicator whether to consider perfect fourth a consonant interval
        (for two voices, it usually considered dissonant)
    :return:
        indicator whether the interval is consonant
    """
    n_semitones_to_consonance = {
        0: True,
        1: False,
        2: False,
        3: True,
        4: True,
        5: is_perfect_fourth_consonant,
        6: False,
        7: True,
        8: True,
        9: True,
        10: False,
        11: False
    }
    interval = abs(first.position_in_semitones - second.position_in_semitones)
    interval %= len(n_semitones_to_consonance)
    return n_semitones_to_consonance[interval]
