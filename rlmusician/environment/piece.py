"""
Author: Nikolay Lysenko
"""


from typing import Any, Dict, List, NamedTuple

from sinethesizer.io.utils import get_note_to_position_mapping


N_SEMITONES_PER_OCTAVE = 12
NOTE_TO_POSITION = get_note_to_position_mapping()


def get_positions_from_scale(tonic: str, scale: str) -> List[int]:
    """"""
    scale_patterns = {
        'major': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        'minor': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
    }
    tonic_position = NOTE_TO_POSITION[tonic + '1']
    positions_from_scale = [
        x for x in range(len(NOTE_TO_POSITION))
        if scale_patterns[scale][(x - tonic_position) % N_SEMITONES_PER_OCTAVE]
    ]
    return positions_from_scale


def get_tonic_triad_positions(tonic: str, scale: str) -> List[int]:
    """"""
    scale_patterns = {
        'major': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        'minor': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    }
    tonic_position = NOTE_TO_POSITION[tonic + '1']
    tonic_triad_positions = [
        x for x in range(len(NOTE_TO_POSITION))
        if scale_patterns[scale][(x - tonic_position) % N_SEMITONES_PER_OCTAVE]
    ]
    return tonic_triad_positions


def slice_positions(
        positions: List[int],
        lowest_note: str,
        highest_note: str,
) -> List[int]:
    """"""
    lowest_position = NOTE_TO_POSITION[lowest_note]
    highest_position = NOTE_TO_POSITION[highest_note]
    sliced_positions = [
        x for x in positions
        if lowest_position <= x <= highest_position
    ]
    return sliced_positions


class AvailablePitch(NamedTuple):
    """"""

    absolute_position: int
    is_from_tonic_triad: bool
    allowed_movements: List[int]


class Piece:
    """"""

    def __init__(
            self,
            tonic: str,
            scale: str,
            n_measures: int,
            max_skip: int,
            line_specifications: List[Dict[str, Any]],
            rendering_params: Dict[str, Any]
    ):
        """Initialize instance."""
        self.tonic = tonic
        self.scale = scale
        self.n_measures = n_measures
        self.max_skip = max_skip
        self.line_specifications = line_specifications
        self.rendering_params = rendering_params

        self.positions_from_scale = get_positions_from_scale(tonic, scale)
        self.tonic_triad_positions = get_tonic_triad_positions(tonic, scale)

        self.lines = []
        self.line_elements = []
        for line_specification in line_specifications:
            self.__define_elements(line_specification)
            self.__validate_end_note(line_specification['start_note'], 'start')
            self.__validate_end_note(line_specification['end_note'], 'end')
        self.piano_roll = None
        self.__create_piano_roll()

    def __define_elements(self, specs: Dict[str, Any]) -> None:
        """Define list of pitches that can be used within a line."""
        sliced_positions = slice_positions(
            self.positions_from_scale,
            specs['lowest_note'], specs['highest_note']
        )
        if not sliced_positions:
            raise ValueError(
                f"No pitches from {self.tonic}-{self.scale} are between "
                f"{specs['lowest_note']} and {specs['highest_note']}."
            )
        elements = []
        movements = list(range(-self.max_skip, self.max_skip + 1))
        for pitch_number, absolute_position in enumerate(sliced_positions):
            is_from_triad = absolute_position in self.tonic_triad_positions
            allowed_movements = [
                x for x in movements
                if 0 <= pitch_number + x < len(sliced_positions)
                and (x != 0 or is_from_triad)
            ]  # TODO: Move it to outer function `find_allowed_movements`.
            pitch = AvailablePitch(
                absolute_position,
                is_from_triad,
                allowed_movements
            )
            elements.append(pitch)
        self.line_elements.append(elements)

    def __validate_end_note(self, note: str, end_type: str) -> None:
        """Validate start note or end note for a line."""
        available_pitches = self.line_elements[-1]
        note_position = NOTE_TO_POSITION[note]
        note_is_valid = max(
            x.is_from_tonic_triad
            for x in available_pitches
            if x.absolute_position == note_position
        )
        if not note_is_valid:
            raise ValueError(
                f"{note} is not a tonic triad member for "
                f"{self.tonic}-{self.scale}; it can not be {end_type} note."
            )

    def __create_piano_roll(self) -> None:
        """"""

    def validate_movements(self, movements: List[int]) -> bool:
        """"""

    def add_measure(self, movements: List[int]) -> None:
        """"""
        if not self.validate_movements(movements):
            raise ValueError('Passed movements are not permitted.')
