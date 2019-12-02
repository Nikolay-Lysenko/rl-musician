"""
Author: Nikolay Lysenko
"""


from typing import Any, Dict, List

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
    positions_from_scale = [
        x for x in range(len(NOTE_TO_POSITION))
        if scale_patterns[scale][(x - tonic_position) % N_SEMITONES_PER_OCTAVE]
    ]
    return positions_from_scale


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


class Piece:
    """"""

    def __init__(
            self,
            tonic: str,
            scale: str,
            n_measures: int,
            line_params: List[Dict[str, Any]],
            rendering_params: Dict[str, Any]
    ):
        """"""
        self.tonic = tonic
        self.scale = scale
        self.n_measures = n_measures
        self.line_params = line_params
        self.rendering_params = rendering_params

        self.positions_from_scale = get_positions_from_scale(tonic, scale)
        self.tonic_triad_positions = get_tonic_triad_positions(tonic, scale)

        self.lines = []
        self.allowed_positions = []
        for params in line_params:
            self.__create_line(params)
        self.piano_roll = None
        self.__create_piano_roll()

    def __create_line(self, params: Dict[str, Any]) -> None:
        """"""
        sliced_positions = slice_positions(
            self.positions_from_scale,
            params['lowest_note'], params['highest_note']
        )
        if not sliced_positions:
            raise ValueError(
                f"No pitches from {self.tonic}-{self.scale} are between "
                f"{params['lowest_note']} and {params['highest_note']}."
            )
        self.allowed_positions.append(sliced_positions)

    def __create_piano_roll(self) -> None:
        """"""

    def validate_movements(self, movements: List[int]) -> bool:
        """"""

    def add_measure(self, movements: List[int]) -> None:
        """"""
        if not self.validate_movements(movements):
            raise ValueError('Passed movements are not permitted.')
