"""
Define data structure that represents musical piece.

Author: Nikolay Lysenko
"""


from typing import Any, Dict, List, NamedTuple

import numpy as np
from sinethesizer.io.utils import get_note_to_position_mapping

from rlmusician.utils import (
    get_positions_from_scale, get_tonic_triad_positions, slice_positions
)


NOTE_TO_POSITION = get_note_to_position_mapping()


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
        self.all_movements = list(range(-self.max_skip, self.max_skip + 1))

        shape = (len(NOTE_TO_POSITION), self.n_measures)
        self._piano_roll = np.array(shape, dtype=int)
        self.lowest_row_to_show = None
        self.highest_row_to_show = None

        self.lines = []
        self.line_elements = []
        self.line_mappings = []
        for specs in line_specifications:
            self.lines.append([None for _ in range(self.n_measures)])
            self.__define_elements(specs)
            self.__update_range_to_show(specs)
            self.__add_end_note(specs['start_note'], 'start')
            self.__add_end_note(specs['end_note'], 'end')

    def __get_allowed_movements(
            self, current_position: int, end_position: int,
            is_from_tonic_triad: bool
    ) -> List[int]:
        """Get all possible shifts in scale degrees from current position."""
        allowed_movements = [
            movement for movement in self.all_movements
            if 0 <= current_position + movement < end_position
            and (movement != 0 or is_from_tonic_triad)
        ]
        return allowed_movements

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
        mapping = {}
        for pitch_number, absolute_position in enumerate(sliced_positions):
            is_from_triad = absolute_position in self.tonic_triad_positions
            allowed_movements = self.__get_allowed_movements(
                pitch_number, len(sliced_positions), is_from_triad
            )
            pitch = AvailablePitch(
                absolute_position,
                is_from_triad,
                allowed_movements
            )
            elements.append(pitch)
            mapping[absolute_position] = pitch_number
        self.line_elements.append(elements)
        self.line_mappings.append(mapping)

    def __update_range_to_show(self, specs: Dict[str, Any]) -> None:
        """"""
        low_bound = NOTE_TO_POSITION[specs['lowest_note']]
        self.lowest_row_to_show = max(
            self.lowest_row_to_show or 0, low_bound
        )
        high_bound = NOTE_TO_POSITION[specs['highest_note']]
        self.highest_row_to_show = min(
            self.highest_row_to_show or 87, high_bound
        )

    def __add_end_note(self, note: str, end_type: str) -> None:
        """Add start note or end note to its line and to piano roll."""
        absolute_position = NOTE_TO_POSITION[note]
        relative_position = self.line_mappings[-1][absolute_position]
        line_elements = self.line_elements[-1]
        note_is_valid = line_elements[relative_position].is_from_tonic_triad
        if not note_is_valid:
            raise ValueError(
                f"{note} is not a tonic triad member for "
                f"{self.tonic}-{self.scale}; it can not be {end_type} note."
            )
        column = 0 if end_type == 'start' else -1
        self.lines[-1][column] = relative_position
        self._piano_roll[absolute_position, column] = 1

    @property
    def piano_roll(self) -> np.ndarray:
        """"""
        piano_roll = self._piano_roll[
            self.lowest_row_to_show:self.highest_row_to_show+1, :
        ]
        return np.flip(piano_roll, axis=0)

    def validate_movements(self, movements: List[int]) -> bool:
        """"""

    def add_measure(self, movements: List[int]) -> None:
        """"""
        if not self.validate_movements(movements):
            raise ValueError('Passed movements are not permitted.')
