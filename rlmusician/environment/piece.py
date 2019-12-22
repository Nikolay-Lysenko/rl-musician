"""
Define data structure that represents musical piece compliant with some rules.

This data structure contains two nested data structures:
1) instrumental lines (lists of pitches),
2) piano roll (`numpy` 2D-array with rows corresponding to notes,
   columns corresponding to time steps, and cells containing zeros and ones
   and indicating whether a note is played).

Author: Nikolay Lysenko
"""


import datetime
import itertools
import os
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
from sinethesizer.io.utils import (
    get_list_of_notes, get_note_to_position_mapping
)

from rlmusician.utils import (
    create_events_from_piece,
    create_midi_from_piece,
    create_wav_from_events,
    get_positions_from_scale,
    get_tonic_triad_positions,
    slice_positions
)


NOTE_TO_POSITION = get_note_to_position_mapping()


class LineElement(NamedTuple):
    """A pitch that can be used within a line."""

    absolute_position: int
    relative_position: int
    is_from_tonic_triad: bool
    allowed_movements: List[int]


class Piece:
    """Musical piece compliant with some rules of counterpoint writing."""

    def __init__(
            self,
            tonic: str,
            scale: str,
            n_measures: int,
            max_skip: int,
            line_specifications: List[Dict[str, Any]],
            rendering_params: Dict[str, Any]
    ):
        """
        Initialize instance.

        :param tonic:
            tonic pitch class represented by letter (like C or A#)
        :param scale:
            scale (currently, 'major' and 'minor' are supported)
        :param n_measures:
            duration of piece in measures
        :param max_skip:
            maximum allowed skip (in scale degrees) between consecutive notes
            from the same line
        :param line_specifications:
            parameters of lines
        :param rendering_params:
            settings of saving piece to TSV, MIDI, and WAV files
        """
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
        self._piano_roll = np.zeros(shape, dtype=int)
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
        self.last_finished_measure = 0

    def __get_allowed_movements(
            self, current_position: int, end_position: int,
            is_from_tonic_triad: bool
    ) -> List[int]:
        """Get all possible shifts in scale degrees from current position."""
        allowed_movements = [
            movement for movement in self.all_movements
            if 0 <= current_position + movement < end_position
            # Only pitches from tonic triad may be rearticulated.
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
            element = LineElement(
                absolute_position,
                pitch_number,
                is_from_triad,
                allowed_movements
            )
            elements.append(element)
            mapping[absolute_position] = pitch_number
        self.line_elements.append(elements)
        self.line_mappings.append(mapping)

    def __update_range_to_show(self, specs: Dict[str, Any]) -> None:
        """Extend range of pitches that can occur in a piece."""
        low_bound = NOTE_TO_POSITION[specs['lowest_note']]
        self.lowest_row_to_show = min(
            self.lowest_row_to_show or 87, low_bound
        )
        high_bound = NOTE_TO_POSITION[specs['highest_note']]
        self.highest_row_to_show = max(
            self.highest_row_to_show or 0, high_bound
        )

    def __add_end_note(self, note: str, end_type: str) -> None:
        """Add start note or end note to its line and to piano roll."""
        absolute_position = NOTE_TO_POSITION[note]
        relative_position = self.line_mappings[-1].get(absolute_position)
        if relative_position is None:
            raise ValueError(
                f"Passed {end_type} note {note} does not belong to "
                f"{self.tonic}-{self.scale} or is out of line range."
            )
        element = self.line_elements[-1][relative_position]
        if not element.is_from_tonic_triad:
            raise ValueError(
                f"{note} is not a tonic triad member for "
                f"{self.tonic}-{self.scale}; it can not be {end_type} note."
            )
        column = 0 if end_type == 'start' else -1
        self.lines[-1][column] = element
        self._piano_roll[absolute_position, column] = 1

    def __compute_destination(
            self, movement: int, line: List[Optional[LineElement]],
            elements: List[LineElement]
    ) -> Optional[LineElement]:
        """Compute note that is obtained by the movement and validate it."""
        current_element = line[self.last_finished_measure]
        if movement not in current_element.allowed_movements:
            return None
        position = current_element.relative_position + movement
        destination = elements[position]

        # Change of direction can not occur after two non-triad members.
        if self.last_finished_measure > 0:
            previous_element = line[self.last_finished_measure - 1]
            previous_movement = (
                current_element.relative_position
                - previous_element.relative_position
            )
            improper_change_of_direction = min(
                not previous_element.is_from_tonic_triad,
                not current_element.is_from_tonic_triad,
                previous_movement * movement < 0
            )
            if improper_change_of_direction:
                return None

        # Skip must lead to a tonic triad member.
        if abs(movement) > 1 and not destination.is_from_tonic_triad:
            return None

        # There must be a way to reach end note with step motion.
        degrees_to_end_note = abs(
            destination.relative_position - line[-1].relative_position
        )
        measures_left = self.n_measures - self.last_finished_measure - 2
        if degrees_to_end_note > measures_left:
            return None

        return destination

    @staticmethod
    def __check_is_it_consonant(chord: List[LineElement]) -> bool:
        """Check that all intervals from a chord are consonant."""
        n_semitones_to_consonance = {
            0: True, 1: False, 2: False, 3: True, 4: True, 5: True,
            6: False, 7: True, 8: True, 9: True, 10: False, 11: False
        }
        for first, second in itertools.combinations(chord, 2):
            interval = first.absolute_position - second.absolute_position
            interval %= len(n_semitones_to_consonance)
            if not n_semitones_to_consonance[interval]:
                return False
        return True

    def check_movements(self, movements: List[int]) -> bool:
        """
        Check whether suggested movements are compliant with some rules.

        :param movements:
            list of shifts in scale degrees for each line
        :return:
            `True` if movements are permitted, `False` else
        """
        if len(movements) != len(self.lines):
            raise ValueError(
                f"Wrong number of lines: {len(movements)}, "
                f"expected: {len(self.lines)}."
            )
        destinations = []
        zipped = zip(movements, self.lines, self.line_elements)
        for movement, line, elements in zipped:
            result = self.__compute_destination(movement, line, elements)
            if result is None:
                return False
            destinations.append(result)
        return self.__check_is_it_consonant(destinations)

    def add_measure(self, movements: List[int]) -> None:
        """
        Add continuations of all lines for a next measure.

        :param movements:
            list of shifts in scale degrees for each line
        :return:
            None
        """
        if self.last_finished_measure == self.n_measures - 1:
            raise RuntimeError("Attempt to add notes to a finished piece.")
        if not self.check_movements(movements):
            raise ValueError('Passed movements are not permitted.')
        zipped = zip(movements, self.lines, self.line_elements)
        for movement, line, elements in zipped:
            current_element = line[self.last_finished_measure]
            position = current_element.relative_position + movement
            next_element = elements[position]
            line[self.last_finished_measure + 1] = next_element
            self._piano_roll[
                next_element.absolute_position, self.last_finished_measure + 1
            ] = 1
        self.last_finished_measure += 1

    def reset(self) -> None:
        """
        Discard all changes made after initialization.

        :return:
            None
        """
        measures_to_drop = list(range(1, self.n_measures - 1))
        for line, measure in itertools.product(self.lines, measures_to_drop):
            line[measure] = None
        self._piano_roll[:, measures_to_drop] = 0
        self.last_finished_measure = 0

    @property
    def piano_roll(self) -> np.ndarray:
        """Get piece representation as piano roll (without irrelevant rows)."""
        reverted_roll = self._piano_roll[
            self.lowest_row_to_show:self.highest_row_to_show + 1, :
        ]
        roll = np.flip(reverted_roll, axis=0)
        return roll

    def render(self) -> None:  # pragma: no cover
        """
        Save final piano roll as TSV, MIDI, and WAV files.

        :return:
            None
        """
        top_level_dir = self.rendering_params['dir']
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S,%f")
        nested_dir = os.path.join(top_level_dir, f"result_{now}")
        os.mkdir(nested_dir)

        roll_path = os.path.join(nested_dir, 'piano_roll.tsv')
        np.savetxt(roll_path, self.piano_roll, fmt='%i', delimiter='\t')

        midi_path = os.path.join(nested_dir, 'music.mid')
        midi_params = self.rendering_params['midi']
        measure = self.rendering_params['measure_in_seconds']
        create_midi_from_piece(self, midi_path, measure, **midi_params)

        events_path = os.path.join(nested_dir, 'sinethesizer_events.tsv')
        events_params = self.rendering_params['sinethesizer']
        create_events_from_piece(self, events_path, measure, **events_params)

        wav_path = os.path.join(nested_dir, 'music.wav')
        create_wav_from_events(events_path, wav_path)
