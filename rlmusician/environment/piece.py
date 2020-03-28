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
from typing import Any, Dict, List, NamedTuple

import numpy as np
from sinethesizer.io.utils import get_note_to_position_mapping

from rlmusician.environment.rules import (
    get_voice_leading_rules_registry,
    get_harmony_rules_registry
)
from rlmusician.utils import (
    create_events_from_piece,
    create_midi_from_piece,
    create_scale,
    create_wav_from_events,
    slice_scale
)


NOTE_TO_POSITION = get_note_to_position_mapping()
TONIC_TRIAD_DEGREES = (1, 3, 5)


class LineElement(NamedTuple):
    """A pitch that can be used within a line."""

    absolute_position: int
    relative_position: int
    degree: int
    is_from_tonic_triad: bool
    feasible_movements: List[int]


class Piece:
    """Musical piece compliant with some rules of counterpoint writing."""

    def __init__(
            self,
            tonic: str,
            scale_type: str,
            n_measures: int,
            max_skip: int,
            line_specifications: List[Dict[str, Any]],
            voice_leading_rules: Dict[str, Any],
            harmony_rules: Dict[str, Any],
            rendering_params: Dict[str, Any]
    ):
        """
        Initialize instance.

        :param tonic:
            tonic pitch class represented by letter (like C or A#)
        :param scale_type:
            type of scale (currently, 'major', 'natural_minor', and
            'harmonic_minor' are supported)
        :param n_measures:
            duration of piece in measures
        :param max_skip:
            maximum allowed skip (in scale degrees) between successive notes
            from the same line
        :param line_specifications:
            parameters of lines
        :param voice_leading_rules:
            names of applicable voice leading rules and their parameters
        :param harmony_rules:
            names of applicable harmony rules and their parameters
        :param rendering_params:
            settings of saving piece to TSV, MIDI, and WAV files
        """
        self.tonic = tonic
        self.scale_type = scale_type
        self.n_measures = n_measures
        self.max_skip = max_skip
        self.line_specifications = line_specifications
        self.names_of_voice_leading_rules = voice_leading_rules['names']
        self.voice_leading_rules_params = voice_leading_rules['params']
        self.names_of_harmony_rules = harmony_rules['names']
        self.harmony_rules_params = harmony_rules['params']
        self.rendering_params = rendering_params

        self.scale = create_scale(tonic, scale_type)
        self.all_movements = list(range(-self.max_skip, self.max_skip + 1))

        shape = (len(NOTE_TO_POSITION), self.n_measures)
        self._piano_roll = np.zeros(shape, dtype=int)
        self.lowest_row_to_show = None
        self.highest_row_to_show = None

        self.lines = []
        self.line_elements = []
        self.passed_movements = []
        for specs in line_specifications:
            self.lines.append([None for _ in range(self.n_measures)])
            self.passed_movements.append([])
            self.__define_elements(specs)
            self.__update_range_to_show(specs)
            self.__add_boundary_note(specs['start_note'], 'start')
            self.__add_boundary_note(specs['end_note'], 'end')
        self.last_finished_measure = 0

    def __get_feasible_movements(
            self, current_position: int, end_position: int,
    ) -> List[int]:
        """Get intervals in scale degrees that do not go beyond the range."""
        feasible_movements = [
            movement for movement in self.all_movements
            if 0 <= current_position + movement < end_position
        ]
        return feasible_movements

    def __define_elements(self, specs: Dict[str, Any]) -> None:
        """Define list of pitches that can be used within a line."""
        sliced_scale = slice_scale(
            self.scale, specs['lowest_note'], specs['highest_note']
        )
        if not sliced_scale:
            raise ValueError(
                f"No pitches from {self.tonic}-{self.scale} are between "
                f"{specs['lowest_note']} and {specs['highest_note']}."
            )
        elements = []
        for relative_position, scale_element in enumerate(sliced_scale):
            feasible_movements = self.__get_feasible_movements(
                relative_position, len(sliced_scale)
            )
            element = LineElement(
                scale_element.absolute_position,
                relative_position,
                scale_element.degree,
                scale_element.degree in TONIC_TRIAD_DEGREES,
                feasible_movements
            )
            elements.append(element)
        self.line_elements.append(elements)

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

    def __add_boundary_note(self, note: str, bound_type: str) -> None:
        """Add start note or end note to its line and to piano roll."""
        absolute_position = NOTE_TO_POSITION[note]
        element_as_list = [
            x for x in self.line_elements[-1]
            if x.absolute_position == absolute_position
        ]
        if not element_as_list:
            raise ValueError(
                f"Passed {bound_type} note {note} does not belong to "
                f"{self.tonic}-{self.scale} or is out of line range."
            )
        element = element_as_list[0]
        if not element.is_from_tonic_triad:
            raise ValueError(
                f"{note} is not a tonic triad member for "
                f"{self.tonic}-{self.scale}; it can not be {bound_type} note."
            )
        column = 0 if bound_type == 'start' else -1
        self.lines[-1][column] = element
        self._piano_roll[absolute_position, column] = 1

    def __finalize_if_needed(self) -> None:
        """Add movements to final pitches if the piece is finished."""
        if self.last_finished_measure == self.n_measures - 2:
            for line, past_movements in zip(self.lines, self.passed_movements):
                last_position = line[-1].relative_position
                last_but_one_position = line[-2].relative_position
                past_movements.append(last_position - last_but_one_position)
            self.last_finished_measure += 1

    def __find_destinations(self, movements: List[int]) -> List[LineElement]:
        """Find sonority that is added by movements."""
        destination_sonority = []
        zipped = zip(self.lines, self.line_elements, movements)
        for line, line_elements, movement in zipped:
            current_element = line[self.last_finished_measure]
            next_position = current_element.relative_position + movement
            next_element = line_elements[next_position]
            destination_sonority.append(next_element)
        return destination_sonority

    def __check_voice_leading_rules(self, movements: List[int]) -> bool:
        """Check compliance with rules of voice leading."""
        voice_leading_registry = get_voice_leading_rules_registry()
        zipped = zip(
            self.lines,
            self.line_elements,
            movements,
            self.passed_movements
        )
        for line, line_elements, movement, previous_movements in zipped:
            last_pitch = line[self.last_finished_measure]
            if movement not in last_pitch.feasible_movements:
                return False
            inputs = {
                'line': line,
                'line_elements': line_elements,
                'movement': movement,
                'previous_movements': previous_movements,
                'measure': self.last_finished_measure
            }
            for rule_name in self.names_of_voice_leading_rules:
                rule_fn = voice_leading_registry[rule_name]
                fn_params = self.voice_leading_rules_params.get(rule_name, {})
                is_compliant = rule_fn(**inputs, **fn_params)
                if not is_compliant:
                    return False
        return True

    def __check_harmony_rules(self, movements: List[int]) -> bool:
        """Check compliance with rules of harmony."""
        harmony_registry = get_harmony_rules_registry()
        destination_sonority = self.__find_destinations(movements)
        for rule_name in self.names_of_harmony_rules:
            rule_fn = harmony_registry[rule_name]
            rule_fn_params = self.harmony_rules_params.get(rule_name, {})
            is_compliant = rule_fn(destination_sonority, **rule_fn_params)
            if not is_compliant:
                return False
        return True

    def check_movements(self, movements: List[int]) -> bool:
        """
        Check whether suggested movements are compliant with the rules.

        :param movements:
            list of shifts in scale degrees for each line
        :return:
            `True` if movements are in accordance with the rules, `False` else
        """
        if len(movements) != len(self.lines):
            raise ValueError(
                f"Wrong number of lines: {len(movements)}, "
                f"expected: {len(self.lines)}."
            )
        if not self.__check_voice_leading_rules(movements):
            return False
        if not self.__check_harmony_rules(movements):
            return False
        return True

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
            raise ValueError('Suggested movements break some rules.')
        next_sonority = self.__find_destinations(movements)
        for line, next_element in zip(self.lines, next_sonority):
            line[self.last_finished_measure + 1] = next_element
            self._piano_roll[
                next_element.absolute_position, self.last_finished_measure + 1
            ] = 1
        for movement, past_movements in zip(movements, self.passed_movements):
            past_movements.append(movement)
        self.last_finished_measure += 1
        self.__finalize_if_needed()

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
        self.passed_movements = [[] for _ in self.passed_movements]

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
