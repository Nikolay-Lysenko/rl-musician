"""
Define data structure that represents musical piece compliant with some rules.

This data structure contains two nested data structures:
1) melodic lines (loosely speaking, lists of pitches),
2) piano roll (`numpy` 2D-array with rows corresponding to notes,
   columns corresponding to time steps, and cells containing zeros and ones
   and indicating whether a note is played).

The rules for a piece and its structure come from 5th species counterpoint
(also known as florid counterpoint).

Author: Nikolay Lysenko
"""


import datetime
import os
from typing import Any, Dict, List, NamedTuple

import numpy as np
from sinethesizer.utils.music_theory import get_note_to_position_mapping

from rlmusician.environment.rules import get_rules_registry
from rlmusician.utils import (
    Scale,
    ScaleElement,
    check_consonance,
    create_events_from_piece,
    create_midi_from_piece,
    create_wav_from_events,
)


NOTE_TO_POSITION = get_note_to_position_mapping()
N_EIGHTHS_PER_MEASURE = 8


class LineElement(NamedTuple):
    """An element of a melodic line."""

    scale_element: ScaleElement
    start_time_in_eighths: int
    end_time_in_eighths: int


class Piece:
    """Piece where florid counterpoint line is created given cantus firmus."""

    def __init__(
            self,
            tonic: str,
            scale_type: str,
            cantus_firmus: List[str],
            counterpoint_specifications: Dict[str, Any],
            rules: Dict[str, Any],
            rendering_params: Dict[str, Any]
    ):
        """
        Initialize instance.

        :param tonic:
            tonic pitch class represented by letter (like C or A#)
        :param scale_type:
            type of scale (currently, 'major', 'natural_minor', and
            'harmonic_minor' are supported)
        :param cantus_firmus:
            cantus firmus as a sequence of notes
        :param counterpoint_specifications:
            parameters of a counterpoint line
        :param rules:
            names of applicable rules and parameters of these rules
        :param rendering_params:
            settings of saving the piece to TSV, MIDI, and WAV files
        """
        # Initial inputs.
        self.tonic = tonic
        self.scale_type = scale_type
        self.counterpoint_specifications = counterpoint_specifications
        self.names_of_rules = rules['names']
        self.rules_params = rules['params']
        self.rendering_params = rendering_params

        # Calculated attributes.
        self.scale = Scale(tonic, scale_type)
        self.max_skip = counterpoint_specifications['max_skip_in_degrees']
        self.all_movements = list(range(-self.max_skip, self.max_skip + 1))
        self.n_measures = len(cantus_firmus)
        self.total_duration_in_eighths = (
            N_EIGHTHS_PER_MEASURE * self.n_measures
        )

        # Melodic lines.
        self.cantus_firmus = self.__create_cantus_firmus(cantus_firmus)
        self.counterpoint = self.__create_beginning_of_counterpoint()
        self.is_counterpoint_above = (
            self.counterpoint[0].scale_element.position_in_semitones
            > self.cantus_firmus[0].scale_element.position_in_semitones
        )

        # Boundaries.
        end_note = counterpoint_specifications['end_note']
        self.end_scale_element = self.scale.get_element_by_note(end_note)
        self.lowest_element = self.scale.get_element_by_note(
            counterpoint_specifications['lowest_note']
        )
        self.highest_element = self.scale.get_element_by_note(
            counterpoint_specifications['highest_note']
        )
        self.__validate_boundary_notes()

        # Piano roll.
        self._piano_roll = None
        self.__initialize_piano_roll()
        self.lowest_row_to_show = None
        self.highest_row_to_show = None
        self.__set_range_to_show()

        # Runtime variables.
        self.current_time_in_eighths = None
        self.current_measure_durations = None
        self.past_movements = None
        self.current_motion_start_element = None
        self.is_last_element_consonant = None
        self.__set_defaults_to_runtime_variables()

    def __create_cantus_firmus(
            self, cantus_firmus_as_notes: List[str]
    ) -> List[LineElement]:
        """Create cantus firmus from a sequence of its notes."""
        cantus_firmus = [
            LineElement(
                scale_element=self.scale.get_element_by_note(note),
                start_time_in_eighths=N_EIGHTHS_PER_MEASURE * i,
                end_time_in_eighths=N_EIGHTHS_PER_MEASURE * (i+1)
            )
            for i, note in enumerate(cantus_firmus_as_notes)
        ]
        return cantus_firmus

    def __create_beginning_of_counterpoint(self) -> List[LineElement]:
        """Create beginning (first measure) of the counterpoint line."""
        start_note = self.counterpoint_specifications['start_note']
        start_element = LineElement(
            self.scale.get_element_by_note(start_note),
            self.counterpoint_specifications['start_pause_in_eighths'],
            N_EIGHTHS_PER_MEASURE
        )
        counterpoint = [start_element]
        return counterpoint

    def __validate_boundary_notes(self) -> None:
        """Check that boundary notes for both lines are valid."""
        if not self.cantus_firmus[0].scale_element.is_from_tonic_triad:
            raise ValueError(
                f"{self.cantus_firmus[0].scale_element.note} is not "
                f"a tonic triad member for {self.tonic}-{self.scale_type}; "
                f"therefore, cantus firmus can not start with it."
            )
        if not self.cantus_firmus[-1].scale_element.is_from_tonic_triad:
            raise ValueError(
                f"{self.cantus_firmus[-1].scale_element.note} is not "
                f"a tonic triad member for {self.tonic}-{self.scale_type}; "
                f"therefore, cantus firmus can not end with it."
            )
        if not self.counterpoint[0].scale_element.is_from_tonic_triad:
            raise ValueError(
                f"{self.counterpoint[0].scale_element.note} is not "
                f"a tonic triad member for {self.tonic}-{self.scale_type}; "
                f"therefore, counterpoint line can not start with it."
            )
        if not self.end_scale_element.is_from_tonic_triad:
            raise ValueError(
                f"{self.end_scale_element.note} is not "
                f"a tonic triad member for {self.tonic}-{self.scale_type}; "
                f"therefore, counterpoint line can not end with it."
            )
        lowest_position = self.lowest_element.position_in_semitones
        highest_position = self.highest_element.position_in_semitones
        if lowest_position >= highest_position:
            raise ValueError(
                "Lowest note and highest note are in wrong order: "
                f"{self.counterpoint_specifications['lowest_note']} "
                "is higher than "
                f"{self.counterpoint_specifications['highest_note']}."
            )

    def __initialize_piano_roll(self) -> None:
        """Create piano roll and place all pre-defined notes to it."""
        shape = (len(NOTE_TO_POSITION), self.total_duration_in_eighths)
        self._piano_roll = np.zeros(shape, dtype=np.int32)

        for line_element in self.cantus_firmus:
            self.__add_to_piano_roll(line_element)
        self.__add_to_piano_roll(self.counterpoint[0])

    def __add_to_piano_roll(self, line_element: LineElement) -> None:
        """Add a line element to the piano roll."""
        self._piano_roll[
            line_element.scale_element.position_in_semitones,
            line_element.start_time_in_eighths:line_element.end_time_in_eighths
        ] = 1

    def __set_range_to_show(self) -> None:
        """Set range of pitch positions that can occur in a piece."""
        cantus_firmus_positions = [
            line_element.scale_element.position_in_semitones
            for line_element in self.cantus_firmus
        ]
        cantus_firmus_lower_bound = min(cantus_firmus_positions)
        cantus_firmus_upper_bound = max(cantus_firmus_positions)

        counterpoint_lower_bound = self.lowest_element.position_in_semitones
        counterpoint_upper_bound = self.highest_element.position_in_semitones

        self.lowest_row_to_show = min(
            cantus_firmus_lower_bound,
            counterpoint_lower_bound
        )
        self.highest_row_to_show = max(
            cantus_firmus_upper_bound,
            counterpoint_upper_bound
        )

    def __set_defaults_to_runtime_variables(self) -> None:
        """Set default values to variables that change at runtime."""
        self.current_time_in_eighths = N_EIGHTHS_PER_MEASURE
        self.current_measure_durations = []
        self.past_movements = []
        self.current_motion_start_element = self.counterpoint[0]
        self.is_last_element_consonant = True

    def __find_next_position_in_degrees(self, movement: int) -> int:
        """Find position (in scale degrees) that is reached by movement."""
        next_position = (
            self.counterpoint[-1].scale_element.position_in_degrees
            + movement
        )
        return next_position

    def __find_next_element(self, movement: int, duration: int) -> LineElement:
        """Find continuation of counterpoint line by movement and duration."""
        next_position = self.__find_next_position_in_degrees(movement)
        next_line_element = LineElement(
            self.scale.get_element_by_position_in_degrees(next_position),
            self.current_time_in_eighths,
            self.current_time_in_eighths + duration
        )
        return next_line_element

    def __find_cf_elements(self, duration: int) -> List[LineElement]:
        """Find what in cantus firmus sounds simultaneously with a new note."""
        start_index = self.current_time_in_eighths // N_EIGHTHS_PER_MEASURE
        end_time = self.current_time_in_eighths + duration
        end_index = (end_time - 1) // N_EIGHTHS_PER_MEASURE + 1
        results = self.cantus_firmus[start_index:end_index]
        return results

    def __find_previous_cf_element(self) -> LineElement:
        """Find what in cantus firmus sounds before a new note."""
        index = (self.current_time_in_eighths - 1) // N_EIGHTHS_PER_MEASURE
        result = self.cantus_firmus[index]
        return result

    def __check_range(self, movement: int) -> bool:
        """Check that movement does not lead beyond a range of a line."""
        next_position = self.__find_next_position_in_degrees(movement)
        if next_position < self.lowest_element.position_in_degrees:
            return False
        if next_position > self.highest_element.position_in_degrees:
            return False
        return True

    def __check_total_duration(self, duration: int) -> bool:
        """Check that nothing is suspended to the last measure."""
        available_duration = N_EIGHTHS_PER_MEASURE * (self.n_measures - 1)
        return self.current_time_in_eighths + duration <= available_duration

    def __check_rules(self, movement: int, duration: int) -> bool:
        """Check compliance with the rules."""
        registry = get_rules_registry()
        continuation = self.__find_next_element(movement, duration)
        durations = [x for x in self.current_measure_durations] + [duration]
        cantus_firmus_elements = self.__find_cf_elements(duration)
        previous_cantus_firmus_element = self.__find_previous_cf_element()
        state = {
            'line': self.counterpoint,
            'counterpoint_continuation': continuation,
            'movement': movement,
            'past_movements': self.past_movements,
            'piece_duration': self.total_duration_in_eighths,
            'current_measure_durations': self.current_measure_durations,
            'durations': durations,
            'cantus_firmus_elements': cantus_firmus_elements,
            'previous_cantus_firmus_element': previous_cantus_firmus_element,
            'current_motion_start_element': self.current_motion_start_element,
            'is_last_element_consonant': self.is_last_element_consonant,
            'is_counterpoint_above': self.is_counterpoint_above,
            'counterpoint_end': self.end_scale_element,
        }
        for rule_name in self.names_of_rules:
            rule_fn = registry[rule_name]
            rule_fn_params = self.rules_params.get(rule_name, {})
            is_compliant = rule_fn(**state, **rule_fn_params)
            if not is_compliant:
                return False
        return True

    def check_validity(self, movement: int, duration: int) -> bool:
        """
        Check whether suggested continuation is valid.

        :param movement:
            shift (in scale degrees) from previous element to a new one
        :param duration:
            duration (in eighths) of a new element
        :return:
            `True` if the continuation is valid, `False` else
        """
        if movement not in self.all_movements:
            return False
        if not self.__check_range(movement):
            return False
        if not self.__check_total_duration(duration):
            return False
        if not self.__check_rules(movement, duration):
            return False
        return True

    def __update_current_measure_durations(self, duration: int) -> None:
        """Update division of current measure by played notes."""
        total_duration = sum(self.current_measure_durations) + duration
        if total_duration < N_EIGHTHS_PER_MEASURE:
            self.current_measure_durations.append(duration)
        elif total_duration == N_EIGHTHS_PER_MEASURE:
            self.current_measure_durations = []
        else:
            syncopated_duration = total_duration - N_EIGHTHS_PER_MEASURE
            self.current_measure_durations = [syncopated_duration]

    def __update_current_motion_start(self) -> None:
        """Update element opening continuous motion in one direction."""
        if len(self.past_movements) < 2:
            return
        if self.past_movements[-1] * self.past_movements[-2] < 0:
            self.current_motion_start_element = self.counterpoint[-2]

    def __update_indicator_of_consonance(self, duration: int) -> None:
        """Update indicator of current vertical consonance between lines."""
        cantus_firmus_elements = self.__find_cf_elements(duration)
        cantus_firmus_element = cantus_firmus_elements[-1].scale_element
        counterpoint_element = self.counterpoint[-1].scale_element
        self.is_last_element_consonant = check_consonance(
            cantus_firmus_element, counterpoint_element
        )

    def __update_runtime_variables(self, movement: int, duration: int) -> None:
        """Update runtime variables representing current state."""
        self.__update_indicator_of_consonance(duration)
        self.current_time_in_eighths += duration
        self.past_movements.append(movement)
        self.__update_current_measure_durations(duration)
        self.__update_current_motion_start()

    def __finalize_if_needed(self) -> None:
        """Add final measure of counterpoint line if the piece is finished."""
        penultimate_measure_end = N_EIGHTHS_PER_MEASURE * (self.n_measures - 1)
        if self.current_time_in_eighths < penultimate_measure_end:
            return
        end_line_element = LineElement(
            self.end_scale_element,
            penultimate_measure_end,
            self.total_duration_in_eighths
        )
        self.counterpoint.append(end_line_element)
        self.__add_to_piano_roll(end_line_element)
        last_movement = (
            self.end_scale_element.position_in_degrees
            - self.counterpoint[-2].scale_element.position_in_degrees
        )
        self.past_movements.append(last_movement)
        self.current_time_in_eighths = self.total_duration_in_eighths

    def add_line_element(self, movement: int, duration: int) -> None:
        """
        Add a continuation of counterpoint line.

        :param movement:
            shift (in scale degrees) from previous element to a new one
        :param duration:
            duration (in eighths) of a new element
        :return:
            None
        """
        if self.current_time_in_eighths == self.total_duration_in_eighths:
            raise RuntimeError("Attempt to add notes to a finished piece.")
        if not self.check_validity(movement, duration):
            raise ValueError(
                "The suggested continuation is not valid. "
                "It either breaks some rules or goes beyond ranges."
            )
        next_line_element = self.__find_next_element(movement, duration)
        self.counterpoint.append(next_line_element)
        self.__add_to_piano_roll(next_line_element)
        self.__update_runtime_variables(movement, duration)
        self.__finalize_if_needed()

    def reset(self) -> None:
        """
        Discard all changes made after initialization.

        :return:
            None
        """
        self.counterpoint = self.counterpoint[0:1]
        self.__initialize_piano_roll()
        self.__set_defaults_to_runtime_variables()

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
        Save piece as TSV, MIDI, and WAV files.

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
