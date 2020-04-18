"""
Define data structure that represents musical piece compliant with some rules.

This data structure contains two nested data structures:
1) melodic lines (lists of pitches),
2) piano roll (`numpy` 2D-array with rows corresponding to notes,
   columns corresponding to time steps, and cells containing zeros and ones
   and indicating whether a note is played).

Author: Nikolay Lysenko
"""


import datetime
import os
from typing import Any, Dict, List, NamedTuple

import numpy as np
from sinethesizer.io.utils import get_note_to_position_mapping

from rlmusician.environment.rules import (
    get_harmony_rules_registry,
    get_rhythm_rules_registry,
    get_voice_leading_rules_registry,
)
from rlmusician.utils import (
    Scale,
    ScaleElement,
    create_events_from_piece,
    create_midi_from_piece,
    create_wav_from_events,
)


NOTE_TO_POSITION = get_note_to_position_mapping()
N_EIGHTS_PER_MEASURE = 8


class LineElement(NamedTuple):
    """An element of a melodic line."""

    scale_element: ScaleElement
    start_time_in_eights: int
    end_time_in_eights: int


class Piece:
    """Piece where florid counterpoint line is created given cantus firmus."""

    def __init__(
            self,
            tonic: str,
            scale_type: str,
            n_measures: int,
            cantus_firmus: List[str],
            counterpoint_specifications: Dict[str, Any],
            rhythm_rules: Dict[str, Any],
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
        :param cantus_firmus:
            cantus firmus as a sequence of notes
        :param counterpoint_specifications:
            parameters of a counterpoint line
        :param rhythm_rules:
            names of applicable rhythm rules and their parameters
        :param voice_leading_rules:
            names of applicable voice leading rules and their parameters
        :param harmony_rules:
            names of applicable harmony rules and their parameters
        :param rendering_params:
            settings of saving the piece to TSV, MIDI, and WAV files
        """
        self.tonic = tonic
        self.scale_type = scale_type
        self.n_measures = n_measures
        self.counterpoint_specifications = counterpoint_specifications
        self.names_of_rhythm_rules = rhythm_rules['names']
        self.rhythm_rules_params = rhythm_rules['params']
        self.names_of_voice_leading_rules = voice_leading_rules['names']
        self.voice_leading_rules_params = voice_leading_rules['params']
        self.names_of_harmony_rules = harmony_rules['names']
        self.harmony_rules_params = harmony_rules['params']
        self.rendering_params = rendering_params

        self.scale = Scale(tonic, scale_type)
        self.max_skip = counterpoint_specifications['max_skip']
        self.all_movements = list(range(-self.max_skip, self.max_skip + 1))

        self.current_time_in_eights = None
        self.current_measure_durations = None
        self.past_movements = None
        self.__set_defaults_to_runtime_variables()

        self.cantus_firmus = self.__create_cantus_firmus(cantus_firmus)
        self.counterpoint = self.__create_beginning_of_counterpoint()

        end_note = counterpoint_specifications['end_note']
        self.end_scale_element = self.scale.get_element_by_note(end_note)
        self.__validate_boundary_notes()

        self.lowest_element = self.scale.get_element_by_note(
            counterpoint_specifications['lowest_note']
        )
        self.highest_element = self.scale.get_element_by_note(
            counterpoint_specifications['highest_note']
        )

        self._piano_roll = None
        self.__initialize_piano_roll()
        self.lowest_row_to_show = None
        self.highest_row_to_show = None
        self.__set_range_to_show()

    def __set_defaults_to_runtime_variables(self) -> None:
        """Set default values to variables that change at runtime."""
        self.current_time_in_eights = N_EIGHTS_PER_MEASURE
        self.current_measure_durations = []
        self.past_movements = []

    def __create_cantus_firmus(
            self, cantus_firmus_as_notes: List[str]
    ) -> List[LineElement]:
        """Create cantus firmus from a sequence of its notes."""
        cantus_firmus = [
            LineElement(
                scale_element=self.scale.get_element_by_note(note),
                start_time_in_eights=N_EIGHTS_PER_MEASURE * i,
                end_time_in_eights=N_EIGHTS_PER_MEASURE * (i+1)
            )
            for i, note in enumerate(cantus_firmus_as_notes)
        ]
        return cantus_firmus

    def __create_beginning_of_counterpoint(self) -> List[LineElement]:
        """Create beginning (first measure) of the counterpoint line."""
        start_note = self.counterpoint_specifications['start_note']
        start_element = LineElement(
            self.scale.get_element_by_note(start_note),
            self.counterpoint_specifications['start_pause_in_eights'],
            N_EIGHTS_PER_MEASURE
        )
        counterpoint = [start_element]
        return counterpoint

    def __validate_boundary_notes(self) -> None:
        """Check that boundary notes for both lines are from tonic triad."""
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

    def __initialize_piano_roll(self) -> None:
        """Create piano roll and place all pre-defined notes to it."""
        shape = (len(NOTE_TO_POSITION), N_EIGHTS_PER_MEASURE * self.n_measures)
        self._piano_roll = np.zeros(shape, dtype=int)

        for line_element in self.cantus_firmus:
            self.__add_to_piano_roll(line_element)
        self.__add_to_piano_roll(self.counterpoint[0])

    def __add_to_piano_roll(self, line_element: LineElement) -> None:
        """Add a line element to the piano roll."""
        self._piano_roll[
            line_element.scale_element.position_in_semitones,
            line_element.start_time_in_eights:line_element.end_time_in_eights
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

    def __find_next_position_in_degrees(self, movement: int) -> int:
        """Find position (in scale degrees) that is reached by movement."""
        next_position = (
            self.counterpoint[-1].scale_element.position_in_degrees
            + movement
        )
        return next_position

    def __find_next_element(self, movement: int, duration: int) -> LineElement:
        """Find line element that can be added with movement and duration."""
        next_position = self.__find_next_position_in_degrees(movement)
        next_line_element = LineElement(
            self.scale.get_element_by_position_in_degrees(next_position),
            self.current_time_in_eights,
            self.current_time_in_eights + duration
        )
        return next_line_element

    def __find_cantus_firmus_elements(
            self, duration: int
    ) -> List[LineElement]:
        """Find what in cantus firmus sounds simultaneously with a new note."""
        results = []
        for element in self.cantus_firmus:
            if element.end_time_in_eights <= self.current_time_in_eights:
                continue
            new_note_end_time = self.current_time_in_eights + duration
            if element.start_time_in_eights >= new_note_end_time:
                break
            results.append(element)
        return results

    def __check_range(self, movement: int) -> bool:
        """Check that movement does not lead beyond a range of a line."""
        next_position = self.__find_next_position_in_degrees(movement)
        if next_position < self.lowest_element.position_in_degrees:
            return False
        if next_position > self.highest_element.position_in_degrees:
            return False
        return True

    def __check_rhythm_rules(self, duration: int) -> bool:
        """Check compliance with rules of rhythm."""
        rhythm_registry = get_rhythm_rules_registry()
        durations = [x for x in self.current_measure_durations] + [duration]
        inputs = {'durations': durations}
        for rule_name in self.names_of_rhythm_rules:
            rule_fn = rhythm_registry[rule_name]
            rule_fn_params = self.rhythm_rules_params.get(rule_name, {})
            is_compliant = rule_fn(**inputs, **rule_fn_params)
            if not is_compliant:
                return False
        return True

    def __check_voice_leading_rules(self, movement: int) -> bool:
        """Check compliance with rules of voice leading."""
        voice_leading_registry = get_voice_leading_rules_registry()
        inputs = {
            'line': self.counterpoint,
            'movement': movement,
            'past_movements': self.past_movements,
            'current_time': self.current_time_in_eights
        }
        for rule_name in self.names_of_voice_leading_rules:
            rule_fn = voice_leading_registry[rule_name]
            fn_params = self.voice_leading_rules_params.get(rule_name, {})
            is_compliant = rule_fn(**inputs, **fn_params)
            if not is_compliant:
                return False
        return True

    def __check_harmony_rules(self, movement: int, duration: int) -> bool:
        """Check compliance with rules of harmony."""
        harmony_registry = get_harmony_rules_registry()
        next_line_element = self.__find_next_element(movement, duration)
        cantus_firmus_elements = self.__find_cantus_firmus_elements(duration)
        inputs = {
            'next_line_element': next_line_element,
            'cantus_firmus_elements': cantus_firmus_elements,
            'current_measure_durations': self.current_measure_durations,
        }
        for rule_name in self.names_of_harmony_rules:
            rule_fn = harmony_registry[rule_name]
            rule_fn_params = self.harmony_rules_params.get(rule_name, {})
            is_compliant = rule_fn(**inputs, **rule_fn_params)
            if not is_compliant:
                return False
        return True

    def check_rules(self, movement: int, duration: int) -> bool:
        """
        Check whether suggested continuation is compliant with the rules.

        :param movement:
            shift (in scale degrees) from previous element to a new one
        :param duration:
            duration (in eights) of a new element
        :return:
            `True` if continuation is in accordance with the rules,
            `False` else
        """
        if not self.__check_range(movement):
            return False
        if not self.__check_rhythm_rules(duration):
            return False
        if not self.__check_voice_leading_rules(movement):
            return False
        if not self.__check_harmony_rules(movement, duration):
            return False
        return True

    def __update_current_measure_durations(self, duration: int) -> None:
        """Update division of current measure by played notes."""
        total_duration = sum(self.current_measure_durations) + duration
        if total_duration < N_EIGHTS_PER_MEASURE:
            self.current_measure_durations.append(duration)
        elif total_duration == N_EIGHTS_PER_MEASURE:
            self.current_measure_durations = []
        elif total_duration > N_EIGHTS_PER_MEASURE:
            syncopated_duration = total_duration - N_EIGHTS_PER_MEASURE
            self.current_measure_durations = [syncopated_duration]

    def __finalize_if_needed(self) -> None:
        """Add final measure of counterpoint line if the piece is finished."""
        penultimate_measure_end = N_EIGHTS_PER_MEASURE * (self.n_measures - 1)
        if self.current_time_in_eights == penultimate_measure_end:
            end_line_element = LineElement(
                self.end_scale_element,
                penultimate_measure_end,
                N_EIGHTS_PER_MEASURE * self.n_measures
            )
            self.counterpoint.append(end_line_element)
            self.__add_to_piano_roll(end_line_element)
        last_movement = (
            self.end_scale_element.position_in_degrees
            - self.counterpoint[-2].scale_element.position_in_degrees
        )
        self.past_movements.append(last_movement)
        self.current_time_in_eights = N_EIGHTS_PER_MEASURE * self.n_measures

    def add_line_element(self, movement: int, duration: int) -> None:
        """
        Add a continuation of counterpoint line.

        :param movement:
            shift (in scale degrees) from previous element to a new one
        :param duration:
            duration (in eights) of a new element
        :return:
            None
        """
        piece_duration = N_EIGHTS_PER_MEASURE * self.n_measures
        if self.current_time_in_eights == piece_duration:
            raise RuntimeError("Attempt to add notes to a finished piece.")
        if not self.check_rules(movement, duration):
            raise ValueError('The suggested continuation breaks some rules.')
        next_line_element = self.__find_next_element(movement, duration)
        self.counterpoint.append(next_line_element)
        self.__add_to_piano_roll(next_line_element)
        self.current_time_in_eights += duration
        self.past_movements.append(movement)
        self.__update_current_measure_durations(duration)
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
