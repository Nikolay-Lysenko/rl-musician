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
    get_voice_leading_rules_registry,
    get_harmony_rules_registry
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
        self.names_of_voice_leading_rules = voice_leading_rules['names']
        self.voice_leading_rules_params = voice_leading_rules['params']
        self.names_of_harmony_rules = harmony_rules['names']
        self.harmony_rules_params = harmony_rules['params']
        self.rendering_params = rendering_params

        self.scale = Scale(tonic, scale_type)
        self.max_skip = counterpoint_specifications['max_skip']
        self.all_movements = list(range(-self.max_skip, self.max_skip + 1))
        self.current_time_in_eights = N_EIGHTS_PER_MEASURE

        self.cantus_firmus = self.__create_cantus_firmus(cantus_firmus)
        self.counterpoint = self.__create_beginning_of_counterpoint()

        end_note = self.counterpoint_specifications['end_note']
        self.end_scale_element = self.scale.get_element_by_note(end_note)
        self.__validate_boundary_notes()

        self._piano_roll = None
        self.__initialize_piano_roll()
        self.lowest_row_to_show = None
        self.highest_row_to_show = None
        self.__set_range_to_show()

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

        lowest_element = self.scale.get_element_by_note(
            self.counterpoint_specifications['lowest_note']
        )
        counterpoint_lower_bound = lowest_element.position_in_semitones
        highest_element = self.scale.get_element_by_note(
            self.counterpoint_specifications['highest_note']
        )
        counterpoint_upper_bound = highest_element.position_in_semitones

        self.lowest_row_to_show = min(
            cantus_firmus_lower_bound,
            counterpoint_lower_bound
        )
        self.highest_row_to_show = max(
            cantus_firmus_upper_bound,
            counterpoint_upper_bound
        )

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
        self.counterpoint = self.counterpoint[0:1]
        self.__initialize_piano_roll()
        self.current_time_in_eights = N_EIGHTS_PER_MEASURE

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
