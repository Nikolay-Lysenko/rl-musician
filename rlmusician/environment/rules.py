"""
Check compliance with some rules of rhythm, voice leading, and harmony.

Author: Nikolay Lysenko
"""


from math import ceil
from typing import Callable, Dict, List

from rlmusician.utils.music_theory import ScaleElement, check_consonance


N_EIGHTHS_PER_MEASURE = 8


# Rhythm rules.

def check_validity_of_rhythmic_pattern(durations: List[int], **kwargs) -> bool:
    """
    Check that current measure is properly divided by notes.

    :param durations:
        durations (in eighths) of all notes from a current measure
        (including a new note); if a new note prolongs to the next measure,
        its full duration is included; however, if the first note starts
        in the previous measure, only its duration within the current measure
        is included
    :return:
        indicator whether a continuation is in accordance with the rule
    """
    valid_patterns = [
        [4, 4],
        [4, 2, 2],
        [4, 2, 1, 1],
        [2, 2, 2, 2],
        [2, 2, 2, 1, 1],
        [2, 1, 1, 2, 2],
        [4, 8],
        [2, 2, 8],
        [2, 1, 1, 8],
    ]
    for valid_pattern in valid_patterns:
        if valid_pattern[:len(durations)] == durations:
            return True
    return False


# Voice leading rules.

def check_stability_of_rearticulated_pitch(
        counterpoint_continuation: 'LineElement',
        movement: int,
        **kwargs
) -> bool:
    """
    Check that a pitch to be rearticulated (repeated) is stable.

    :param counterpoint_continuation:
        current continuation of counterpoint line
    :param movement:
        melodic interval (in scale degrees) for line continuation
    :return:
        indicator whether a continuation is in accordance with the rule
    """
    if movement != 0:
        return True
    return counterpoint_continuation.scale_element.is_from_tonic_triad


def check_absence_of_stalled_pitches(
        movement: int,
        past_movements: List[int],
        max_n_repetitions: int = 2,
        **kwargs
) -> bool:
    """
    Check that a pitch is not excessively repeated.

    :param movement:
        melodic interval (in scale degrees) for line continuation
    :param past_movements:
        list of past movements
    :param max_n_repetitions:
        maximum allowed number of repetitions in a row
    :return:
        indicator whether a continuation is in accordance with the rule
    """
    if movement != 0:
        return True
    if len(past_movements) < max_n_repetitions - 1:
        return True
    changes = [x for x in past_movements[-max_n_repetitions+1:] if x != 0]
    return len(changes) > 0


def check_absence_of_monotonous_long_motion(
        counterpoint_continuation: 'LineElement',
        current_motion_start_element: 'LineElement',
        max_distance_in_semitones: int = 9,
        **kwargs
) -> bool:
    """
    Check that line does not move too far without any changes in direction.

    :param counterpoint_continuation:
        current continuation of counterpoint line
    :param current_motion_start_element:
        element of counterpoint line such that there are no
        changes in direction after it
    :param max_distance_in_semitones:
        maximum allowed distance (in semitones)
    :return:
        indicator whether a continuation is in accordance with the rule
    """
    current = counterpoint_continuation.scale_element.position_in_semitones
    start = current_motion_start_element.scale_element.position_in_semitones
    if abs(current - start) > max_distance_in_semitones:
        return False
    return True


def check_absence_of_skip_series(
        movement: int,
        past_movements: List[int],
        max_n_skips: int = 2,
        **kwargs
) -> bool:
    """
    Check that there are no long series of skips.

    :param movement:
        melodic interval (in scale degrees) for line continuation
    :param past_movements:
        list of past movements
    :param max_n_skips:
        maximum allowed number of skips in a row
    :return:
        indicator whether a continuation is in accordance with the rule
    """
    if abs(movement) <= 1:
        return True
    if len(past_movements) < max_n_skips:
        return True
    only_skips = all(abs(x) > 1 for x in past_movements[-max_n_skips:])
    return not only_skips


def check_that_skip_is_followed_by_opposite_step_motion(
        movement: int,
        past_movements: List[int],
        min_n_scale_degrees: int = 3,
        **kwargs
) -> bool:
    """
    Check that after a large skip there is a step motion in opposite direction.

    :param movement:
        melodic interval (in scale degrees) for line continuation
    :param past_movements:
        list of past movements
    :param min_n_scale_degrees:
        minimum size of a large enough skip (in scale degrees)
    :return:
        indicator whether a continuation is in accordance with the rule
    """
    if len(past_movements) == 0:
        return True
    previous_movement = past_movements[-1]
    if abs(previous_movement) < min_n_scale_degrees:
        return True
    return movement == -previous_movement / abs(previous_movement)


def check_resolution_of_submediant_and_leading_tone(
        line: List['LineElement'],
        movement: int,
        **kwargs
) -> bool:
    """
    Check that a sequence of submediant and leading tone properly resolves.

    If a line has submediant followed by leading tone, tonic must be used
    after leading tone, because there is strong attraction to it;
    similarly, if a line has leading tone followed by submediant,
    dominant must be used after submediant.

    :param line:
        counterpoint line in progress
    :param movement:
        melodic interval (in scale degrees) for line continuation
    :return:
        indicator whether a continuation is in accordance with the rule
    """
    if len(line) < 2:
        return True
    elif line[-1].scale_element.degree == 6 and line[-2].scale_element.degree == 7:
        return movement == -1
    elif line[-1].scale_element.degree == 7 and line[-2].scale_element.degree == 6:
        return movement == 1
    return True


def check_step_motion_to_final_pitch(
        counterpoint_continuation: 'LineElement',
        counterpoint_end: ScaleElement,
        piece_duration: int,
        prohibit_rearticulation: bool = True,
        **kwargs
) -> bool:
    """
    Check that there is a way to reach final pitch with step motion.

    :param counterpoint_continuation:
        current continuation of counterpoint line
    :param counterpoint_end:
        element that ends counterpoint line
    :param piece_duration:
        total duration of piece (in eighths)
    :param prohibit_rearticulation:
        if it is set to `True`, the last but one pitch can not be the same as
        the final pitch
    :return:
        indicator whether a continuation is in accordance with the rule
    """
    degrees_to_end_note = abs(
        counterpoint_continuation.scale_element.position_in_degrees
        - counterpoint_end.position_in_degrees
    )
    eighths_left = (
        (piece_duration - N_EIGHTHS_PER_MEASURE)
        - counterpoint_continuation.end_time_in_eighths
    )
    quarters_left = ceil(eighths_left / 2)
    if quarters_left == 0 and degrees_to_end_note == 0:
        return not prohibit_rearticulation
    return degrees_to_end_note <= quarters_left + 1


# Harmony rules.

def check_consonance_on_strong_beat(
        counterpoint_continuation: 'LineElement',
        cantus_firmus_elements: List['LineElement'],
        **kwargs
) -> bool:
    """
    Check that there is consonance if current beat is strong.

    :param counterpoint_continuation:
        current continuation of counterpoint line
    :param cantus_firmus_elements:
        list of elements from cantus firmus that sound simultaneously with
        the counterpoint element
    :return:
        indicator whether a continuation is in accordance with the rule
    """
    if counterpoint_continuation.start_time_in_eighths % 4 != 0:
        return True
    return check_consonance(
        counterpoint_continuation.scale_element,
        cantus_firmus_elements[0].scale_element
    )


def check_step_motion_to_dissonance(
        counterpoint_continuation: 'LineElement',
        cantus_firmus_elements: List['LineElement'],
        movement: int,
        **kwargs
) -> bool:
    """
    Check that there is step motion to a dissonating element.

    Note that this rule prohibits double neighboring tones.

    :param counterpoint_continuation:
        current continuation of counterpoint line
    :param cantus_firmus_elements:
        list of elements from cantus firmus that sound simultaneously with
        the counterpoint element
    :param movement:
        melodic interval (in scale degrees) for line continuation
    :return:
        indicator whether a continuation is in accordance with the rule
    """
    ctp_scale_element = counterpoint_continuation.scale_element
    cf_scale_element = cantus_firmus_elements[0].scale_element
    if check_consonance(ctp_scale_element, cf_scale_element):
        return True
    return movement in [-1, 1]


def check_step_motion_from_dissonance(
        movement: int,
        is_last_element_consonant: bool,
        **kwargs
) -> bool:
    """
    Check that there is step motion from a dissonating element.

    Note that this rule prohibits double neighboring tones.

    :param movement:
        melodic interval (in scale degrees) for line continuation
    :param is_last_element_consonant:
        indicator whether last element of counterpoint line (not including
        a new continuation in question) forms consonance with cantus firmus
    :return:
        indicator whether a continuation is in accordance with the rule
    """
    if is_last_element_consonant:
        return True
    return movement in [-1, 1]


def check_resolution_of_suspended_dissonance(
        line: List['LineElement'],
        movement: int,
        counterpoint_continuation: 'LineElement',
        cantus_firmus_elements: List['LineElement'],
        is_last_element_consonant: bool,
        **kwargs
) -> bool:
    """
    Check that suspended dissonance is resolved by downward step motion.

    :param line:
        counterpoint line in progress
    :param movement:
        melodic interval (in scale degrees) for line continuation
    :param counterpoint_continuation:
        current continuation of counterpoint line
    :param cantus_firmus_elements:
        list of elements from cantus firmus that sound simultaneously with
        the counterpoint element
    :param is_last_element_consonant:
        indicator whether last element of counterpoint line (not including
        a new continuation in question) forms consonance with cantus firmus
    :return:
        indicator whether a continuation is in accordance with the rule
    """
    last_note_start = line[-1].start_time_in_eighths
    last_note_end = line[-1].end_time_in_eighths
    last_note_duration = last_note_end - last_note_start
    if last_note_duration != N_EIGHTHS_PER_MEASURE:
        return True
    if is_last_element_consonant:
        return True
    if movement != -1:
        return False
    return check_consonance(
        counterpoint_continuation.scale_element,
        cantus_firmus_elements[-1].scale_element
    )


def check_absence_of_large_intervals(
        counterpoint_continuation: 'LineElement',
        cantus_firmus_elements: List['LineElement'],
        max_n_semitones: int = 16,
        **kwargs
) -> bool:
    """
    Check that there are no large intervals between adjacent pitches.

    :param counterpoint_continuation:
        current continuation of counterpoint line
    :param cantus_firmus_elements:
        list of elements from cantus firmus that sound simultaneously with
        the counterpoint element
    :param max_n_semitones:
        maximum allowed interval in semitones between two
        simultaneously sounding pitches
    :return:
        indicator whether a continuation is in accordance with the rule
    """
    cpt_pitch = counterpoint_continuation.scale_element.position_in_semitones
    for cantus_firmus_element in cantus_firmus_elements:
        cf_pitch = cantus_firmus_element.scale_element.position_in_semitones
        if abs(cpt_pitch - cf_pitch) > max_n_semitones:
            return False
    return True


def check_absence_of_lines_crossing(
        counterpoint_continuation: 'LineElement',
        cantus_firmus_elements: List['LineElement'],
        is_counterpoint_above: bool,
        prohibit_unisons: bool = True,
        **kwargs
) -> bool:
    """
    Check that there are no lines crossings.

    :param counterpoint_continuation:
        current continuation of counterpoint line
    :param cantus_firmus_elements:
        list of elements from cantus firmus that sound simultaneously with
        the counterpoint element
    :param is_counterpoint_above:
        indicator whether counterpoint must be above cantus firmus
    :param prohibit_unisons:
        if it is set to `True`, unison are considered a special case of
        lines crossing
    :return:
        indicator whether a continuation is in accordance with the rule
    """
    initial_sign = 1 if is_counterpoint_above else -1
    cpt_pitch = counterpoint_continuation.scale_element.position_in_semitones
    for cantus_firmus_element in cantus_firmus_elements:
        cf_pitch = cantus_firmus_element.scale_element.position_in_semitones
        if prohibit_unisons and cpt_pitch == cf_pitch:
            return False
        elif initial_sign * (cpt_pitch - cf_pitch) < 0:
            return False
    return True


def check_absence_of_overlapping_motion(
        counterpoint_continuation: 'LineElement',
        previous_cantus_firmus_element: 'LineElement',
        is_counterpoint_above: bool,
        **kwargs
) -> bool:
    """
    Check that there is no overlapping motion.

    :param counterpoint_continuation:
        current continuation of counterpoint line
    :param previous_cantus_firmus_element:
        the latest element of cantus firmus that sounds simultaneously
        with the last counterpoint element (excluding its continuation)
    :param is_counterpoint_above:
        indicator whether counterpoint must be above cantus firmus
    :return:
        indicator whether a continuation is in accordance with the rule
    """
    initial_sign = 1 if is_counterpoint_above else -1
    cpt_pitch = counterpoint_continuation.scale_element.position_in_semitones
    cf_pitch = previous_cantus_firmus_element.scale_element.position_in_semitones
    return initial_sign * (cpt_pitch - cf_pitch) > 0


# Registry.

def get_rules_registry() -> Dict[str, Callable]:
    """
    Get mapping from names to corresponding functions that check rules.

    :return:
        registry of functions checking rules of rhythm, voice leading,
        and harmony
    """
    registry = {
        # Rhythm rules:
        'rhythmic_pattern_validity': check_validity_of_rhythmic_pattern,
        # Voice leading rules:
        'rearticulation_stability': check_stability_of_rearticulated_pitch,
        'absence_of_stalled_pitches': check_absence_of_stalled_pitches,
        'absence_of_long_motion': check_absence_of_monotonous_long_motion,
        'absence_of_skip_series': check_absence_of_skip_series,
        'turn_after_skip': check_that_skip_is_followed_by_opposite_step_motion,
        'VI_VII_resolution': check_resolution_of_submediant_and_leading_tone,
        'step_motion_to_end': check_step_motion_to_final_pitch,
        # Harmony rules:
        'consonance_on_strong_beat': check_consonance_on_strong_beat,
        'step_motion_to_dissonance': check_step_motion_to_dissonance,
        'step_motion_from_dissonance': check_step_motion_from_dissonance,
        'resolution_of_suspended_dissonance': check_resolution_of_suspended_dissonance,
        'absence_of_large_intervals': check_absence_of_large_intervals,
        'absence_of_lines_crossing': check_absence_of_lines_crossing,
        'absence_of_overlapping_motion': check_absence_of_overlapping_motion,
    }
    return registry
