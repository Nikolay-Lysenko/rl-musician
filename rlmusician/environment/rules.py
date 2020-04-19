"""
Check compliance with some rules of rhythm, voice leading, and harmony.

Author: Nikolay Lysenko
"""


from typing import Callable, Dict, List, Optional

from rlmusician.utils.music_theory import check_consonance


N_EIGHTS_PER_MEASURE = 8


# Rhythm rules.

def check_validity_of_rhythmic_pattern(durations: List[int], **kwargs) -> bool:
    """
    Check that current measure is properly divided by notes.

    :param durations:
        durations (in eights) of all notes from a current measure
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
        line: List[Optional['LineElement']], measure: int, movement: int,
        **kwargs
) -> bool:
    """
    Check that a pitch to be rearticulated (repeated) is stable.

    :param line:
        melodic line as list of pitches
    :param measure:
        last finished measure in a line
    :param movement:
        melodic interval in scale degrees for line continuation
    :return:
        indicator whether a continuation is in accordance with the rule
    """
    if movement != 0:
        return True
    return line[measure].is_from_tonic_triad


def check_that_skip_leads_to_stable_pitch(
        line: List[Optional['LineElement']],
        line_elements: List['LineElement'],
        measure: int, movement: int, **kwargs
) -> bool:
    """
    Check that a skip (leap) leads to a stable pitch.

    :param line:
        melodic line as list of pitches
    :param line_elements:
        list of pitches available for the line
    :param measure:
        last finished measure in a line
    :param movement:
        melodic interval in scale degrees for line continuation
    :return:
        indicator whether a continuation is in accordance with the rule
    """
    if abs(movement) <= 1:
        return True
    next_position = line[measure].relative_position + movement
    next_element = line_elements[next_position]
    return next_element.is_from_tonic_triad


def check_that_skip_is_followed_by_opposite_step_motion(
        movement: int, previous_movements: List[int],
        min_n_scale_degrees: int = 3, **kwargs
) -> bool:
    """
    Check that after a large skip there is a step motion in opposite direction.

    :param movement:
        melodic interval in scale degrees for line continuation
    :param previous_movements:
        list of previous movements
    :param min_n_scale_degrees:
        minimum size of a large enough skip (in scale degrees)
    :return:
        indicator whether a continuation is in accordance with the rule
    """
    if len(previous_movements) == 0:
        return True
    previous_movement = previous_movements[-1]
    if abs(previous_movement) < min_n_scale_degrees:
        return True
    return movement == -previous_movement / abs(previous_movement)


def check_resolution_of_submediant_and_leading_tone(
        line: List[Optional['LineElement']], measure: int, movement: int,
        **kwargs
) -> bool:
    """
    Check that a sequence of submediant and leading tone properly resolves.

    If a line has submediant followed by leading tone, tonic must be used
    after leading tone, because there is strong attraction to it;
    similarly, if a line has leading tone followed by submediant,
    dominant must be used after submediant.

    :param line:
        melodic line as list of pitches
    :param measure:
        last finished measure in a line
    :param movement:
        melodic interval in scale degrees for line continuation
    :return:
        indicator whether a continuation is in accordance with the rule
    """
    if measure < 2:
        return True
    elif line[measure].degree == 6 and line[measure - 1].degree == 7:
        return movement == -1
    elif line[measure].degree == 7 and line[measure - 1].degree == 6:
        return movement == 1
    return True


def check_step_motion_to_final_pitch(
        line: List[Optional['LineElement']],
        line_elements: List['LineElement'],
        measure: int, movement: int, prohibit_rearticulation: bool = True,
        **kwargs
) -> bool:
    """
    Check that there is a way to reach final pitch with step motion.

    :param line:
        melodic line as list of pitches
    :param line_elements:
        list of pitches available for the line
    :param measure:
        last finished measure in a line
    :param movement:
        melodic interval in scale degrees for line continuation
    :param prohibit_rearticulation:
        if it is set to `True`, the last but one pitch can not be the same as
        the final pitch
    :return:
        indicator whether a continuation is in accordance with the rule
    """
    next_position = line[measure].relative_position + movement
    next_element = line_elements[next_position]
    next_degree = next_element.relative_position
    final_degree = line[-1].relative_position
    degrees_to_end_note = abs(next_degree - final_degree)
    measures_left = len(line) - measure - 2
    if measures_left == 1 and degrees_to_end_note == 0:
        return not prohibit_rearticulation
    return degrees_to_end_note <= measures_left


# Harmony rules.

def check_consonance_on_strong_beat(
        counterpoint_element: 'LineElement',
        cantus_firmus_elements: List['LineElement'],
        current_time: int,
        **kwargs
) -> bool:
    """
    Check that there is consonance if current beat is strong.

    :param counterpoint_element:
        current element of counterpoint line
    :param cantus_firmus_elements:
        list of elements from cantus firmus that sound simultaneously with
        the counterpoint element
    :param current_time:
        time of counterpoint element start (in eights)
    :return:
        indicator whether a continuation is in accordance with the rule
    """
    if current_time % 4 != 0:
        return True
    return check_consonance(counterpoint_element, cantus_firmus_elements[0])


def check_step_motion_to_dissonance(
        counterpoint_element: 'LineElement',
        cantus_firmus_elements: List['LineElement'],
        movement: int,
        **kwargs
) -> bool:
    """
    Check that there is step motion to a dissonating element.

    Note that this rule prohibits double neighboring tones.

    :param counterpoint_element:
        current element of counterpoint line
    :param cantus_firmus_elements:
        list of elements from cantus firmus that sound simultaneously with
        the counterpoint element
    :param movement:
        movement (in scale degrees) that continues counterpoint line
    :return:
        indicator whether a continuation is in accordance with the rule
    """
    ctp_scale_element = counterpoint_element.scale_element
    cf_scale_element = cantus_firmus_elements[0].scale_element
    if check_consonance(ctp_scale_element, cf_scale_element):
        return True
    return movement in [-1, 1]


def check_step_motion_from_dissonance(
        movement: int,
        is_last_element_dissonant: bool,
        **kwargs
) -> bool:
    """
    Check that there is step motion from a dissonating element.

    Note that this rule prohibits double neighboring tones.

    :param movement:
        movement (in scale degrees) that continues counterpoint line
    :param is_last_element_dissonant:
        indicator whether last element of counterpoint line (not including
        a new continuation in question) forms dissonance with cantus firmus
    :return:
        indicator whether a continuation is in accordance with the rule
    """
    if not is_last_element_dissonant:
        return True
    return movement in [-1, 1]


def check_resolution_of_suspended_dissonance(
        line: List['LineElement'],
        movement: int,
        counterpoint_element: 'LineElement',
        cantus_firmus_elements: List['LineElement'],
        is_last_element_dissonant: bool,
        **kwargs
) -> bool:
    """
    Check that suspended dissonance is resolved by downward step motion.

    :param line:
        counterpoint line
    :param movement:
        movement (in scale degrees) that continues counterpoint line
    :param counterpoint_element:
        current element of counterpoint line
    :param cantus_firmus_elements:
        list of elements from cantus firmus that sound simultaneously with
        the counterpoint element
    :param is_last_element_dissonant:
        indicator whether last element of counterpoint line (not including
        a new continuation in question) forms dissonance with cantus firmus
    :return:
        indicator whether a continuation is in accordance with the rule
    """
    last_duration = line[-1].end_time_in_eights - line[-1].start_time_in_eights
    if last_duration != N_EIGHTS_PER_MEASURE:
        return True
    if not is_last_element_dissonant:
        return True
    if movement != -1:
        return False
    return check_consonance(
        counterpoint_element.scale_element,
        cantus_firmus_elements[-1].scale_element
    )


def check_absence_of_large_intervals(
        counterpoint_element: 'LineElement',
        cantus_firmus_elements: List['LineElement'],
        max_n_semitones: int = 16,
        **kwargs
) -> bool:
    """
    Check that there are no large intervals between adjacent pitches.

    :param counterpoint_element:
        current element of counterpoint line
    :param cantus_firmus_elements:
        list of elements from cantus firmus that sound simultaneously with
        the counterpoint element
    :param max_n_semitones:
        maximum allowed interval in semitones between two
        simultaneously sounding pitches
    :return:
        indicator whether a continuation is in accordance with the rule
    """
    cpt_pitch = counterpoint_element.scale_element.position_in_semitones
    for cantus_firmus_element in cantus_firmus_elements:
        cf_pitch = cantus_firmus_element.scale_element.position_in_semitones
        if abs(cpt_pitch - cf_pitch) > max_n_semitones:
            return False
    return True


def check_absence_of_lines_crossing(
        counterpoint_element: 'LineElement',
        cantus_firmus_elements: List['LineElement'],
        is_counterpoint_above: bool,
        prohibit_unisons: bool = True,
        **kwargs
) -> bool:
    """
    Check that there are no lines crossings.

    :param counterpoint_element:
        current element of counterpoint line
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
    cpt_pitch = counterpoint_element.scale_element.position_in_semitones
    for cantus_firmus_element in cantus_firmus_elements:
        cf_pitch = cantus_firmus_element.scale_element.position_in_semitones
        if prohibit_unisons and cpt_pitch == cf_pitch:
            return False
        elif initial_sign * (cpt_pitch - cf_pitch) < 0:
            return False
    return True


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
        'rearticulation': check_stability_of_rearticulated_pitch,
        'destination_of_skip': check_that_skip_leads_to_stable_pitch,
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
    }
    return registry
