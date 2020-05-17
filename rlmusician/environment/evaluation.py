"""
Evaluate a musical composition represented as a `Piece` instance.

Author: Nikolay Lysenko
"""


from collections import Counter
from typing import Any, Callable, Dict, Optional

import numpy as np
from scipy.stats import entropy

from rlmusician.environment.piece import Piece
from rlmusician.utils import rolling_aggregate


def evaluate_absence_of_looped_fragments(
        piece: Piece, min_size: int = 4, max_size: Optional[int] = None
) -> float:
    """
    Evaluate non-triviality of a piece based on absence of looped fragments.

    :param piece:
        `Piece` instance
    :param min_size:
        minimum duration of a fragment (in eighths)
    :param max_size:
        maximum duration of a fragment (in eighths)
    :return:
        multiplied by -1 number of looped fragments
    """
    score = 0
    max_size = max_size or piece.total_duration_in_eighths // 2
    for size in range(min_size, max_size + 1):
        max_position = piece.total_duration_in_eighths - 2 * size
        penultimate_measure_end = piece.total_duration_in_eighths - 8
        max_position = min(max_position, penultimate_measure_end - 1)
        for position in range(0, max_position + 1):
            fragment = piece.piano_roll[:, position:position+size]
            next_fragment = piece.piano_roll[:, position+size:position+2*size]
            if np.array_equal(fragment, next_fragment):
                score -= 1
    return score


def evaluate_entropy(piece: Piece) -> float:
    """
    Evaluate non-triviality of counterpoint line based on entropy.

    :param piece:
        `Piece` instance
    :return:
        normalized average over all lines entropy of pitches distribution
    """
    positions = [
        x.scale_element.position_in_degrees
        for x in piece.counterpoint
    ]
    counter = Counter(positions)
    lower_position = piece.lowest_element.position_in_degrees
    upper_position = piece.highest_element.position_in_degrees
    elements = piece.scale.elements[lower_position:upper_position + 1]
    distribution = [
        counter[element.position_in_degrees] / len(piece.counterpoint)
        for element in elements
    ]
    raw_score = entropy(distribution)
    max_entropy_distribution = [1 / len(elements) for _ in elements]
    denominator = entropy(max_entropy_distribution)
    score = raw_score / denominator
    return score


def evaluate_absence_of_narrow_ranges(
        piece: Piece, min_size: int = 9,
        penalties: Optional[Dict[int, float]] = None
) -> float:
    """
    Evaluate melodic fluency based on absence of narrow ranges.

    :param piece:
        `Piece` instance
    :param min_size:
        minimum size of narrow range (in line elements)
    :param penalties:
        mapping from width of a range (in scale degrees) to penalty
        applicable to ranges of not greater width
    :return:
        multiplied by -1 count of narrow ranges weighted based on their width
    """
    penalties = penalties or {2: 1, 3: 0.5}
    pitches = [x.scale_element.position_in_degrees for x in piece.counterpoint]
    rolling_mins = rolling_aggregate(pitches, min, min_size)[min_size-1:]
    rolling_maxs = rolling_aggregate(pitches, max, min_size)[min_size-1:]
    borders = zip(rolling_mins, rolling_maxs)
    score = 0
    for lower_border, upper_border in borders:
        range_width = upper_border - lower_border
        curr_penalties = [v for k, v in penalties.items() if k >= range_width]
        penalty = max(curr_penalties) if curr_penalties else 0
        score -= penalty
    return score


def evaluate_climax_explicity(
        piece: Piece,
        shortage_penalty: float = 0.3, duplication_penalty: float = 0.5
) -> float:
    """
    Evaluate goal-orientedness of counterpoint line based on climax explicity.

    :param piece:
        `Piece` instance
    :param shortage_penalty:
        penalty for each scale degree between declared highest pitch of a line
        and actual highest pitch of this line
    :param duplication_penalty:
        penalty for each non-first occurrence of line's highest pitch within
        this line
    :return:
        one minus all applicable penalties
    """
    max_position = piece.counterpoint[0].scale_element.position_in_degrees
    n_duplications = 0
    for line_element in piece.counterpoint[1:]:
        current_position = line_element.scale_element.position_in_degrees
        if current_position == max_position:
            n_duplications += 1
        elif current_position > max_position:
            max_position = current_position
            n_duplications = 0
    declared_max_position = piece.highest_element.position_in_degrees
    shortage = declared_max_position - max_position
    shortage_term = shortage_penalty * shortage
    duplication_term = duplication_penalty * n_duplications
    score = 1 - shortage_term - duplication_term
    return score


def evaluate_number_of_skips(
        piece: Piece, rewards: Optional[Dict[int, float]] = None
) -> float:
    """
    Evaluate interestingness/coherency of counterpoint based on skips number.

    :param piece:
        `Piece` instance
    :param rewards:
        mapping from number of skips to reward
    :return:
        reward assigned to balancing between interestingess and coherency
        of counterpoint line
    """
    rewards = rewards or {1: 0.8, 2: 0.9, 3: 1, 4: 0.9, 5: 0.5, 6: 0.25}
    n_skips = 0
    for movement in piece.past_movements:
        if abs(movement) > 1:
            n_skips += 1
    score = rewards.get(n_skips, 0)
    return score


def get_scoring_functions_registry() -> Dict[str, Callable]:
    """
    Get mapping from names of scoring functions to scoring functions.

    :return:
        registry of scoring functions
    """
    registry = {
        'looped_fragments': evaluate_absence_of_looped_fragments,
        'entropy': evaluate_entropy,
        'narrow_ranges': evaluate_absence_of_narrow_ranges,
        'climax_explicity': evaluate_climax_explicity,
        'number_of_skips': evaluate_number_of_skips,
    }
    return registry


def evaluate(
        piece: Piece,
        scoring_coefs: Dict[str, float],
        scoring_fn_params: Dict[str, Dict[str, Any]],
        verbose: bool = False
) -> float:
    """
    Evaluate piece.

    :param piece:
        `Piece` instance
    :param scoring_coefs:
        mapping from scoring function names to their weights in final score
    :param scoring_fn_params:
        mapping from scoring function names to their parameters
    :param verbose:
        if it is set to `True`, scores are printed with detailing by functions
    :return:
        weighted sum of scores returned by various scoring functions
    """
    score = 0
    registry = get_scoring_functions_registry()
    for fn_name, weight in scoring_coefs.items():
        fn = registry[fn_name]
        fn_params = scoring_fn_params.get(fn_name, {})
        curr_score = weight * fn(piece, **fn_params)
        if verbose:
            print(f'{fn_name:>30}: {curr_score}')  # pragma: no cover
        score += curr_score
    return score
