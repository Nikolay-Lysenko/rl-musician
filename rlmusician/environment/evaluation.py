"""
Evaluate a musical composition represented as a `Piece` instance.

Author: Nikolay Lysenko
"""


import itertools
import warnings
from collections import Counter
from typing import Any, Callable, Dict, Optional

import numpy as np
from scipy.stats import entropy

from rlmusician.environment.piece import Piece


def evaluate_absence_of_looped_pitches(
        piece: Piece, max_n_repetitions: int = 2
) -> float:
    """
    Evaluate non-triviality of a piece based on absence of looped pitches.

    :param piece:
        `Piece` instance
    :param max_n_repetitions:
        maximum number of repetitions of the same pitch in a row that is not
        penalized
    :return:
        multiplied by -1 number of repetitions that exceed limit
    """
    score = 0
    for line in piece.lines:
        previous_pitch = line[0].absolute_position
        n_repetitions = 1
        for element in line[1:]:
            current_pitch = element.absolute_position
            if current_pitch == previous_pitch:
                n_repetitions += 1
            else:
                score -= max(n_repetitions - max_n_repetitions, 0)
                previous_pitch = current_pitch
                n_repetitions = 1
    return score


def evaluate_absence_of_looped_fragments(
        piece: Piece, min_size: int = 1, max_size: Optional[int] = None
) -> float:
    """
    Evaluate non-triviality of a piece based on absence of looped fragments.

    :param piece:
        `Piece` instance
    :param min_size:
        minimum duration of a fragment (in measures)
    :param max_size:
        maximum duration of a fragment (in measures)
    :return:
        multiplied by -1 number of looped fragments
    """
    score = 0
    max_size = max_size or piece.n_measures // 2
    for size in range(min_size, max_size + 1):
        for position in range(0, piece.n_measures - 2 * size + 1):
            fragment = piece.piano_roll[:, position:position+size]
            next_fragment = piece.piano_roll[:, position+size:position+2*size]
            if np.array_equal(fragment, next_fragment):
                score -= 1
    return score


def evaluate_entropy(piece: Piece) -> float:
    """
    Evaluate non-triviality of a piece based on entropy of pitch distribution.

    :param piece:
        `Piece` instance
    :return:
        normalized average over all lines entropy of pitches distribution
    """
    scores = []
    for line, elements in zip(piece.lines, piece.line_elements):
        positions = [element.relative_position for element in line]
        counter = Counter(positions)
        distribution = [
            counter[element.relative_position] / piece.n_measures
            for element in elements
        ]
        raw_entropy = entropy(distribution)
        max_entropy_distribution = [1 / len(elements) for _ in elements]
        denominator = entropy(max_entropy_distribution)
        normalized_entropy = raw_entropy / denominator
        scores.append(normalized_entropy)
    score = sum(scores) / len(scores)
    return score


def evaluate_absence_of_pitch_class_clashes(
        piece: Piece, pure_clash_coef: float = 3
) -> float:
    """
    Evaluate distinguishability of lines based on absence of clashes.

    Here, pitch class clashes are divided into two groups:
    1) pure clashes where exactly the same pitch is played in two lines,
    2) unison intervals where pitch class is the same,
       but pitches are different.

    :param piece:
        `Piece` instance
    :param pure_clash_coef:
        coefficient of pure clashes penalization relative to unison intervals
    :return:
        normalized and multiplied by -1 weighted count of clashing intervals
        amongst all intervals from any measures except the first one
        and the last one
    """
    n_clashes = 0
    n_unisons = 0
    for first_line, second_line in itertools.combinations(piece.lines, 2):
        paired = zip(first_line[1:-1], second_line[1:-1])
        for first, second in paired:
            if first.absolute_position == second.absolute_position:
                n_clashes += 1
            elif first.degree == second.degree:
                n_unisons += 1
    n_lines = len(piece.lines)
    n_intervals = n_lines * (n_lines - 1) / 2 * (piece.n_measures - 2)
    clash_score = pure_clash_coef * n_clashes / n_intervals
    unison_score = n_unisons / n_intervals
    score = -(clash_score + unison_score)
    return score


def evaluate_motion_by_types(
        piece: Piece, parallel_coef: float, similar_coef: float,
        oblique_coef: float, contrary_coef: float
) -> float:
    """
    Evaluate distinguishability of lines based on their motion.

    To see definitions of motion types, look here:
    https://en.wikipedia.org/wiki/Contrapuntal_motion

    :param piece:
        `Piece` instance
    :param parallel_coef:
        coefficient for parallel motion
    :param similar_coef:
        coefficient for similar, but not parallel motion
    :param oblique_coef:
        coefficient for oblique motion
    :param contrary_coef:
        coefficient for contrary motion
    :return:
        normalized weighted sum of scores granted for each movement
    """
    score = 0
    pairs = itertools.combinations(piece.passed_movements, 2)
    for first_movements, second_movements in pairs:
        paired = zip(first_movements, second_movements)
        for first_movement, second_movement in paired:
            if first_movement == second_movement:
                score += parallel_coef
            elif first_movement * second_movement > 0:
                score += similar_coef
            elif first_movement * second_movement == 0:
                score += oblique_coef
            else:
                score += contrary_coef
    n_lines = len(piece.lines)
    n_intervals = n_lines * (n_lines - 1) / 2 * (piece.n_measures - 1)
    score /= n_intervals
    return score


def evaluate_lines_correlation(piece: Piece) -> float:
    """
    Evaluate distinguishability of lines based on average pairwise correlation.

    :param piece:
        `Piece` instance
    :return:
        multiplied by -1 and then rescaled to be from [0, 1]
        average correlation between lines
    """
    correlations = []
    for first_line, second_line in itertools.combinations(piece.lines, 2):
        first_pitches = [element.absolute_position for element in first_line]
        second_pitches = [element.absolute_position for element in second_line]
        correlation = np.corrcoef(first_pitches, second_pitches)[0, 1].item()
        if np.isnan(correlation):  # At least one line is constant.
            correlation = 1
        correlations.append(correlation)
    score = (1 - sum(correlations) / len(correlations)) / 2
    return score


def evaluate_climax_explicity(
        piece: Piece,
        shortage_penalty: float = 0.3, duplication_penalty: float = 0.5
) -> float:
    """
    Evaluate goal-orientedness of lines motion based on climax explicity.

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
    scores = []
    for line, all_elements in zip(piece.lines, piece.line_elements):
        declared_max_position = len(all_elements) - 1
        current_max_position = line[0].relative_position
        current_n_duplications = 0
        for element in line[1:]:
            if element.relative_position == current_max_position:
                current_n_duplications += 1
            elif element.relative_position > current_max_position:
                current_max_position = element.relative_position
                current_n_duplications = 0
        shortage = declared_max_position - current_max_position
        shortage_term = shortage_penalty * shortage
        duplication_term = duplication_penalty * current_n_duplications
        score = 1 - shortage_term - duplication_term
        scores.append(score)
    score = sum(scores) / len(scores)
    return score


def evaluate_number_of_skips(
        piece: Piece, min_n_skips: int = 1, max_n_skips: int = 3
) -> float:
    """
    Evaluate interestingness of lines based on number of skips in them.

    :param piece:
        `Piece` instance
    :param min_n_skips:
        minimum number of skips for a line to be interesting
    :param max_n_skips:
        maximum number of skips for a line to be still coherent
    :return:
        share of lines where number of skips lies within specified range
    """
    scores = []
    for movements in piece.passed_movements:
        n_skips = 0
        for movement in movements:
            if abs(movement) > 1:
                n_skips += 1
        scores.append(1 if min_n_skips <= n_skips <= max_n_skips else 0)
    score = sum(scores) / len(scores)
    return score


def evaluate_absence_of_downward_skips(
        piece: Piece, size_penalty_power: float = 2.0
) -> float:
    """
    Evaluate presence of downward step motion due to absence of downward skips.

    :param piece:
        `Piece` instance
    :param size_penalty_power:
        strength of large downward skis penalization relatively to smaller ones
    :return:
        sum of penalties assigned to each downward skip
    """
    score = 0
    for movements in piece.passed_movements:
        for movement in movements:
            if movement < -1:
                score -= (-movement) ** size_penalty_power
    return score


def get_scoring_functions_registry() -> Dict[str, Callable]:
    """
    Get mapping from names of scoring functions to scoring functions.

    :return:
        registry of scoring functions
    """
    registry = {
        'looped_pitches': evaluate_absence_of_looped_pitches,
        'looped_fragments': evaluate_absence_of_looped_fragments,
        'entropy': evaluate_entropy,
        'pitch_class_clashes': evaluate_absence_of_pitch_class_clashes,
        'types_of_motion': evaluate_motion_by_types,
        'lines_correlation': evaluate_lines_correlation,
        'climax_explicity': evaluate_climax_explicity,
        'number_of_skips': evaluate_number_of_skips,
        'downward_skips': evaluate_absence_of_downward_skips,
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            curr_score = weight * fn(piece, **fn_params)
        if verbose:
            print(f'{fn_name:>30}: {curr_score}')  # pragma: no cover
        score += curr_score
    return score
