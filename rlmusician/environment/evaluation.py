"""
Evaluate a musical composition represented as a `Piece` instance.

Author: Nikolay Lysenko
"""


import itertools
import warnings
from collections import Counter
from typing import Any, Callable, Dict

import numpy as np
from scipy.stats import entropy

from rlmusician.environment.piece import Piece


N_SEMITONES_PER_OCTAVE = 12


def evaluate_autocorrelation(piece: Piece, max_lag: int = 8) -> float:
    """
    Evaluate non-triviality of a piece based on pitch-wise autocorrelation.

    :param piece:
        `Piece` instance
    :param max_lag:
        maximum lag to consider
    :return:
        multiplied by -1 and then rescaled to be from [0, 1]
        maximum over all lags average pitch-wise absolute autocorrelation
    """
    lag_scores = []
    lags = range(2, max_lag + 1)
    for lag in lags:
        first = piece.piano_roll[:, :-lag]
        second = piece.piano_roll[:, lag:]
        corr_matrix = np.corrcoef(first, second)
        offset = corr_matrix.shape[0] // 2
        row_wise_correlations = [
            np.abs(corr_matrix[i, i + offset]) for i in range(offset)
        ]
        # Here, `nan` values can occur for pitches that are out of scale
        # (it is correct to ignore them), constantly played pitches (it is
        # correct to ignore them too and also they are extremely rare),
        # and pitches that are played in `first` only or in `second` only
        # (it is not clear, is it correct to ignore such pitches or not).
        lag_score = np.nanmean(row_wise_correlations)
        if np.isnan(lag_score):
            lag_score = 0  # Do not allow this lag to affect results.
        lag_scores.append(lag_score)
    score = 1 - max(lag_scores)
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
            diff = first.absolute_position - second.absolute_position
            if diff == 0:
                n_clashes += 1
            elif diff % N_SEMITONES_PER_OCTAVE == 0:
                n_unisons += 1
    n_lines = len(piece.lines)
    n_intervals = n_lines * (n_lines - 1) / 2 * (piece.n_measures - 2)
    clash_score = pure_clash_coef * n_clashes / n_intervals
    unison_score = n_unisons / n_intervals
    score = -(clash_score + unison_score)
    return score


def evaluate_independence_of_motion(
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
    for first_line, second_line in itertools.combinations(piece.lines, 2):
        prev_first = first_line[0]
        prev_second = second_line[0]
        paired = zip(first_line[1:], second_line[1:])
        for first, second in paired:
            first_diff = first.relative_position - prev_first.relative_position
            second_diff = second.relative_position - prev_second.relative_position
            if first_diff == second_diff:
                score += parallel_coef
            elif first_diff * second_diff > 0:
                score += similar_coef
            elif first_diff * second_diff == 0:
                score += oblique_coef
            else:
                score += contrary_coef
            prev_first = first
            prev_second = second
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


def get_scoring_functions_registry() -> Dict[str, Callable]:
    """
    Get mapping from names of scoring functions to scoring functions.

    :return:
        registry of scoring functions
    """
    registry = {
        'autocorrelation': evaluate_autocorrelation,
        'absence_of_pitch_class_clashes': evaluate_absence_of_pitch_class_clashes,
        'independence_of_motion': evaluate_independence_of_motion,
        'lines_correlation': evaluate_lines_correlation,
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
