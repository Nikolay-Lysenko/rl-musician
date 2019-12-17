"""
Evaluate a musical composition represented as a `Piece` instance.

Author: Nikolay Lysenko
"""


import itertools
from typing import Any, Callable, Dict

import numpy as np

from rlmusician.environment.piece import Piece


N_SEMITONES_PER_OCTAVE = 12


def evaluate_absence_of_unisons(piece: Piece) -> float:
    """
    Evaluate distinguishability of lines based on absence of unisons.

    :param piece:
        `Piece` instance
    :return:
        multiplied by -1 share of unison intervals amongst all intervals
        from any measures except the first one and the last one
    """
    n_unisons = 0
    for first_line, second_line in itertools.combinations(piece.lines, 2):
        paired = zip(first_line[1:-1], second_line[1:-1])
        for first, second in paired:
            diff = first.absolute_position - second.absolute_position
            if diff % N_SEMITONES_PER_OCTAVE == 0:
                n_unisons += 1
    n_lines = len(piece.lines)
    n_intervals = n_lines * (n_lines - 1) / 2 * (piece.n_measures - 2)
    score = -n_unisons / n_intervals
    return score


def evaluate_lines_correlation(piece: Piece) -> float:
    """
    Evaluate independence of lines based on average pairwise correlation.

    :param piece:
        `Piece` instance
    :return:
        average correlation between lines multiplied by -1 and then
        rescaled to be from [0, 1]
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
        'absence_of_unisons': evaluate_absence_of_unisons,
        'lines_correlation': evaluate_lines_correlation,
    }
    return registry


def evaluate(
        piece: Piece,
        scoring_coefs: Dict[str, float],
        scoring_fn_params: Dict[str, Any],
        verbose: bool = False
) -> float:
    """
    Evaluate piano roll.

    :param piece:
        `Piece` instance
    :param scoring_coefs:
        mapping from scoring function names to their weights in final score
    :param scoring_fn_params:
        mapping from scoring function names to their parameters
    :param verbose:
        if it is set to `True`, scores by all functions are printed separately
    :return:
        overall score of piano roll quality
    """
    score = 0
    registry = get_scoring_functions_registry()
    for fn_name, weight in scoring_coefs.items():
        fn = registry[fn_name]
        curr_score = weight * fn(piece, **scoring_fn_params.get(fn_name, {}))
        if verbose:
            print(f'{fn_name:>25}: {curr_score}')  # pragma: no cover
        score += curr_score
    return score
