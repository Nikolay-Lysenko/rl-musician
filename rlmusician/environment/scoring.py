"""
Score a musical composition represented as a piano roll.

Author: Nikolay Lysenko
"""


from typing import Dict

import numpy as np


N_SEMITONES_PER_OCTAVE = 12


def score_palette_entropy(roll: np.ndarray) -> float:
    """
    Score composition based on entropy of usage of each note.

    It is a proxy of how non-trivial music is.

    :param roll:
        piano roll with rows corresponding to notes, columns corresponding to
        time steps, and cells containing zeros and ones and indicating
        whether a note is played
    :return:
        averaged over all notes variance of their usage in time
    """
    return np.mean(np.var(roll, axis=1))


def score_chord_entropy(roll: np.ndarray) -> float:
    """
    Score composition based on entropy of notes played at each time step.

    It is a proxy of how non-trivial music is.

    :param roll:
        piano roll with rows corresponding to notes, columns corresponding to
        time steps, and cells containing zeros and ones and indicating
        whether a note is played
    :return:
        averaged over all time steps variance of notes played there
    """
    return np.mean(np.var(roll, axis=0))


def shift_note_timeline(note_timeline: np.ndarray, shift: int) -> np.ndarray:
    """
    Shift note timeline.

    Length of output timeline is the same as length of input timeline,
    because non-fitting values are removed and gaps are padded with zeros.

    :param note_timeline:
        array of shape (1, n_time_steps) with cells containing
        zeros if the note is not played and ones if it is played
    :param shift:
        signed value of shift in time steps, positive for shift to the right
        and negative for shift to the left
    :return:
        shifted timeline
    """
    if shift == 0:
        return note_timeline
    elif shift > 0:
        return np.hstack((np.zeros((1, shift)), note_timeline[:, :-shift]))
    else:
        return np.hstack((note_timeline[:, -shift:], np.zeros((1, -shift))))


def compute_consonance_score_between_note_and_roll(
        note_timeline: np.ndarray,
        upper_roll: np.ndarray,
        interval_consonances: Dict[int, float]
) -> float:
    """
    Compute consonance score for a note's timeline and a roll of higher notes.

    It is a helper function for `score_consonances` function.

    :param note_timeline:
        array of shape (1, n_time_steps) with cells containing
        zeros if the note is not played and ones if it is played
    :param upper_roll:
        array of shape (n_higher_notes, n_time_steps) with rows corresponding
        to all notes that are higher than the note from `note_timeline`
    :param interval_consonances:
        mapping from interval in semitones to its score of consonance;
        keys must be all integers from 0 to 11, necessary number of octaves
        is subtracted for intervals bigger than octave
    :return:
        consonance score for the note and roll of all upper notes
    """
    intervals = ((note_timeline + upper_roll) > 1).astype(int)
    scores = [
        interval_consonances[n_semitones % N_SEMITONES_PER_OCTAVE]
        for n_semitones in range(intervals.shape[0], 0, -1)
    ]
    scores = np.array(scores).reshape((-1, 1))
    score = np.sum(scores * intervals)
    return score


def score_consonances(
        roll: np.ndarray,
        interval_consonances: Dict[int, float],
        distance_weights: Dict[int, float]
) -> float:
    """
    Score composition based on its consonances.

    It is a proxy of how pleasant to ear music is.

    :param roll:
        piano roll with rows corresponding to notes, columns corresponding to
        time steps, and cells containing zeros and ones and indicating
        whether a note is played
    :param interval_consonances:
        mapping from interval in semitones to its score of consonance;
        keys must be all integers from 0 to 11, necessary number of octaves
        is subtracted for intervals bigger than octave
    :param distance_weights:
        mapping from distance in time steps between played notes and
        relative weights for averaging consonance scores;
        if distance is absent, consonance score between these notes is not
        accounted
    :return:
        weighted sum of consonance scores with summation over pairs of
        played notes that are close enough (according to `distance_weights`)
    """
    score = 0
    distances = list(distance_weights.keys())
    signed_distances = set(distances + [-x for x in distances])
    distance_weights = {k: distance_weights[abs(k)] for k in signed_distances}
    for note_position in range(1, roll.shape[0]):
        upper_roll = roll[:note_position, :]
        note_timeline = roll[note_position, :].reshape((1, -1))
        for distance, weight in distance_weights.items():
            shifted_timeline = shift_note_timeline(note_timeline, distance)
            curr_score = compute_consonance_score_between_note_and_roll(
                shifted_timeline, upper_roll, interval_consonances
            )
            score += weight * curr_score
    return score
