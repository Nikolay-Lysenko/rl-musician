"""
Do auxiliary tasks.

Author: Nikolay Lysenko
"""


import tempfile
from typing import Any, Dict, Tuple

import numpy as np


def measure_compressed_size(arr: np.ndarray) -> int:
    """
    Measure size (in bytes) of compressed array.

    :param arr:
        array
    :return:
        size (in bytes) of compressed array
    """
    with tempfile.TemporaryFile() as tmp_file:
        np.savez_compressed(tmp_file, arr)
        tmp_file.seek(0, 2)  # Move current position to the end of file.
        size_of_compressed_arr = tmp_file.tell()
        return size_of_compressed_arr


def estimate_max_compressed_size_given_shape(
        shape: Tuple[int, int], n_trials: int = 100
) -> float:
    """
    Estimate maximum compressed size of binary array of given shape.

    :param shape:
        shape of binary array
    :param n_trials:
        number of trials to draw array of high entropy
    :return:
        estimation of maximum compressed size (in bytes)
    """
    max_size = max(
        measure_compressed_size(np.random.binomial(1, 0.5, shape))
        for _ in range(n_trials)
    )
    return max_size


def add_reference_size_for_repetitiveness(
        settings: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Add 'reference_size' parameter for scoring repetitiveness (if needed).

    :param settings:
        configuration of an experiment
    :return:
        modified configuration of an experiment
    """
    if 'repetitiveness' in settings['environment']['scoring_coefs']:
        roll_shape = (
            settings['environment']['n_semitones'],
            settings['environment']['n_roll_steps']
        )
        reference_size = estimate_max_compressed_size_given_shape(roll_shape)
        dct = settings['environment']['scoring_fn_params'].get(
            'repetitiveness', {}
        )
        dct['reference_size'] = reference_size
        settings['environment']['scoring_fn_params']['repetitiveness'] = dct
    return settings
