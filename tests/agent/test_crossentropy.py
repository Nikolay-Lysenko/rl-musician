"""
Test `rlmusician.agent.crossentropy` module.

Author: Nikolay Lysenko
"""


from typing import Callable, List

import pytest

from rlmusician.agent.crossentropy import optimize_with_cem


def function_to_be_optimized(args: List[float]) -> float:
    """Be optimized as an example of multivariate function."""
    return 1 - args[0]**2 - args[1]**2


@pytest.mark.parametrize(
    "target_fn, n_populations, population_size, elite_fraction, "
    "smoothing_coef, initial_mean, std, n_trials_per_candidate, "
    "optimum, expected_mse_threshold",
    [
        (
            # `target_fn`
            function_to_be_optimized,
            # `n_populations`
            15,
            # `population_size`
            100,
            # `elite_fraction`
            0.1,
            # `smoothing_coef`
            0.1,
            # `initial_mean`
            [1, 1],
            # `std`
            0.1,
            # `n_trials_per_candidate`
            1,
            # `optimum`
            [0, 0],
            # `expected_mse_threshold`
            0.01
        )
    ]
)
def test_optimize_with_cem(
        target_fn: Callable[[List[float]], float],
        n_populations: int,
        population_size: int,
        elite_fraction: float,
        smoothing_coef: float,
        initial_mean: List[int],
        std: float,
        n_trials_per_candidate: int,
        optimum: List[float],
        expected_mse_threshold: float
) -> None:
    """Test `optimize_with_cem` function."""
    result = optimize_with_cem(
        target_fn, n_populations, population_size, elite_fraction,
        smoothing_coef, initial_mean, std, n_trials_per_candidate,
        n_processes=1
    )
    mse = sum(
        (calculated - expected)**2
        for calculated, expected in zip(result, optimum)
    )
    assert mse < expected_mse_threshold
