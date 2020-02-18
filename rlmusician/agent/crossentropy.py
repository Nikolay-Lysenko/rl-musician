"""
Implement Cross-Entropy Method (CEM) for optimization.

References:
    1) de Boer, Kroese, Mannor, and Rubinstein. A tutorial on the
    cross-entropy method. Annals of Operations Research, 2005.

Author: Nikolay Lysenko
"""


import os
from typing import Any, Dict, Callable, List, Optional

import numpy as np

from rlmusician.utils import map_in_parallel


def estimate_at_random_point(
        target_fn: Callable[[np.ndarray], float],
        mean: np.ndarray,
        std: float,
        n_trials: int,
        aggregation_fn: Callable[[List[float]], float],
        target_fn_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate point from given distribution and estimate function at it.

    :param target_fn:
        function to be estimated at a random point (it can be stochastic;
        if so, its aggregated value is estimated)
    :param mean:
        mean of multivariate Gaussian distribution from which random point
        is drawn
    :param std:
        standard deviation of multivariate Gaussian distribution
        from which random point is drawn
    :param n_trials:
        number of runs for function estimation (it makes sense to set it
        to more than 1 only if `target_fn` is stochastic)
    :param aggregation_fn:
        name of function that aggregates results from multiple trials into
        a single value ('min', 'mean', 'median', and 'max' are supported)
    :param target_fn_kwargs:
        additional keyword arguments for `target_fn`
    :return:
        record with sampled point and aggregated value of `target_fn` at it
    """
    # Reseed `np` in order to surely have independent results among processes.
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    epsilons = np.random.randn(len(mean))
    params = mean + std * epsilons
    target_fn_kwargs = target_fn_kwargs or {}
    results = [target_fn(params, **target_fn_kwargs) for _ in range(n_trials)]
    score = aggregation_fn(results)
    entry = {'params': params, 'score': score}
    return entry


def optimize_with_cem(
        target_fn: Callable[[np.ndarray], float],
        n_populations: int,
        population_size: int,
        elite_fraction: float,
        smoothing_coef: float,
        initial_mean: np.ndarray,
        std: float,
        n_trials: int,
        aggregation_fn: str = 'mean',
        target_fn_kwargs: Optional[Dict[str, Any]] = None,
        parallelling_params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Optimize with Cross-Entropy Method (CEM).

    :param target_fn:
        function to be maximized (it can be stochastic; if so, its
        aggregated value is maximized)
    :param n_populations:
        number of training steps at each of which a pool of candidates is
        evaluated, best candidates are selected, and sampling distribution
        is updated based on them
    :param population_size:
        number of candidates to draw and evaluate at each training step
    :param elite_fraction:
        share of best candidates that are used for training update
    :param smoothing_coef:
        coefficient of smoothing for updates of candidates mean
    :param initial_mean:
        mean of multivariate Gaussian distribution from which candidates
        are drawn initially
    :param std:
        standard deviation of all multivariate Gaussian distributions
        from which candidates are drawn (for each training step, there is its
        own distribution with its own mean)
    :param n_trials:
        number of runs for each candidate evaluation (it makes sense to set it
        to more than 1 only if `target_fn` is stochastic)
    :param aggregation_fn:
        name of function that aggregates results from multiple trials into
        a single score of candidate ('min', 'mean', 'median', and 'max'
        are supported)
    :param target_fn_kwargs:
        additional keyword arguments for `target_fn`
    :param parallelling_params:
        settings of parallel evaluation of candidates;
        by default, number of processes is set to number of cores
        and each worker is not replaced with a newer one after some number of
        tasks are finished
    :return:
        found optimal parameters
    """
    name_to_fn = {
        'min': min,
        'mean': np.mean,
        'median': np.median,
        'max': max
    }
    agg_fn = name_to_fn[aggregation_fn]
    n_top_candidates = round(elite_fraction * population_size)

    current_mean = initial_mean
    best = {'params': initial_mean, 'score': -np.inf}
    print("population | avg_score_over_current_top |   global_best_score")
    for i_population in range(n_populations):
        args = [
            (target_fn, current_mean, std, n_trials, agg_fn, target_fn_kwargs)
            for _ in range(population_size)
        ]
        pool_kwargs = parallelling_params or {}
        entries = map_in_parallel(estimate_at_random_point, args, pool_kwargs)
        sorted_entries = sorted(entries, key=lambda x: x['score'])
        top_entries = sorted_entries[-n_top_candidates:]
        top_params = [x['params'] for x in top_entries]
        top_params = np.vstack(top_params)
        current_mean = (
            smoothing_coef * current_mean
            + (1 - smoothing_coef) * np.mean(top_params, axis=0)
        )

        new_best = top_entries[-1]
        if new_best['score'] > best['score']:
            best = new_best
        top_scores = [x['score'] for x in top_entries]
        avg_top_score = np.mean(top_scores)
        res = f"{i_population:>10} | {avg_top_score:>26} | {best['score']:>19}"
        print(res)
    found_params = best['params']
    return found_params
