"""
Provide an agent that is trained with so called cross-entropy method.

In context of Monte-Carlo reinforcement learning, cross-entropy method
is a sort of genetic algorithm applied to parameters of actor model
(i.e., a model that maps observations to actions).

References:
    1) de Boer, Kroese, Mannor, and Rubinstein. A tutorial on the
    cross-entropy method. Annals of Operations Research, 2004.

Author: Nikolay Lysenko
"""


import os
from copy import deepcopy
from typing import Any, Dict, Callable, List, Optional

import gym
import numpy as np

from rlmusician.utils import map_in_parallel


def compute_model_properties(
        model_fn: Callable[..., 'keras.models.Model'],
        model_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute properties of model created by given function with given arguments.

    :param model_fn:
        function that returns model
    :param model_params:
        parameters that should be passed to the function that creates model
    :return:
        properties of model
    """
    model = model_fn(**model_params)
    shapes = [w.shape for w in model.get_weights()]
    sizes = [w.size for w in model.get_weights()]
    n_weights = sum(sizes)
    properties = {'shapes': shapes, 'sizes': sizes, 'n_weights': n_weights}
    return properties


def run_episode(actor_model: 'keras.models.Model', env: gym.Env) -> float:
    """
    Run an episode with given actor model.

    :param actor_model:
        model that maps observations to probabilities of actions
    :param env:
        environment
    return:
        reward for an episode
    """
    observation = env.reset()
    reward = None
    done = False
    while not done:
        observation = observation.reshape((1,) + observation.shape)
        probabilities = actor_model.predict(observation)[0]
        action = np.random.choice(len(probabilities), p=probabilities)
        observation, reward, done, info = env.step(action)
    return reward


def evaluate_random_candidate(
        agent: 'CrossEntropyAgent', env: gym.Env
) -> Dict[str, Any]:
    """
    Create candidate weights from current distribution and evaluate them.

    :param agent:
        agent
    :param env:
        environment
    :return:
        record with sampled weights and their score
    """
    # Reseed `np` in order to surely have independent results among processes.
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    epsilons = np.random.randn(agent.n_weights)
    flat_weights = agent.weights_std * epsilons + agent.weights_mean
    actor_model = agent.create_model(flat_weights)
    rewards = [
        run_episode(actor_model, env)
        for _ in range(agent.n_episodes_per_candidate)
    ]
    score = agent.aggregation_fn(rewards)
    entry = {'flat_weights': flat_weights, 'score': score}
    return entry


class CrossEntropyAgent:
    """An implementation of Cross-Entropy Method for deep RL."""

    def __init__(
            self,
            model_fn: Callable[..., 'keras.models.Model'],
            model_params: Dict[str, Any],
            population_size: int = 100,
            elite_fraction: float = 0.1,
            n_episodes_per_candidate: int = 10,
            aggregation_fn: str = 'mean',
            smoothing_coef: float = 0.25,
            initial_weights_mean: Optional[np.ndarray] = None,
            weights_std: float = 1,
            n_processes: Optional[int] = None
    ):
        """
        Initialize instance.

        :param model_fn:
            function that returns actor model (weights can be arbitrary)
        :param model_params:
            parameters that should be passed to the function that creates
            actor model
        :param population_size:
            number of candidate weights to draw and evaluate at each training
            step
        :param elite_fraction:
            share of best candidate weights that are used for training update
        :param n_episodes_per_candidate:
            number of episodes to play with each candidate weights
        :param aggregation_fn:
            name of function to aggregate rewards from multiple episodes into
            a single score of candidate weights ('min', 'mean', 'median',
            and 'max' are supported)
        :param smoothing_coef:
            coefficient of smoothing for updates of weights mean
        :param initial_weights_mean:
            mean of multivariate Gaussian distribution from which weights
            are drawn initially
        :param weights_std:
            standard deviation of all multivariate Gaussian distributions
            from which weights of candidates are drawn
        :param n_processes:
            number of processes for parallel candidate evaluation;
            by default, it is set to number of cores
        """
        self.model_fn = model_fn
        self.model_params = model_params

        # `keras` must not be imported in the main process.
        args = [(model_fn, model_params)]
        model_properties = map_in_parallel(compute_model_properties, args, 1)
        model_properties = model_properties[0]
        self.shapes = model_properties['shapes']
        self.sizes = model_properties['sizes']
        self.n_weights = model_properties['n_weights']

        self.weights_mean = initial_weights_mean or np.zeros(self.n_weights)
        self.weights_std = weights_std * np.ones(self.n_weights)

        self.smoothing_coef = smoothing_coef
        self.n_episodes_per_candidate = n_episodes_per_candidate
        self.aggregation_fn = self.__get_aggregation_fn(aggregation_fn)
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.n_top_candidates = round(elite_fraction * population_size)
        self.n_processes = n_processes

        self.model = None
        self.best = {'flat_weights': None, 'score': -np.inf}

    @staticmethod
    def __get_aggregation_fn(fn_name: str) -> Callable[[List[float]], float]:
        # Get function that aggregates rewards of candidate actor models.
        name_to_fn = {
            'min': min,
            'mean': np.mean,
            'median': np.median,
            'max': max
        }
        aggregation_fn = name_to_fn[fn_name]
        return aggregation_fn

    def create_model(self, flat_weights: np.ndarray) -> 'keras.models.Model':
        """
        Create new model with given weights.

        :param flat_weights:
            1D array of new weights
        :return:
            new model
        """
        model = self.model_fn(**self.model_params)
        weights = []
        position = 0
        for layer_shape, layer_size in zip(self.shapes, self.sizes):
            arr = flat_weights[position:(position + layer_size)]
            arr = arr.reshape(layer_shape)
            weights.append(arr)
            position += layer_size
        model.set_weights(weights)
        return model

    def fit(self, env: gym.Env, n_populations: int) -> None:
        """
        Train agent.

        :param env:
            environment
        :param n_populations:
            number of populations to be generated for update of candidates
            distribution and search of the best candidate
        :return:
            None
        """
        for i_population in range(n_populations):
            args = [(self, deepcopy(env)) for _ in range(self.population_size)]
            entries = map_in_parallel(
                evaluate_random_candidate, args, self.n_processes
            )
            sorted_entries = sorted(entries, key=lambda x: x['score'])
            top_entries = sorted_entries[-self.n_top_candidates:]
            top_flat_weights = [x['flat_weights'] for x in top_entries]
            top_flat_weights = np.vstack(top_flat_weights)
            self.weights_mean = (
                self.smoothing_coef * self.weights_mean
                + (1 - self.smoothing_coef) * np.mean(top_flat_weights, axis=0)
            )

            best_new_entry = top_entries[-1]
            if best_new_entry['score'] > self.best['score']:
                self.best = best_new_entry
            top_scores = [x['score'] for x in top_entries]
            avg_top_score = np.mean(top_scores)
            print(
                f"Population {i_population}: "
                f"mean score over top candidates is {avg_top_score}, "
                f"global best score is {self.best['score']}."
            )
        best_flat_weights = self.best['flat_weights']
        self.model = self.create_model(best_flat_weights)

    def test(self, env: gym.Env, n_episodes: int) -> None:
        """
        Run trained agent.

        :param env:
            environment
        :param n_episodes:
            number of episodes to run
        :return:
            None
        """
        for i_episode in range(n_episodes):
            reward = run_episode(self.model, env)
            env.render()
            print(f"Episode {i_episode}: reward is {reward}.")
