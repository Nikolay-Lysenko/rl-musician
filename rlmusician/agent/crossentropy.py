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


import random
from functools import reduce
from operator import mul
from typing import Any, Callable, Dict, List, Optional

import gym
import numpy as np
from keras.models import Model


class CrossEntropyAgentMemory:
    """Memory for Cross-Entropy Agent."""

    def __init__(self, size: int):
        """
        Initialize instance.

        :param size:
            maximum number of candidate actor models to keep
        """
        self.size: int = size
        self.position: int = 0
        self.data: List[Optional[Dict[str, Any]]] = [None for _ in range(size)]
        self.best: Dict[str, Any] = {'flat_weights': None, 'score': -np.inf}

    def add(self, flat_weights: np.ndarray, score: float) -> None:
        """
        Add candidate actor model and its score to the memory.

        :param flat_weights:
            actor model represented as its flattened weights
        :param score:
            score for the candidate actor model; the higher, the better
        :return:
            None
        """
        entry = {'flat_weights': flat_weights, 'score': score}
        self.data[self.position % self.size] = entry
        self.position += 1
        if score > self.best['score']:
            self.best = entry

    def sample(self, n_entries: int) -> List[Dict[str, Any]]:
        """
        Extract some random candidates and their scores from memory.

        :param n_entries:
            number of candidates to extract
        :return:
            random candidates and their scores
        """
        is_memory_full = self.data[self.size - 1] is not None
        total_n_entries = self.size if is_memory_full else self.position
        if n_entries > total_n_entries:
            raise RuntimeError(
                f"There are only {total_n_entries} entries in memory, "
                f"but {n_entries} entries are requested. "
            )
        indices = random.sample(range(total_n_entries), n_entries)
        sampled_entries = [self.data[x] for x in indices]
        return sampled_entries


class CrossEntropyAgent:
    """An implementation of Cross-Entropy Method for deep RL."""

    def __init__(
            self,
            model: Model,
            population_size: int = 100,
            elite_fraction: float = 0.1,
            n_episodes_per_candidate: int = 10,
            aggregation_fn: str = 'mean',
            n_candidates_to_keep: Optional[int] = None,
            initial_weights_mean: Optional[np.ndarray] = None,
            weights_std: float = 1,
            n_warmup_candidates: int = 0
    ):
        """
        Initialize instance.

        :param model:
            actor model; its weights are ignored, only its architecture is used
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
        :param n_candidates_to_keep:
            number of last candidate weights of actor model to keep in memory
        :param initial_weights_mean:
            mean of multivariate Gaussian distribution from which weights
            are drawn initially
        :param weights_std:
            standard deviation of all multivariate Gaussian distributions
            from which weights of candidates are drawn
        :param n_warmup_candidates:
            number of random candidate weights to evaluate before training
        """
        self.model = model
        self.shapes = [w.shape for w in model.get_weights()]
        self.sizes = [w.size for w in model.get_weights()]
        self.n_weights = sum(self.sizes)
        self.weights_mean = initial_weights_mean or np.zeros(self.n_weights)
        self.weights_std = weights_std * np.ones(self.n_weights)

        n_candidates_to_keep = n_candidates_to_keep or population_size
        self.memory = CrossEntropyAgentMemory(n_candidates_to_keep)
        self.n_episodes_per_candidate = n_episodes_per_candidate
        self.aggregation_fn = self.__get_aggregation_fn(aggregation_fn)
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.n_top_candidates = round(elite_fraction * population_size)

        self.n_warmup_candidates = n_warmup_candidates

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

    def __set_weights(self, flat_weights: np.ndarray) -> None:
        # Set weights of actor model.
        weights = []
        position = 0
        for layer_shape in self.shapes:
            layer_size = reduce(mul, layer_shape)
            arr = flat_weights[position:(position + layer_size)]
            arr = arr.reshape(layer_shape)
            weights.append(arr)
            position += layer_size
        self.model.set_weights(weights)

    def __choose_action(self, observation: np.ndarray) -> int:
        # Choose an action stochastically.
        observation = observation.reshape((1,) + observation.shape)
        probabilities = self.model.predict(observation)[0]
        action = np.random.choice(len(probabilities), p=probabilities)
        return action

    def __run_episode(self, env: gym.Env) -> float:
        # Run episode with current actor model.
        observation = env.reset()
        reward = None
        done = False
        while not done:
            action = self.__choose_action(observation)
            observation, reward, done, info = env.step(action)
        return reward

    def __evaluate_random_candidate(self, env: gym.Env) -> None:
        # Create candidate from current distribution and evaluate it.
        epsilons = np.random.randn(self.n_weights)
        flat_weights = self.weights_std * epsilons + self.weights_mean
        self.__set_weights(flat_weights)
        rewards = [
            self.__run_episode(env)
            for _ in range(self.n_episodes_per_candidate)
        ]
        score = self.aggregation_fn(rewards)
        self.memory.add(flat_weights, score)

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
        for _ in range(self.n_warmup_candidates):
            self.__evaluate_random_candidate(env)
        for i_population in range(n_populations):
            for _ in range(self.population_size):
                self.__evaluate_random_candidate(env)
            entries = self.memory.sample(self.population_size)
            sorted_entries = sorted(entries, key=lambda x: x['score'])
            top_entries = sorted_entries[-self.n_top_candidates:]
            top_flat_weights = [x['flat_weights'] for x in top_entries]
            top_flat_weights = np.vstack(top_flat_weights)
            self.weights_mean = np.mean(top_flat_weights, axis=0)

            top_scores = [x['score'] for x in top_entries]
            avg_top_score = np.mean(top_scores)
            print(
                f"Population {i_population}: "
                f"mean score over top candidates is {avg_top_score}, "
                f"global best score is {self.memory.best['score']}."
            )
        best_flat_weights = self.memory.best['flat_weights']
        self.__set_weights(best_flat_weights)

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
            reward = self.__run_episode(env)
            env.render()
            print(f"Episode {i_episode}: reward is {reward}.")
