"""
Create agent for interacting with `CounterpointEnv`.

Author: Nikolay Lysenko.
"""


from typing import Any, Callable, Dict, List

import numpy as np
from scipy.special import softmax

from rlmusician.environment import CounterpointEnv
from rlmusician.utils import convert_to_base, map_in_parallel


class CounterpointEnvAgent:
    """Agent for interacting with `CounterpointEnv`."""

    def __init__(
            self,
            model_fn: Callable[..., 'keras.models.Model'],
            observation_len: int,
            n_lines: int,
            n_movements_per_line: int,
            hidden_layer_size: int
    ):
        """
        Initialize instance.

        :param model_fn:
            function that creates actor model
        :param observation_len:
            length of observation returned by environment
        :param n_lines:
            number of lines in a piece to be created within environment
        :param n_movements_per_line:
            maximum number of line movements supported by environment
        :param hidden_layer_size:
            size of actor model's hidden layer
        """
        self.model_fn = model_fn
        self.observation_len = observation_len
        self.n_lines = n_lines
        self.n_movements_per_line = n_movements_per_line
        self.hidden_layer_size = hidden_layer_size

        self.input_size = observation_len + n_lines * n_movements_per_line
        self.model = model_fn((self.input_size,), hidden_layer_size)
        self.shapes = [w.shape for w in self.model.get_weights()]
        self.sizes = [w.size for w in self.model.get_weights()]
        self.n_weights = sum(self.sizes)

    def create_candidates(
            self, observation: np.ndarray, actions: List[int]
    ) -> np.ndarray:
        """
        Create batch of candidate actions' representation for actor model.

        :param observation:
            observation returned by environment
        :param actions:
            list of actions allowed at the next step
        """
        actions_part_width = self.n_lines * self.n_movements_per_line
        actions_part = np.zeros((len(actions), actions_part_width))
        shifts = [i * self.n_movements_per_line for i in range(self.n_lines)]
        for row_number, action in enumerate(actions):
            raw_encoded_action = convert_to_base(
                action, self.n_movements_per_line, self.n_lines
            )
            encoded_action = [
                idx + shift
                for idx, shift in zip(raw_encoded_action, shifts)
            ]
            actions_part[row_number, encoded_action] = 1
        observation_part = np.tile(observation, (len(actions), 1))
        candidates = np.hstack((observation_part, actions_part))
        return candidates

    def set_weights(self, flat_weights: np.ndarray) -> None:
        """
        Set weights of model.

        :param flat_weights:
            list or 1D array of new weights
        :return:
            None
        """
        weights = []
        position = 0
        for layer_shape, layer_size in zip(self.shapes, self.sizes):
            arr = flat_weights[position:(position + layer_size)]
            arr = arr.reshape(layer_shape)
            weights.append(arr)
            position += layer_size
        self.model.set_weights(weights)

    def run_episode(self, env: CounterpointEnv) -> float:
        """
        Run an episode.

        :param env:
            environment
        return:
            reward for the episode
        """
        observation = env.reset()
        reward = None
        done = False
        valid_actions = env.valid_actions
        while not done:
            candidates = self.create_candidates(observation, valid_actions)
            probabilities = self.model.predict(candidates)
            probabilities = softmax(probabilities.reshape((-1,)))
            action = np.random.choice(valid_actions, p=probabilities)
            observation, reward, done, info = env.step(action)
            valid_actions = info['next_actions']
        return reward


def __find_n_weights_by_params(agent_params: Dict[str, Any]) -> int:
    """Run internals for `find_n_weights_by_params`."""
    agent = CounterpointEnvAgent(**agent_params)
    n_weights = agent.n_weights
    return n_weights


def find_n_weights_by_params(agent_params: Dict[str, Any]) -> int:
    """
    Find number of weights in agent's network by parameters of the agent.

    :param agent_params:
        arguments that must be passed to create an agent
    :return:
        number of weights in agent's actor model
    """
    # `tf` has dead lock if parent process launches it before a child process.
    results = map_in_parallel(
        __find_n_weights_by_params,
        [(agent_params,)],
        {'n_processes': 1}
    )
    n_weights = results[0]
    return n_weights
