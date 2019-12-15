"""
Create agent for interacting with `CounterpointEnv`.

Author: Nikolay Lysenko.
"""


from typing import Callable, List, Union

import numpy as np

from rlmusician.environment import CounterpointEnv
from rlmusician.utils import convert_to_base


class CounterpointEnvAgent:
    """Agent for interacting with `CounterpointEnv`."""

    def __init__(
            self,
            model_fn: Callable[..., 'keras.models.Model'],
            observation_size: int,
            n_lines: int,
            n_movements_per_line: int,
            hidden_layer_size: int
    ):
        """
        Initialize instance.

        :param model_fn:
            function that creates actor model
        :param observation_size:
            size of observation returned by environment
        :param n_lines:
            number of lines in a piece to be created within environment
        :param n_movements_per_line:
            maximum number of line movements supported by environment
        :param hidden_layer_size:
            size of actor model's hidden layer
        """
        self.model_fn = model_fn
        self.observation_size = observation_size
        self.n_lines = n_lines
        self.n_movements_per_line = n_movements_per_line
        self.hidden_layer_size = hidden_layer_size

        self.input_size = observation_size + n_lines * n_movements_per_line
        self.model = model_fn(self.input_size, hidden_layer_size)
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

    def set_weights(
            self, flat_weights: Union[np.ndarray, List[float]]
    ) -> None:
        """
        Set weights of model.

        :param flat_weights:
            list or 1D array of new weights
        :return:
            None
        """
        if isinstance(flat_weights, list):
            flat_weights = np.array(flat_weights)
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
            probabilities = 1 / (1 + np.exp(-probabilities))
            action = np.random.choice(valid_actions, p=probabilities)
            observation, reward, done, info = env.step(action)
            valid_actions = info['next_actions']
        return reward
