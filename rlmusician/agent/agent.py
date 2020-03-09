"""
Create agent for interacting with `CounterpointEnv`.

Author: Nikolay Lysenko.
"""


from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy.special import softmax

from rlmusician.environment import CounterpointEnv
from rlmusician.utils import convert_to_base, map_in_parallel


class CounterpointEnvAgent:
    """Agent for interacting with `CounterpointEnv`."""

    def __init__(
            self,
            policy_fn: Callable[..., 'keras.models.Model'],
            observation_len: int,
            n_lines: int,
            n_movements_per_line: int,
            hidden_layer_size: int,
            softmax_temperature: float
    ):
        """
        Initialize instance.

        :param policy_fn:
            function that creates shallow neural network policy
        :param observation_len:
            length of observation returned by environment
        :param n_lines:
            number of lines in a piece to be created within environment
        :param n_movements_per_line:
            maximum number of line movements supported by environment
        :param hidden_layer_size:
            size of policy's hidden layer
        :param softmax_temperature:
            temperature parameter for Boltzmann softmax which is used for
            mapping scores of candidate actions to their probabilities
        """
        self.policy_fn = policy_fn
        self.observation_len = observation_len
        self.n_lines = n_lines
        self.n_movements_per_line = n_movements_per_line
        self.hidden_layer_size = hidden_layer_size
        self.softmax_temperature = softmax_temperature

        self.input_size = observation_len + n_lines * n_movements_per_line
        self.policy = policy_fn((self.input_size,), hidden_layer_size)
        self.shapes = [w.shape for w in self.policy.get_weights()]
        self.sizes = [w.size for w in self.policy.get_weights()]
        self.n_weights = sum(self.sizes)

    def represent_actions(
            self, observation: np.ndarray, actions: List[int]
    ) -> np.ndarray:
        """
        Create batch of potential actions' representation for policy.

        :param observation:
            observation returned by environment
        :param actions:
            list of actions allowed at the next step
        :return:
            batch of actions' representation
        """
        actions_part_width = self.n_lines * self.n_movements_per_line
        actions_part = np.zeros((len(actions), actions_part_width))
        offsets = [i * self.n_movements_per_line for i in range(self.n_lines)]
        for row_number, action in enumerate(actions):
            raw_encoded_action = convert_to_base(
                action, self.n_movements_per_line, self.n_lines
            )
            encoded_action = [
                idx + offset
                for idx, offset in zip(raw_encoded_action, offsets)
            ]
            actions_part[row_number, encoded_action] = 1
        observation_part = np.tile(observation, (len(actions), 1))
        representation = np.hstack((observation_part, actions_part))
        return representation

    def set_weights(self, flat_weights: np.ndarray) -> None:
        """
        Set weights of policy.

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
        self.policy.set_weights(weights)

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
            batch = self.represent_actions(observation, valid_actions)
            action_scores = self.policy.predict(batch)
            action_scores /= self.softmax_temperature
            action_probabilities = softmax(action_scores.reshape((-1,)))
            action = np.random.choice(valid_actions, p=action_probabilities)
            observation, reward, done, info = env.step(action)
            valid_actions = info['next_actions']
        return reward


def get_zero_weights(
        agent_params: Dict[str, Any]
) -> np.ndarray:  # pragma: no cover
    """Get zero weights of length compatible with the specified agent."""
    agent = CounterpointEnvAgent(**agent_params)
    n_weights = agent.n_weights
    weights = np.array([0 for _ in range(n_weights)])
    return weights


def load_flat_weigths(
        weights_path: str, agent_params: Dict[str, Any]
) -> np.ndarray:  # pragma: no cover
    """Load weights from file and flatten them."""
    agent = CounterpointEnvAgent(**agent_params)
    agent.policy.load_weights(weights_path)
    weights = np.hstack(tuple(w.flatten() for w in agent.policy.get_weights()))
    return weights


def extract_initial_weights(
        weights_path: Optional[str], agent_params: Dict[str, Any]
) -> np.ndarray:  # pragma: no cover
    """
    Load initial weights from a file or fill all of them with zeros.

    :param weights_path:
        path to a file with saved weights;
        if it is `None`, zeros are used as weights
    :param agent_params:
        arguments that must be passed to create an agent
    :return:
        initial value of agent weights
    """
    # `tf` has dead lock if parent process launches it before a child process.
    if weights_path is None:
        results = map_in_parallel(
            get_zero_weights,
            [(agent_params,)],
            {'n_processes': 1}
        )
    else:
        results = map_in_parallel(
            load_flat_weigths,
            [(weights_path, agent_params)],
            {'n_processes': 1}
        )
    weights = results[0]
    return weights
