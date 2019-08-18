"""
Create environment with Gym API.

Author: Nikolay Lysenko
"""


import os
from time import time
from typing import Any, Dict, Tuple

import gym
import numpy as np

from rlmusician.environment.scoring import (
    score_notewise_entropy, score_consonances
)


SCORING_FN_REGISTRY = {
    'note-wise_entropy': score_notewise_entropy,
    'consonances': score_consonances
}


class MusicCompositionEnv(gym.Env):
    """
    An environment where agent composes piano roll.
    """

    reward_range = (-np.inf, np.inf)

    def __init__(
            self, n_semitones: int, n_time_steps: int, observed_length: int,
            scoring_coefs: Dict[str, float],
            scoring_fn_params: Dict[str, Dict[str, Any]],
            data_dir: str
    ):
        """
        Initialize instance.

        :param n_semitones:
            number of consecutive semitones (piano keys) available to agent
        :param n_time_steps:
            total duration of composition in time steps
        :param observed_length:
            number of piano roll's time steps available for observing
        :param scoring_coefs:
            mapping from scoring function names to their weights in final score
        :param scoring_fn_params:
            mapping from scoring function names to their parameters
        :param data_dir:
            directory where rendered results are going to be saved
        """
        self.scoring_coefs = scoring_coefs
        self.scoring_fn_params = scoring_fn_params
        self.n_semitones = n_semitones
        self.n_time_steps = n_time_steps
        self.observed_length = observed_length
        self.data_dir = data_dir

        self.piano_roll = None
        self.n_piano_roll_steps_passed = None
        self.n_episode_steps_passed = None

        self.action_space = gym.spaces.Discrete(n_semitones + 1)
        self.observation_space = gym.spaces.Tuple([
            gym.spaces.Box(
                low=0,
                high=1,
                shape=(n_semitones, observed_length),
                dtype=np.int32
            ),
            gym.spaces.Box(
                low=-1e10,
                high=1e10,
                shape=(1,),
                dtype=np.float32
            )
        ])

    def __evaluate(self) -> float:
        """Evaluate current state of piano roll."""
        score = 0
        for fn_name, weight in self.scoring_coefs.items():
            fn = SCORING_FN_REGISTRY[fn_name]
            score += weight * fn(
                self.piano_roll,
                **self.scoring_fn_params.get(fn_name, {})
            )
        return score

    def step(
            self, action: int
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, Dict]:
        """
        Run one step of the environment's dynamics.

        :param action:
            an action provided by an agent to the environment
        :return:
            a tuple of:
            - observation: agent's observation of the current environment,
            - reward: amount of reward returned after previous action,
            - done: whether the episode has ended, in which case further
                    `step()` calls will return undefined results,
            - info: auxiliary diagnostic information
                    (helpful for debugging and sometimes learning).
        """
        # Act.
        if action == self.n_semitones:  # Reserved action for shift forward.
            self.n_piano_roll_steps_passed += 1
        else:
            self.piano_roll[action, self.n_piano_roll_steps_passed] += 1
            self.piano_roll[action, self.n_piano_roll_steps_passed] %= 2
        self.n_episode_steps_passed += 1

        # Provide feedback.
        steps_to_see = (
            self.n_piano_roll_steps_passed - self.observed_length + 1,
            self.n_piano_roll_steps_passed + 1
        )
        if steps_to_see[0] >= 0:
            roll_to_see = self.piano_roll[:, steps_to_see[0]:steps_to_see[1]]
        else:
            roll_to_see = np.hstack((
                np.zeros((self.n_semitones, -steps_to_see[0]), dtype=np.int32),
                self.piano_roll[:, 0:steps_to_see[1]]
            ))
        observation = (roll_to_see, np.array([self.__evaluate()]))
        done = self.n_piano_roll_steps_passed == self.n_time_steps - 1
        reward = observation[1][0] if done else 0
        info = {}
        return observation, reward, done, info

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset the state of the environment and return an initial observation.

        :return:
            the initial observation of the space
        """
        self.n_episode_steps_passed = 0
        self.n_piano_roll_steps_passed = 0

        piano_roll_shape = (self.n_semitones, self.n_time_steps)
        self.piano_roll = np.zeros(piano_roll_shape, dtype=np.int32)

        observed_roll_shape = (self.n_semitones, self.observed_length)
        roll_to_see = np.zeros(observed_roll_shape, dtype=np.int32)
        observation = (roll_to_see, np.array([self.__evaluate()]))
        return observation

    def render(self, mode='human') -> None:
        """
        Save final piano roll to TSV file.

        :return:
            None
        """
        episode_end = self.n_piano_roll_steps_passed == self.n_time_steps - 1
        if not episode_end:
            return
        file_name = f"roll_{str(time()).replace('.', ',')}.tsv"
        file_path = os.path.join(self.data_dir, 'piano_rolls', file_name)
        np.savetxt(file_path, self.piano_roll, fmt='%i', delimiter='\t')
