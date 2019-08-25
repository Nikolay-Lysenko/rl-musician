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
    score_horizontal_variance,
    score_vertical_variance,
    score_repetitiveness,
    score_consonances
)


SCORING_FN_REGISTRY = {
    'horizontal_variance': score_horizontal_variance,
    'vertical_variance': score_vertical_variance,
    'repetitiveness': score_repetitiveness,
    'consonances': score_consonances
}


class MusicCompositionEnv(gym.Env):
    """
    An environment where agent composes piano roll.
    """

    reward_range = (-np.inf, np.inf)

    def __init__(
            self,
            n_semitones: int,
            n_roll_steps: int,
            n_observed_roll_steps: int,
            max_n_stalled_episode_steps: int,
            scoring_coefs: Dict[str, float],
            scoring_fn_params: Dict[str, Dict[str, Any]],
            data_dir: str
    ):
        """
        Initialize instance.

        :param n_semitones:
            number of consecutive semitones (piano keys) available to an agent
        :param n_roll_steps:
            total duration of composition in piano roll's time steps
            (in other words, number of columns of piano roll)
        :param n_observed_roll_steps:
            number of previous piano roll's time steps available for observing
        :param max_n_stalled_episode_steps:
            number of episode steps after which forced movement forward occurs
            on piano roll
        :param scoring_coefs:
            mapping from scoring function names to their weights in final score
        :param scoring_fn_params:
            mapping from scoring function names to their parameters
        :param data_dir:
            directory where rendered results are going to be saved
        """
        self.n_semitones = n_semitones
        self.n_roll_steps = n_roll_steps
        self.n_observed_roll_steps = n_observed_roll_steps
        self.max_n_stalled_episode_steps = max_n_stalled_episode_steps
        self.scoring_coefs = scoring_coefs
        self.scoring_fn_params = scoring_fn_params
        self.data_dir = data_dir

        self.piano_roll = None
        self.n_piano_roll_steps_passed = None
        self.n_episode_steps_passed = None
        self.n_stalled_episode_steps = None

        self.action_space = gym.spaces.Discrete(
            n_semitones + 1  # The last action stands for step forward on roll.
        )
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(n_semitones, n_observed_roll_steps),
            dtype=np.int32
        )

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

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
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
        if action != self.n_semitones:
            self.piano_roll[action, self.n_piano_roll_steps_passed] += 1
            self.piano_roll[action, self.n_piano_roll_steps_passed] %= 2
            self.n_stalled_episode_steps += 1
        force_movement = (
            self.n_stalled_episode_steps == self.max_n_stalled_episode_steps
        )
        if force_movement or action == self.n_semitones:
            self.n_piano_roll_steps_passed += 1
            self.n_stalled_episode_steps = 0
        self.n_episode_steps_passed += 1

        # Provide feedback.
        steps_to_see = (
            self.n_piano_roll_steps_passed - self.n_observed_roll_steps + 1,
            self.n_piano_roll_steps_passed + 1
        )
        if steps_to_see[0] >= 0:
            observation = self.piano_roll[:, steps_to_see[0]:steps_to_see[1]]
        else:
            observation = np.hstack((
                np.zeros((self.n_semitones, -steps_to_see[0]), dtype=np.int32),
                self.piano_roll[:, 0:steps_to_see[1]]
            ))
        done = self.n_piano_roll_steps_passed == self.n_roll_steps - 1
        reward = self.__evaluate() if done else 0
        info = {}
        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        """
        Reset the state of the environment and return an initial observation.

        :return:
            the initial observation of the space
        """
        self.n_episode_steps_passed = 0
        self.n_piano_roll_steps_passed = 0
        self.n_stalled_episode_steps = 0

        piano_roll_shape = (self.n_semitones, self.n_roll_steps)
        self.piano_roll = np.zeros(piano_roll_shape, dtype=np.int32)

        observed_roll_shape = (self.n_semitones, self.n_observed_roll_steps)
        observation = np.zeros(observed_roll_shape, dtype=np.int32)
        return observation

    def render(self, mode='human') -> None:
        """
        Save final piano roll to TSV file.

        :return:
            None
        """
        episode_end = self.n_piano_roll_steps_passed == self.n_roll_steps - 1
        if not episode_end:
            return
        file_name = f"roll_{str(time()).replace('.', ',')}.tsv"
        file_path = os.path.join(self.data_dir, 'piano_rolls', file_name)
        np.savetxt(file_path, self.piano_roll, fmt='%i', delimiter='\t')
