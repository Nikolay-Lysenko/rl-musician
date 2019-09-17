"""
Create environment with Gym API.

In this environment, an agent sequentially composes piano roll.
A piano roll is a `numpy` 2D-array with rows corresponding to notes,
columns corresponding to time steps, and cells containing zeros and ones
and indicating whether a note is played.

References:
    https://en.wikipedia.org/wiki/Piano_roll#In_digital_audio_workstations

Author: Nikolay Lysenko
"""


import datetime
import os
from typing import Any, Dict, Tuple

import gym
import numpy as np
from sinethesizer.io.piano_roll_to_tsv import write_roll_to_tsv_file

from rlmusician.environment.scoring import get_scoring_functions_registry
from rlmusician.utils import (
    create_midi_from_piano_roll, create_wav_from_events
)


class PianoRollEnv(gym.Env):
    """
    An environment where agent sequentially composes piano roll.
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
            rendering_params: Dict[str, Any]
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
        :param rendering_params:
            settings of environment rendering
        """
        self.n_semitones = n_semitones
        self.n_roll_steps = n_roll_steps
        self.n_observed_roll_steps = n_observed_roll_steps
        self.max_n_stalled_episode_steps = max_n_stalled_episode_steps
        self.scoring_coefs = scoring_coefs
        self.scoring_fn_params = scoring_fn_params
        self.rendering_params = rendering_params

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
        registry = get_scoring_functions_registry()
        for fn_name, weight in self.scoring_coefs.items():
            fn = registry[fn_name]
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

    def render(self, mode='human') -> None:  # pragma: no cover
        """
        Save final piano roll to TSV file.

        :return:
            None
        """
        episode_end = self.n_piano_roll_steps_passed == self.n_roll_steps - 1
        if not episode_end:
            return

        top_level_dir = self.rendering_params['dir']
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S,%f")
        nested_dir = os.path.join(top_level_dir, f"result_{now}")
        os.mkdir(nested_dir)

        roll_path = os.path.join(nested_dir, 'piano_roll.tsv')
        np.savetxt(roll_path, self.piano_roll, fmt='%i', delimiter='\t')

        midi_instrument = self.rendering_params['midi_instrument']
        lowest_note = self.rendering_params['lowest_note']
        midi_path = os.path.join(nested_dir, 'music.midi')
        create_midi_from_piano_roll(
            self.piano_roll, midi_path, midi_instrument, lowest_note
        )

        events_path = os.path.join(nested_dir, 'sinethesizer_events.tsv')
        events_params = self.rendering_params['sinethesizer']
        write_roll_to_tsv_file(
            self.piano_roll, events_path, lowest_note, **events_params
        )

        wav_path = os.path.join(nested_dir, 'music.wav')
        create_wav_from_events(events_path, wav_path)
