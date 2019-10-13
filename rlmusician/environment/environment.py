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
            observation_decay: float,
            n_draws_per_roll_step: int,
            scoring_coefs: Dict[str, float],
            scoring_fn_params: Dict[str, Dict[str, Any]],
            rendering_params: Dict[str, Any]
    ):
        """
        Initialize instance.

        :param n_semitones:
            size of available pitch range in semitones; in other words,
            number of consecutive piano keys available to an agent
        :param n_roll_steps:
            total duration of composition in piano roll's time steps;
            in other words, number of columns of piano roll
        :param observation_decay:
            coefficient of exponential decay for previous piano roll's
            time steps
        :param n_draws_per_roll_step:
            number of episode steps after which movement to the next
            piano roll's time step occurs
        :param scoring_coefs:
            mapping from scoring function names to their weights in final score
        :param scoring_fn_params:
            mapping from scoring function names to their parameters
        :param rendering_params:
            settings of environment rendering
        """
        self.n_semitones = n_semitones
        self.n_roll_steps = n_roll_steps
        self.observation_decay = observation_decay
        self.n_draws_per_roll_step = n_draws_per_roll_step
        self.scoring_coefs = scoring_coefs
        self.scoring_fn_params = scoring_fn_params
        self.rendering_params = rendering_params

        self.piano_roll = None
        self.current_episode_step = None
        self.current_roll_step = None
        self.n_draws_at_current_roll_step = None

        self.action_space = gym.spaces.Discrete(n_semitones)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1 / (1 - observation_decay),
            shape=(n_semitones,),
            dtype=np.float32
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

    @property
    def __decay_coefs(self) -> np.ndarray:
        """Get decay coefficients for passed piano roll's steps."""
        decay_coefs = np.empty((self.current_roll_step + 1,))
        decay_coefs[0] = 1
        decay_coefs[1:] = self.observation_decay
        decay_coefs = np.cumprod(decay_coefs)
        decay_coefs = np.flip(decay_coefs)
        decay_coefs = decay_coefs.reshape((1, -1))
        return decay_coefs

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
        self.piano_roll[action, self.current_roll_step] = 1
        self.n_draws_at_current_roll_step += 1
        if self.n_draws_at_current_roll_step == self.n_draws_per_roll_step:
            self.current_roll_step += 1
            self.n_draws_at_current_roll_step = 0
        self.current_episode_step += 1

        # Provide feedback.
        known_piano_roll = self.piano_roll[:, :self.current_roll_step + 1]
        observation = np.sum(self.__decay_coefs * known_piano_roll, axis=1)
        done = self.current_roll_step == self.n_roll_steps - 1
        reward = self.__evaluate() if done else 0
        info = {}
        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        """
        Reset the state of the environment and return an initial observation.

        :return:
            the initial observation of the space
        """
        self.current_episode_step = 0
        self.current_roll_step = 0
        self.n_draws_at_current_roll_step = 0

        piano_roll_shape = (self.n_semitones, self.n_roll_steps)
        self.piano_roll = np.zeros(piano_roll_shape, dtype=np.int32)

        observation = np.zeros((self.n_semitones,), dtype=np.float32)
        return observation

    def render(self, mode='human') -> None:  # pragma: no cover
        """
        Save final piano roll to TSV file.

        :return:
            None
        """
        episode_end = self.current_roll_step == self.n_roll_steps - 1
        if not episode_end:
            return

        top_level_dir = self.rendering_params['dir']
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S,%f")
        nested_dir = os.path.join(top_level_dir, f"result_{now}")
        os.mkdir(nested_dir)

        roll_path = os.path.join(nested_dir, 'piano_roll.tsv')
        np.savetxt(roll_path, self.piano_roll, fmt='%i', delimiter='\t')

        lowest_note = self.rendering_params['lowest_note']
        step_in_seconds = self.rendering_params['step_in_seconds']

        midi_path = os.path.join(nested_dir, 'music.mid')
        midi_params = self.rendering_params['midi']
        n_seconds_per_minute = 60
        create_midi_from_piano_roll(
            self.piano_roll,
            midi_path,
            lowest_note,
            n_seconds_per_minute / step_in_seconds,
            **midi_params
        )

        events_path = os.path.join(nested_dir, 'sinethesizer_events.tsv')
        events_params = self.rendering_params['sinethesizer']
        events_params['step_in_seconds'] = step_in_seconds
        write_roll_to_tsv_file(
            self.piano_roll,
            events_path,
            lowest_note,
            **events_params
        )

        wav_path = os.path.join(nested_dir, 'music.wav')
        create_wav_from_events(events_path, wav_path)
