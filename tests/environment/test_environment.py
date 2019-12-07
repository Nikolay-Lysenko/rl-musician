"""
Test `rlmusician.environment.environment` module.

Author: Nikolay Lysenko
"""


from typing import List

import numpy as np
import pytest

from rlmusician.environment import PianoRollEnv


class TestPianoRollEnv:
    """Tests for `PianoRollEnv` class."""

    @pytest.mark.parametrize(
        "env, actions, expected",
        [
            (
                # `env`
                PianoRollEnv(
                    n_semitones=5,
                    n_roll_steps=5,
                    observation_decay=0.5,
                    n_draws_per_roll_step=2,
                    scoring_coefs={'absence_of_outer_notes': 1},
                    scoring_fn_params={},
                    rendering_params={}
                ),
                # `actions`
                [2, 2, 1, 1, 1, 3, 3],
                # `expected`
                np.array([0, 0.75, 0.125, 1.5, 0])
            )
        ]
    )
    def test_observation(
            self, env: PianoRollEnv, actions: List[int], expected: np.ndarray
    ) -> None:
        """Test that `step` method returns proper observation."""
        env.reset()
        for action in actions:
            observation, reward, done, info = env.step(action)
        assert not done
        np.testing.assert_equal(observation, expected)

    @pytest.mark.parametrize(
        "env, actions, expected",
        [
            (
                # `env`
                PianoRollEnv(
                    n_semitones=5,
                    n_roll_steps=5,
                    observation_decay=0.5,
                    n_draws_per_roll_step=2,
                    scoring_coefs={'absence_of_outer_notes': 1},
                    scoring_fn_params={},
                    rendering_params={}
                ),
                # `actions`
                [2] + [0 for _ in range(7)],
                # `expected`
                -1
            )
        ]
    )
    def test_reward(
            self, env: PianoRollEnv, actions: List[int], expected: float
    ) -> None:
        """Test that `step` method returns proper reward."""
        env.reset()
        for action in actions:
            observation, reward, done, info = env.step(action)
        assert done
        assert reward == expected

    @pytest.mark.parametrize(
        "env, expected",
        [
            (
                # `env`
                PianoRollEnv(
                    n_semitones=5,
                    n_roll_steps=5,
                    observation_decay=0.5,
                    n_draws_per_roll_step=2,
                    scoring_coefs={'absence_of_outer_notes': 1},
                    scoring_fn_params={},
                    rendering_params={}
                ),
                # `expected`
                np.zeros((5,))
            )
        ]
    )
    def test_reset(self, env: PianoRollEnv, expected: np.ndarray) -> None:
        """Test `reset` method."""
        observation = env.reset()
        np.testing.assert_equal(observation, expected)
        assert env.current_episode_step == 0
        assert env.current_roll_step == 0
        assert env.n_draws_at_current_roll_step == 0
