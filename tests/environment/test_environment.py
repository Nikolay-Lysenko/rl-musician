"""
Test `rlmusician.environment.environment` module.

Author: Nikolay Lysenko
"""


from typing import List

import numpy as np
import pytest

from rlmusician.environment import CounterpointEnv, Piece


class TestCounterpointEnv:
    """Tests for `CounterpointEnv` class."""

    @pytest.mark.parametrize(
        "env, actions, expected",
        [
            (
                # `env`
                CounterpointEnv(
                    piece=Piece(
                        tonic='C',
                        scale='major',
                        n_measures=10,
                        max_skip=2,
                        line_specifications=[
                            {
                                'lowest_note': 'C4',
                                'highest_note': 'C5',
                                'start_note': 'E4',
                                'end_note': 'C4'
                            },
                        ],
                        rendering_params={}
                    ),
                    observation_decay=0.75,
                    reward_for_dead_end=-100,
                    scoring_coefs={'lines_correlation': 1},
                    scoring_fn_params={},
                ),
                # `actions`
                [2, 3, 3, 0, 0],
                # `expected`
                np.array([
                    0, 0, 0, 0, 0, 0.5625, 0, 0.421875,
                    1.3037109375, 0, 0, 0, 1
                ])
            ),
        ]
    )
    def test_observation(
            self, env: CounterpointEnv, actions: List[int],
            expected: np.ndarray
    ) -> None:
        """Test that `step` method returns proper observation."""
        for action in actions:
            observation, reward, done, info = env.step(action)
        assert not done
        np.testing.assert_equal(observation, expected)

    @pytest.mark.parametrize(
        "env, actions, expected",
        [
            (
                # `env`
                CounterpointEnv(
                    piece=Piece(
                        tonic='C',
                        scale='major',
                        n_measures=10,
                        max_skip=2,
                        line_specifications=[
                            {
                                'lowest_note': 'G3',
                                'highest_note': 'G4',
                                'start_note': 'C4',
                                'end_note': 'C4'
                            },
                            {
                                'lowest_note': 'G4',
                                'highest_note': 'G5',
                                'start_note': 'G5',
                                'end_note': 'C5'
                            },
                        ],
                        rendering_params={}
                    ),
                    observation_decay=0.75,
                    reward_for_dead_end=-100,
                    scoring_coefs={'lines_correlation': 1},
                    scoring_fn_params={},
                ),
                # `actions`
                [16, 6],
                # `expected`
                [6, 7, 9, 10, 12, 13, 14, 16, 18, 19, 20, 22, 24]
            ),
        ]
    )
    def test_info(
            self, env: CounterpointEnv, actions: List[int],
            expected: np.ndarray
    ) -> None:
        """Test that `step` method returns proper info about next actions."""
        for action in actions:
            observation, reward, done, info = env.step(action)
        result = info['next_actions']
        assert result == expected

    # TODO: Implement when `scoring.py` is updated.
    # @pytest.mark.parametrize(
    #     "env, actions, expected",
    #     [
    #         (
    #             # `env`
    #             PianoRollEnv(
    #                 n_semitones=5,
    #                 n_roll_steps=5,
    #                 observation_decay=0.5,
    #                 n_draws_per_roll_step=2,
    #                 scoring_coefs={'absence_of_outer_notes': 1},
    #                 scoring_fn_params={},
    #                 rendering_params={}
    #             ),
    #             # `actions`
    #             [2] + [0 for _ in range(7)],
    #             # `expected`
    #             -1
    #         )
    #     ]
    # )
    # def test_reward(
    #         self, env: PianoRollEnv, actions: List[int], expected: float
    # ) -> None:
    #     """Test that `step` method returns proper reward."""
    #     for action in actions:
    #         observation, reward, done, info = env.step(action)
    #     assert done
    #     assert reward == expected

    @pytest.mark.parametrize(
        "env, actions, expected",
        [
            (
                # `env`
                CounterpointEnv(
                    piece=Piece(
                        tonic='C',
                        scale='major',
                        n_measures=10,
                        max_skip=2,
                        line_specifications=[
                            {
                                'lowest_note': 'C4',
                                'highest_note': 'C5',
                                'start_note': 'E4',
                                'end_note': 'C4'
                            },
                        ],
                        rendering_params={}
                    ),
                    observation_decay=0.75,
                    reward_for_dead_end=-100,
                    scoring_coefs={'lines_correlation': 1},
                    scoring_fn_params={},
                ),
                # `actions`
                [2, 3, 3, 0, 0],
                # `expected`
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
            ),
        ]
    )
    def test_reset(
            self, env: CounterpointEnv, actions: List[int],
            expected: np.ndarray
    ) -> None:
        """Test `reset` method."""
        for action in actions:
            env.step(action)
        observation = env.reset()
        np.testing.assert_equal(observation, expected)
        assert env.piece.last_finished_measure == 0
