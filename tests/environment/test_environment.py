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
                        scale_type='major',
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
                        voice_leading_rules={
                            'names': [
                                'rearticulation',
                                'destination_of_skip',
                                'turn_after_skip',
                                'VI_VII_resolution',
                                'step_motion_to_end'
                            ],
                            'params': {
                                'turn_after_skip': {
                                    'min_n_scale_degrees': 3
                                },
                                'step_motion_to_end': {
                                    'prohibit_rearticulation': False
                                }
                            }
                        },
                        harmony_rules={
                            'names': [],
                            'params': {}
                        },
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
                        scale_type='major',
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
                        voice_leading_rules={
                            'names': [
                                'rearticulation',
                                'destination_of_skip',
                                'turn_after_skip',
                                'VI_VII_resolution',
                                'step_motion_to_end'
                            ],
                            'params': {
                                'turn_after_skip': {
                                    'min_n_scale_degrees': 3
                                },
                                'step_motion_to_end': {
                                    'prohibit_rearticulation': False
                                }
                            }
                        },
                        harmony_rules={
                            'names': [
                                'consonance',
                                'absence_of_large_intervals'
                            ],
                            'params': {
                                'absence_of_large_intervals': {
                                    'max_n_semitones': 16
                                }
                            }
                        },
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
                [6, 10, 12, 16, 18, 20, 22, 24]
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

    @pytest.mark.parametrize(
        "env, actions, expected",
        [
            (
                # `env`
                CounterpointEnv(
                    piece=Piece(
                        tonic='C',
                        scale_type='major',
                        n_measures=5,
                        max_skip=2,
                        line_specifications=[
                            {
                                'lowest_note': 'G3',
                                'highest_note': 'G4',
                                'start_note': 'C4',
                                'end_note': 'G4'
                            },
                            {
                                'lowest_note': 'G4',
                                'highest_note': 'G5',
                                'start_note': 'C5',
                                'end_note': 'G5'
                            },
                            {
                                'lowest_note': 'G5',
                                'highest_note': 'G6',
                                'start_note': 'G6',
                                'end_note': 'C6'
                            },
                        ],
                        voice_leading_rules={
                            'names': [
                                'rearticulation',
                                'destination_of_skip',
                                'turn_after_skip',
                                'VI_VII_resolution',
                                'step_motion_to_end'
                            ],
                            'params': {
                                'turn_after_skip': {
                                    'min_n_scale_degrees': 3
                                },
                                'step_motion_to_end': {
                                    'prohibit_rearticulation': False
                                }
                            }
                        },
                        harmony_rules={
                            'names': [
                                'consonance',
                                'absence_of_large_intervals'
                            ],
                            'params': {
                                'absence_of_large_intervals': {
                                    'max_n_semitones': 16
                                }
                            }
                        },
                        rendering_params={}
                    ),
                    observation_decay=0.75,
                    reward_for_dead_end=-100,
                    scoring_coefs={'lines_correlation': 1},
                    scoring_fn_params={},
                ),
                # `actions`
                [91, 91, 91],
                # `expected`
                0.6621
            ),
            (
                # `env`
                CounterpointEnv(
                    piece=Piece(
                        tonic='C',
                        scale_type='major',
                        n_measures=5,
                        max_skip=2,
                        line_specifications=[
                            {
                                'lowest_note': 'A3',
                                'highest_note': 'G4',
                                'start_note': 'C4',
                                'end_note': 'C4'
                            },
                        ],
                        voice_leading_rules={
                            'names': [
                                'rearticulation',
                                'destination_of_skip',
                                'turn_after_skip',
                                'VI_VII_resolution',
                                'step_motion_to_end'
                            ],
                            'params': {
                                'turn_after_skip': {
                                    'min_n_scale_degrees': 3
                                },
                                'step_motion_to_end': {
                                    'prohibit_rearticulation': False
                                }
                            }
                        },
                        harmony_rules={
                            'names': [
                                'consonance',
                                'absence_of_large_intervals'
                            ],
                            'params': {
                                'absence_of_large_intervals': {
                                    'max_n_semitones': 16
                                }
                            }
                        },
                        rendering_params={}
                    ),
                    observation_decay=0.75,
                    reward_for_dead_end=-100,
                    scoring_coefs={'lines_correlation': 1},
                    scoring_fn_params={},
                ),
                # `actions`
                [1, 1],
                # `expected`
                -100
            ),
        ]
    )
    def test_reward(
            self, env: CounterpointEnv, actions: List[int], expected: float
    ) -> None:
        """Test that `step` method returns proper reward."""
        for action in actions:
            observation, reward, done, info = env.step(action)
        assert done
        assert round(reward, 4) == expected

    @pytest.mark.parametrize(
        "env, actions, expected",
        [
            (
                # `env`
                CounterpointEnv(
                    piece=Piece(
                        tonic='C',
                        scale_type='major',
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
                        voice_leading_rules={
                            'names': [
                                'rearticulation',
                                'destination_of_skip',
                                'turn_after_skip',
                                'VI_VII_resolution',
                                'step_motion_to_end'
                            ],
                            'params': {
                                'turn_after_skip': {
                                    'min_n_scale_degrees': 3
                                },
                                'step_motion_to_end': {
                                    'prohibit_rearticulation': False
                                }
                            }
                        },
                        harmony_rules={
                            'names': [],
                            'params': {}
                        },
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
