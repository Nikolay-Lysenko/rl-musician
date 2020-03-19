"""
Test `rlmusician.agent.agent` module.

Author: Nikolay Lysenko
"""


from typing import List

import numpy as np
import pytest

from rlmusician.agent.agent import CounterpointEnvAgent
from rlmusician.agent.policy import create_policy
from rlmusician.environment import CounterpointEnv, Piece


class TestCounterpointEnvAgent:
    """Tests for `CounterpointEnvAgent` class."""

    @pytest.mark.parametrize(
        "n_lines, n_movements_per_line, observation, actions, expected",
        [
            (
                # `n_lines`
                2,
                # `n_movements_per_line`
                3,
                # `observation`
                np.array([1, 2, 3]),
                # `actions`
                [2, 4, 6],
                # `expected`
                np.array([
                    [1, 2, 3, 1, 0, 0, 0, 0, 1],
                    [1, 2, 3, 0, 1, 0, 0, 1, 0],
                    [1, 2, 3, 0, 0, 1, 1, 0, 0],
                ])
            ),
            (
                # `n_lines`
                3,
                # `n_movements_per_line`
                3,
                # `observation`
                np.array([1, 2, 3]),
                # `actions`
                [4, 22],
                # `expected`
                np.array([
                    [1, 2, 3, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                    [1, 2, 3, 0, 0, 1, 0, 1, 0, 0, 1, 0],
                ])
            ),
        ]
    )
    def test_represent_actions(
            self, n_lines: int, n_movements_per_line: int,
            observation: np.ndarray, actions: List[int], expected: np.ndarray
    ) -> None:
        """Test `represent_actions` method."""
        agent = CounterpointEnvAgent(
            create_policy,
            len(observation),
            n_lines,
            n_movements_per_line,
            hidden_layer_size=3,
            softmax_temperature=1
        )
        result = agent.represent_actions(observation, actions)
        np.testing.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "observation_len, n_lines, n_movements_per_line, hidden_layer_size, "
        "flat_weights",
        [
            (
                # `observation_len`
                3,
                # `n_lines`
                2,
                # `n_movements_per_line`
                3,
                # `hidden_layer_size`
                5,
                # `flat_weights`
                np.array([1 for _ in range(56)])
            ),
        ]
    )
    def test_set_weights(
            self, observation_len: int, n_lines: int,
            n_movements_per_line: int, hidden_layer_size: int,
            flat_weights: np.ndarray
    ) -> None:
        """Test `set_weights` method."""
        agent = CounterpointEnvAgent(
            create_policy,
            observation_len,
            n_lines,
            n_movements_per_line,
            hidden_layer_size,
            softmax_temperature=1
        )
        agent.set_weights(flat_weights)
        policy_weights = agent.policy.get_weights()
        flat_policy_weights = np.hstack((x.flatten() for x in policy_weights))
        np.testing.assert_equal(flat_policy_weights, flat_weights)

    @pytest.mark.parametrize(
        "observation_len, n_lines, n_movements_per_line, hidden_layer_size, "
        "env",
        [
            (
                # `observation_len`
                25,
                # `n_lines`
                2,
                # `n_movements_per_line`
                5,
                # `hidden_layer_size`
                5,
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
                                    'prohibit_rearticulation': True
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
                    reward_for_dead_end=-100.0,
                    scoring_coefs={'lines_correlation': 1},
                    scoring_fn_params={},
                ),
            ),
        ]
    )
    def test_run_episode(
            self, observation_len: int, n_lines: int,
            n_movements_per_line: int, hidden_layer_size: int,
            env: CounterpointEnv
    ) -> None:
        """Test that `run_episode` method does not fail."""
        agent = CounterpointEnvAgent(
            create_policy,
            observation_len,
            n_lines,
            n_movements_per_line,
            hidden_layer_size,
            softmax_temperature=1
        )
        reward = agent.run_episode(env)
        assert isinstance(reward, float)
