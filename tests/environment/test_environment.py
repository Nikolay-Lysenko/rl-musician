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
                        cantus_firmus=['C4', 'D4', 'E4', 'D4', 'C4'],
                        counterpoint_specifications={
                            'start_note': 'E4',
                            'end_note': 'E4',
                            'lowest_note': 'G3',
                            'highest_note': 'G4',
                            'start_pause_in_eighths': 4,
                            'max_skip_in_degrees': 2,
                        },
                        rules={
                            'names': ['rearticulation_stability'],
                            'params': {}
                        },
                        rendering_params={}
                    ),
                    scoring_coefs={'number_of_skips': 1},
                    scoring_fn_params={
                        'number_of_skips': {'rewards': {1: 1}}
                    },
                    reward_for_dead_end=-100,
                ),
                # `actions`
                [14, 6, 8],
                # `expected`
                np.array([
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
                        cantus_firmus=['C4', 'D4', 'E4', 'D4', 'C4'],
                        counterpoint_specifications={
                            'start_note': 'E4',
                            'end_note': 'E4',
                            'lowest_note': 'G3',
                            'highest_note': 'G4',
                            'start_pause_in_eighths': 4,
                            'max_skip_in_degrees': 2,
                        },
                        rules={
                            'names': [
                                'rhythmic_pattern_validity',
                                'rearticulation_stability',
                                'consonance_on_strong_beat',
                                'resolution_of_suspended_dissonance',
                            ],
                            'params': {}
                        },
                        rendering_params={}
                    ),
                    scoring_coefs={'number_of_skips': 1},
                    scoring_fn_params={
                        'number_of_skips': {'rewards': {1: 1}}
                    },
                    reward_for_dead_end=-100,
                ),
                # `actions`
                [13, 15],
                # `expected`
                [6, 11]
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
                        cantus_firmus=['C4', 'D4', 'E4', 'D4', 'C4'],
                        counterpoint_specifications={
                            'start_note': 'E4',
                            'end_note': 'E4',
                            'lowest_note': 'G3',
                            'highest_note': 'G4',
                            'start_pause_in_eighths': 4,
                            'max_skip_in_degrees': 2,
                        },
                        rules={
                            'names': ['rearticulation_stability'],
                            'params': {}
                        },
                        rendering_params={}
                    ),
                    scoring_coefs={'number_of_skips': 1},
                    scoring_fn_params={
                        'number_of_skips': {'rewards': {1: 1}}
                    },
                    reward_for_dead_end=-100,
                ),
                # `actions`
                [14, 6, 8, 11, 5, 15, 9],
                # `expected`
                0
            ),
            (
                # `env`
                CounterpointEnv(
                    piece=Piece(
                        tonic='C',
                        scale_type='major',
                        cantus_firmus=['C4', 'D4', 'E4', 'D4', 'C4'],
                        counterpoint_specifications={
                            'start_note': 'E4',
                            'end_note': 'E4',
                            'lowest_note': 'G3',
                            'highest_note': 'G4',
                            'start_pause_in_eighths': 4,
                            'max_skip_in_degrees': 2,
                        },
                        rules={
                            'names': ['rearticulation_stability'],
                            'params': {}
                        },
                        rendering_params={}
                    ),
                    scoring_coefs={'number_of_skips': 1},
                    scoring_fn_params={
                        'number_of_skips': {'rewards': {4: 1}}
                    },
                    reward_for_dead_end=-100,
                ),
                # `actions`
                [14, 6, 8, 11, 5, 15, 9],
                # `expected`
                1
            ),
            (
                # `env`
                CounterpointEnv(
                    piece=Piece(
                        tonic='C',
                        scale_type='major',
                        cantus_firmus=['C4', 'C4', 'C3', 'C4', 'C4'],
                        counterpoint_specifications={
                            'start_note': 'E4',
                            'end_note': 'E4',
                            'lowest_note': 'G3',
                            'highest_note': 'G4',
                            'start_pause_in_eighths': 4,
                            'max_skip_in_degrees': 2,
                        },
                        rules={
                            'names': ['absence_of_large_intervals'],
                            'params': {
                                'absence_of_large_intervals': {
                                    'max_n_semitones': 7
                                }
                            }
                        },
                        rendering_params={}
                    ),
                    scoring_coefs={'number_of_skips': 1},
                    scoring_fn_params={
                        'number_of_skips': {'rewards': {1: 1}}
                    },
                    reward_for_dead_end=-100,
                ),
                # `actions`
                [14, 12],
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
                        cantus_firmus=['C4', 'D4', 'E4', 'D4', 'C4'],
                        counterpoint_specifications={
                            'start_note': 'E4',
                            'end_note': 'E4',
                            'lowest_note': 'G3',
                            'highest_note': 'G4',
                            'start_pause_in_eighths': 4,
                            'max_skip_in_degrees': 2,
                        },
                        rules={
                            'names': ['rearticulation_stability'],
                            'params': {}
                        },
                        rendering_params={}
                    ),
                    scoring_coefs={'number_of_skips': 1},
                    scoring_fn_params={
                        'number_of_skips': {'rewards': {1: 1}}
                    },
                    reward_for_dead_end=-100,
                ),
                # `actions`
                [14, 6, 8, 11, 5, 15],
                # `expected`
                np.array([
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ])
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
        assert env.piece.current_time_in_eighths == 8
