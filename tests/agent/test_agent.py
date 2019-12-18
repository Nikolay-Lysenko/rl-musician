"""
Test `rlmusician.agent.agent` module.

Author: Nikolay Lysenko
"""


from typing import List

import numpy as np
import pytest

from rlmusician.agent.agent import CounterpointEnvAgent
from rlmusician.agent.actor_model import create_actor_model


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
        dummy_model = create_actor_model((10,), 5)
        agent = CounterpointEnvAgent(
            lambda x, y: dummy_model,
            len(observation),
            n_lines,
            n_movements_per_line,
            hidden_layer_size=0,
            softmax_temperature=1
        )
        result = agent.represent_actions(observation, actions)
        np.testing.assert_equal(result, expected)
