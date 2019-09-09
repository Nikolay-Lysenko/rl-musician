"""
Test `rlmusician.agent.crossentropy` module.

Author: Nikolay Lysenko
"""


from typing import Any, Dict, List, Optional, Tuple

import gym
import numpy as np
import pytest

from rlmusician.agent.crossentropy import (
     CrossEntropyAgent, CrossEntropyAgentMemory
)


class TestCrossEntropyAgentMemory:
    """Test `CrossEntropyAgentMemory` class."""

    @pytest.mark.parametrize(
        "entries, size, expected_data, expected_best",
        [
            (
                # `entries`
                [(np.array([1]), 1), (np.array([2]), 0)],
                # `size`
                3,
                # `expected_data`
                [
                    {'flat_weights': np.array([1]), 'score': 1},
                    {'flat_weights': np.array([2]), 'score': 0},
                    None,
                ],
                # `expected_best`
                {'flat_weights': np.array([1]), 'score': 1}
            ),
            (
                # `entries`
                [(np.array([1]), 1), (np.array([2]), 0), (np.array([3]), -1)],
                # `size`
                3,
                # `expected_data`
                [
                    {'flat_weights': np.array([1]), 'score': 1},
                    {'flat_weights': np.array([2]), 'score': 0},
                    {'flat_weights': np.array([3]), 'score': -1},
                ],
                # `expected_best`
                {'flat_weights': np.array([1]), 'score': 1}
            ),
            (
                # `entries`
                [(np.array([1]), 1), (np.array([2]), 0), (np.array([3]), -1)],
                # `size`
                2,
                # `expected_data`
                [
                    {'flat_weights': np.array([3]), 'score': -1},
                    {'flat_weights': np.array([2]), 'score': 0},
                ],
                # `expected_best`
                {'flat_weights': np.array([1]), 'score': 1}
            ),
        ]
    )
    def test_add(
            self, entries: List[Tuple[np.ndarray, float]], size: int,
            expected_data: List[Optional[Dict[str, Any]]],
            expected_best: Dict[str, Any]
    ) -> None:
        """Test `add` method."""
        memory = CrossEntropyAgentMemory(size)
        for entry in entries:
            memory.add(*entry)
        assert memory.data == expected_data
        assert memory.best == expected_best

    @pytest.mark.parametrize(
        "entries, size, n_entries",
        [
            (
                # `entries`
                [(np.array([1]), 1), (np.array([2]), 0), (np.array([3]), -1)],
                # `size`
                5,
                # `n_entries`
                2
            ),
        ]
    )
    def test_sample_with_valid_values(
            self, entries: List[Tuple[np.ndarray, float]], size: int,
            n_entries: int
    ) -> None:
        """Test that `sample` method works given correct inputs."""
        memory = CrossEntropyAgentMemory(size)
        for entry in entries:
            memory.add(*entry)
        results = memory.sample(n_entries)
        for entry in results:
            assert (entry['flat_weights'], entry['score']) in entries

    @pytest.mark.parametrize(
        "entries, size, n_entries",
        [
            (
                # `entries`
                [(np.array([1]), 1), (np.array([2]), 0), (np.array([3]), -1)],
                # `size`
                3,
                # `n_entries`
                4
            ),
            (
                # `entries`
                [(np.array([1]), 1)],
                # `size`
                3,
                # `n_entries`
                2
            ),
        ]
    )
    def test_sample_with_invalid_values(
            self, entries: List[Tuple[np.ndarray, float]], size: int,
            n_entries: int
    ) -> None:
        """Test that `sample` method raises given incorrect inputs."""
        memory = CrossEntropyAgentMemory(size)
        for entry in entries:
            memory.add(*entry)
        with pytest.raises(RuntimeError):
            memory.sample(n_entries)


class ActorModelMock:
    """Mock for actor model."""

    def __init__(self, weight: float):
        """Initialize instance."""
        self.weight = weight

    def get_weights(self) -> List[np.ndarray]:
        """Get weights."""
        return [np.array([self.weight])]

    def set_weights(self, weights: List[np.ndarray]) -> None:
        """Set weights."""
        self.weight = weights[0][0]

    def predict(self, _: np.ndarray) -> np.ndarray:
        """Compute probabilities of actions."""
        first_proba = np.exp(self.weight) / (np.exp(self.weight) + 1)
        return np.array([[first_proba, 1 - first_proba]])


class EnvMock(gym.Env):
    """Mock for environment."""

    n_steps_per_episode = 5
    observation = np.array([0])
    reward_range = (-np.inf, np.inf)
    action_space = gym.spaces.Discrete(2)
    observation_space = gym.spaces.Discrete(1)

    def __init__(self):
        """Initialize instance."""
        self.position = 0
        self.n_steps_passed = 0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Run one step."""
        self.position += 2 * action - 1
        self.n_steps_passed += 1
        if self.n_steps_passed == self.n_steps_per_episode:
            return self.observation, self.position, True, {}
        else:
            return self.observation, 0, False, {}

    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.position = 0
        self.n_steps_passed = 0
        return self.observation

    def render(self, mode='human') -> None:
        """Render the environment."""
        print(self.position)


class TestCrossEntropyAgent:
    """Tests for `CrossEntropyAgent` class."""

    def test_fit(self) -> None:
        """Test `fit` method."""
        env = EnvMock()
        model = ActorModelMock(0)
        agent = CrossEntropyAgent(model, 10, n_episodes_per_candidate=1)
        agent.fit(env, n_populations=10)
        assert agent.memory.best['score'] > 4.75

    def test_fit_with_warmup(self) -> None:
        """Test `fit` method with warmup."""
        env = EnvMock()
        model = ActorModelMock(0)
        agent = CrossEntropyAgent(model, 10, n_warmup_candidates=1)
        agent.fit(env, n_populations=10)
        assert agent.memory.best['score'] > 4.75

    def test_test(self) -> None:
        """Test that `test` method works."""
        env = EnvMock()
        model = ActorModelMock(1)
        agent = CrossEntropyAgent(model, 10, n_episodes_per_candidate=1)
        agent.test(env, n_episodes=1)
