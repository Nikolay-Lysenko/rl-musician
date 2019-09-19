"""
Test `rlmusician.agent.crossentropy` module.

Author: Nikolay Lysenko
"""


from typing import Dict, Tuple

import gym
import numpy as np

from rlmusician.agent.crossentropy import CrossEntropyAgent


def create_actor_model_mock(n_actions: int = 10) -> 'keras.models.Model':
    """Create mock actor model."""
    from keras.models import Model
    from keras.layers import Input, Dense

    model_input = Input(shape=(1,))
    model_output = Dense(n_actions, activation='softmax')(model_input)
    model = Model(inputs=model_input, outputs=model_output)
    model.compile(optimizer='sgd', loss='mse')  # Arbitrary unused values.
    return model


class EnvMock(gym.Env):
    """Mock for environment."""

    n_steps_per_episode = 5
    observation = np.array([1])
    reward_range = (-np.inf, np.inf)
    n_actions = 10
    action_space = gym.spaces.Discrete(n_actions)
    observation_space = gym.spaces.Discrete(1)

    def __init__(self):
        """Initialize instance."""
        self.position = 0
        self.n_steps_passed = 0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Run one step."""
        self.position += 1 if action == 0 else -1
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

    def test_fit_and_test(self) -> None:
        """Test `fit` and `test` methods."""
        env = EnvMock()
        agent = CrossEntropyAgent(
            model_fn=create_actor_model_mock,
            model_params={},
            population_size=100,
            n_episodes_per_candidate=5
        )
        agent.fit(env, n_populations=10)
        assert agent.best['score'] > 4.9
        agent.test(env, n_episodes=1)
