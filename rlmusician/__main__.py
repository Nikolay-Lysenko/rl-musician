"""
Run agent training and testing.

Author: Nikolay Lysenko
"""


import os
from pkg_resources import resource_string

import yaml

from rlmusician.agent import create_actor_model, CrossEntropyAgent
from rlmusician.environment import MusicCompositionEnv
from rlmusician.utils import add_reference_size_for_repetitiveness


def main() -> None:
    """Run all."""
    # TODO: Read user-defined path to config and data directory.
    config = resource_string(__name__, 'default_config.yml')
    settings = yaml.safe_load(config)

    package_dir = os.path.join(os.path.dirname(__file__), '..')
    data_dir = os.path.join(package_dir, settings['environment']['data_dir'])
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
        os.mkdir(os.path.join(data_dir, 'piano_rolls'))

    settings = add_reference_size_for_repetitiveness(settings)

    env = MusicCompositionEnv(**settings['environment'])
    observation_shape = env.observation_space.shape
    n_actions = env.action_space.n
    model = create_actor_model(observation_shape, n_actions)
    agent = CrossEntropyAgent(model)

    agent.fit(env, n_populations=10)
    weights_path = os.path.join(data_dir, 'agent_weights.h5f')
    agent.model.save_weights(weights_path, overwrite=True)
    agent.test(env, n_episodes=3)


if __name__ == '__main__':
    main()
