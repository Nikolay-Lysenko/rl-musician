"""
Run agent training and testing.

Author: Nikolay Lysenko
"""


import os
from pkg_resources import resource_string

import yaml

from rlmusician.agent import create_dqn_agent
from rlmusician.environment import MusicCompositionEnv


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

    env = MusicCompositionEnv(**settings['environment'])
    agent = create_dqn_agent(
        env.observation_space[0].shape,
        env.action_space.n
    )

    agent.fit(env, nb_steps=1e4)
    weights_path = os.path.join(data_dir, 'agent_weights.h5f')
    agent.save_weights(weights_path, overwrite=True)
    agent.test(env, nb_episodes=10, visualize=True)


if __name__ == '__main__':
    main()
