"""
Run agent training and testing.

Author: Nikolay Lysenko
"""


import argparse
import datetime
import os
from pkg_resources import resource_filename

import yaml

from rlmusician.agent import create_actor_model, CrossEntropyAgent
from rlmusician.environment import PianoRollEnv
from rlmusician.utils import add_reference_size_for_repetitiveness


def parse_cli_args() -> argparse.Namespace:
    """
    Parse arguments passed via Command Line Interface (CLI).

    :return:
        namespace with arguments
    """
    parser = argparse.ArgumentParser(description='Music composition with RL')
    parser.add_argument(
        '-c', '--config_path', type=str, default=None,
        help='path to configuration file'
    )
    parser.add_argument(
        '-p', '--populations', type=int, default=10,
        help='number of populations for agent training'
    )
    parser.add_argument(
        '-e', '--episodes', type=int, default=3,
        help='number of episodes for testing agent after its training'
    )
    cli_args = parser.parse_args()
    return cli_args


def main() -> None:
    """Run all."""
    cli_args = parse_cli_args()

    default_config_path = resource_filename(__name__, 'default_config.yml')
    config_path = cli_args.config_path or default_config_path
    with open(config_path) as config_file:
        settings = yaml.safe_load(config_file)
    settings = add_reference_size_for_repetitiveness(settings)

    results_dir = settings['environment']['rendering_params']['dir']
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    env = PianoRollEnv(**settings['environment'])
    observation_shape = env.observation_space.shape
    n_actions = env.action_space.n
    model = create_actor_model(observation_shape, n_actions)
    agent = CrossEntropyAgent(model, **settings['agent'])

    agent.fit(env, n_populations=cli_args.populations)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S,%f")
    weights_path = os.path.join(results_dir, f'agent_weights_{now}.h5f')
    agent.model.save_weights(weights_path)
    agent.test(env, n_episodes=cli_args.episodes)


if __name__ == '__main__':
    main()
