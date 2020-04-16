"""
Run agent training and testing.

Author: Nikolay Lysenko
"""


import argparse
import os
from pkg_resources import resource_filename

import yaml

from rlmusician.agent import optimize_with_monte_carlo_beam_search
from rlmusician.environment import CounterpointEnv, Piece


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
    cli_args = parser.parse_args()
    return cli_args


def main() -> None:
    """Parse CLI arguments, train agent, and test it."""
    cli_args = parse_cli_args()

    default_config_path = 'configs/default_config.yml'
    default_config_path = resource_filename(__name__, default_config_path)
    config_path = cli_args.config_path or default_config_path
    with open(config_path) as config_file:
        settings = yaml.safe_load(config_file)

    results_dir = settings['piece']['rendering_params']['dir']
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    piece = Piece(**settings['piece'])
    env = CounterpointEnv(piece, **settings['environment'])
    best_action_sequences = optimize_with_monte_carlo_beam_search(
        env, **settings['agent']
    )

    env.verbose = True
    for i_episode, action_sequence in enumerate(best_action_sequences):
        print(f"\nPiece #{i_episode}:")
        env.reset()
        for action in action_sequence:
            observation, reward, done, info = env.step(action)
        env.render()
        print(f"Reward is {reward}.")


if __name__ == '__main__':
    main()
