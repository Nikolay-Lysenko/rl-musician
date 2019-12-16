"""
Run agent training and testing.

Author: Nikolay Lysenko
"""


import argparse
import datetime
import os
from pkg_resources import resource_filename
from typing import Any, Dict

import numpy as np
import yaml

from rlmusician.agent import (
    CounterpointEnvAgent,
    create_actor_model, find_n_weights_by_params, optimize_with_cem
)
from rlmusician.environment import CounterpointEnv, Piece


def evaluate_agent_weights(
        flat_weights: np.ndarray,
        piece_params: Dict[str, Any],
        environment_params: Dict[str, Any],
        agent_params: Dict[str, Any]
) -> float:
    """
    Evaluate weights of actor model for an agent.

    :param flat_weights:
        1D array of weights to be evaluated
    :param piece_params:
        settings of `Piece` instance
    :param environment_params:
        settings of environment
    :param agent_params:
        settings of agent
    :return:
        reward earned by the agent with the given weights
    """
    piece = Piece(**piece_params)
    env = CounterpointEnv(piece, **environment_params)
    agent = CounterpointEnvAgent(**agent_params)
    agent.set_weights(flat_weights)
    reward = agent.run_episode(env)
    return reward


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
        '-p', '--populations', type=int, default=100,
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
    agent_params = {
        'model_fn': create_actor_model,
        'observation_len': env.observation_space.shape[0],
        'n_lines': len(piece.lines),
        'n_movements_per_line': len(piece.all_movements),
        **settings['agent']
    }
    target_fn_kwargs = {
        'piece_params': settings['piece'],
        'environment_params': settings['environment'],
        'agent_params': agent_params
    }
    n_weights = find_n_weights_by_params(agent_params)
    best_weights = optimize_with_cem(
        evaluate_agent_weights,
        target_fn_kwargs=target_fn_kwargs,
        n_populations=cli_args.populations,
        initial_mean=np.array([0 for _ in range(n_weights)]),
        **settings['crossentropy']
    )

    agent = CounterpointEnvAgent(**agent_params)
    agent.set_weights(best_weights)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S,%f")
    weights_path = os.path.join(results_dir, f'agent_weights_{now}.h5f')
    agent.model.save_weights(weights_path)

    env.verbose = True
    for i_episode in range(cli_args.episodes):
        print(f"\nEpisode {i_episode}:")
        reward = agent.run_episode(env)
        env.render()
        print(f"Reward is {reward}.")


if __name__ == '__main__':
    main()
