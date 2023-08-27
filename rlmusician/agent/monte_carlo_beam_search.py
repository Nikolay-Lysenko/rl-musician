"""
Find optimum sequence of actions with beam search over random trials.

Author: Nikolay Lysenko
"""


import functools
import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, NamedTuple

from rlmusician.environment import CounterpointEnv
from rlmusician.utils import generate_copies, imap_in_parallel


class EnvWithActions(NamedTuple):
    """A tuple of `CounterpointEnv` and actions previously applied to it."""

    env: CounterpointEnv
    actions: List[int]


class Record(NamedTuple):
    """A record with finalized sequence of actions and resulting reward."""

    actions: List[int]
    reward: float


def roll_in(env: CounterpointEnv, actions: List[int]) -> EnvWithActions:
    """
    Do roll-in actions.

    :param env:
        environment
    :param actions:
        sequence of roll-in actions
    :return:
        environment after roll-in actions
    """
    env.reset()
    for action in actions:
        env.step(action)
    env_with_actions = EnvWithActions(env, actions)
    return env_with_actions


def roll_out_randomly(env_with_actions: EnvWithActions) -> Record:  # pragma: no cover
    """
    Continue an episode in progress with random actions until it is finished.

    :param env_with_actions:
        environment and sequence of actions that have been taken before
    :return:
        finalized sequence of actions and reward for the episode
    """
    random.seed()  # Reseed to have independent results amongst processes.
    env = env_with_actions.env
    past_actions = env_with_actions.actions
    done = False
    valid_actions = env.valid_actions
    while not done:
        action = random.choice(valid_actions)
        observation, reward, done, info = env.step(action)
        past_actions.append(action)
        valid_actions = info['next_actions']
    record = Record(past_actions, reward)
    return record


def estimate_number_of_trials(
        env: CounterpointEnv,
        n_trials_estimation_depth: int,
        n_trials_estimation_width: int,
        n_trials_factor: float
) -> int:
    """
    Estimate number of trials.

    This procedure is an alternative to DFS in Monte Carlo Beam Search.
    Its advantages over DFS:
    * It is easy to run trials in parallel, no parallel DFS is needed;
    * It works even if next steps given current node are stochastic.
    Its disadvantages compared to DFS:
    * Trials are distributed less even.

    :param env:
        environment
    :param n_trials_estimation_depth:
        number of steps ahead to explore in order to collect statistics
        for inferring number of random trials to continue each stub
    :param n_trials_estimation_width:
        number of exploratory random trials that collect statistics
        for inferring number of random trials to continue each stub
    :param n_trials_factor:
        factor such that estimated number of trials is multiplied by it
    :return:
        number of trials to continue a stub at random
    """
    estimations = []
    for _ in range(n_trials_estimation_width):
        current_env = deepcopy(env)
        done = False
        valid_actions = current_env.valid_actions
        n_steps_passed = 0
        n_options = []
        while not done and n_steps_passed < n_trials_estimation_depth:
            n_options.append(len(valid_actions))
            action = random.choice(valid_actions)
            observation, reward, done, info = current_env.step(action)
            valid_actions = info['next_actions']
            n_steps_passed += 1
        estimation = functools.reduce(lambda x, y: x * y, n_options, 1)
        estimations.append(estimation)
    n_trials = n_trials_factor * sum(estimations) / n_trials_estimation_width
    n_trials = int(round(n_trials))
    return n_trials


def add_records(
        env: CounterpointEnv,
        stubs: List[List[int]],
        records: List[Record],
        n_trials_estimation_depth: int,
        n_trials_estimation_width: int,
        n_trials_factor: float,
        paralleling_params: Dict[str, Any]
) -> List[Record]:
    """
    Play new episodes given roll-in sequences and add new records with results.

    :param env:
        environment
    :param stubs:
        roll-in sequences
    :param records:
        previously collected statistics of finished episodes as sequences
        of actions and corresponding to them rewards
    :param n_trials_estimation_depth:
        number of steps ahead to explore in order to collect statistics
        for inferring number of random trials to continue each stub
    :param n_trials_estimation_width:
        number of exploratory random trials that collect statistics
        for inferring number of random trials to continue each stub
    :param n_trials_factor:
        factor such that estimated number of trials is multiplied by it
    :param paralleling_params:
        settings of parallel playing of episodes
    :return:
        extended statistics of finished episodes as sequences of actions and
        corresponding to them rewards
    """
    for stub in stubs:
        env_with_actions = roll_in(env, stub)
        n_trials = estimate_number_of_trials(
            env_with_actions.env,
            n_trials_estimation_depth,
            n_trials_estimation_width,
            n_trials_factor
        )
        records_for_stub = imap_in_parallel(
            roll_out_randomly,
            generate_copies(env_with_actions, n_trials),
            paralleling_params
        )
        records.extend(records_for_stub)
    return records


def create_stubs(
        records: List[Record],
        n_stubs: int,
        stub_length: int,
        include_finalized_sequences: bool = True
) -> List[List[int]]:
    """
    Create roll-in sequences (stubs) based on collected statistics.

    :param records:
        sorted statistics of played episodes as sequences of actions
        and corresponding to them rewards; elements must be sorted by reward
        in descending order
    :param n_stubs:
        number of stubs to be created
    :param stub_length:
        number of actions in each stub
    :param include_finalized_sequences:
        if it is set to `True`, resulting number of stubs can be less than
        `n_stubs`, because finalized sequences are also counted
    :return:
        new stubs that can be extended further (i.e., without those of them
        that are finalized)
    """
    stubs = []
    for record in records:
        if len(stubs) == n_stubs:
            break
        key = record.actions[:stub_length]
        if key in stubs:
            continue
        if len(record.actions) <= stub_length:
            if include_finalized_sequences:  # pragma: no branch
                n_stubs -= 1
            continue
        stubs.append(key)
    return stubs


def select_distinct_best_records(
        records: List[Record],
        n_records: int
) -> List[Record]:
    """
    Select records related to highest rewards (without duplicates).

    :param records:
        sorted statistics of played episodes as sequences of actions
        and corresponding to them rewards; elements must be sorted by reward
        in descending order
    :param n_records:
        number of unique records to select
    :return:
        best records
    """
    results = []
    for record in records:
        if record not in results:
            results.append(record)
        if len(results) == n_records:
            break
    return results


def optimize_with_monte_carlo_beam_search(
        env: CounterpointEnv,
        beam_width: int,
        n_records_to_keep: int,
        n_trials_estimation_depth: int,
        n_trials_estimation_width: int,
        n_trials_factor: float,
        paralleling_params: Optional[Dict[str, Any]] = None
) -> List[List[int]]:
    """
    Find optimum sequences of actions with Monte Carlo Beam Search.

    :param env:
        environment
    :param beam_width:
        number of best subsequences to be kept after each iteration
    :param n_records_to_keep:
        number of best played episodes to be kept after each iteration
    :param n_trials_estimation_depth:
        number of steps ahead to explore in order to collect statistics
        for inferring number of random trials to continue each stub
    :param n_trials_estimation_width:
        number of exploratory random trials that collect statistics
        for inferring number of random trials to continue each stub
    :param n_trials_factor:
        factor such that estimated number of trials is multiplied by it
    :param paralleling_params:
        settings of parallel playing of episodes;
        by default, number of processes is set to number of cores
        and each worker is not replaced with a newer one after some number of
        tasks are finished
    :return:
        best final sequences of actions
    """
    stubs = [[]]
    records = []
    paralleling_params = paralleling_params or {}
    stub_length = 0
    while len(stubs) > 0:
        records = add_records(
            env,
            stubs,
            records,
            n_trials_estimation_depth,
            n_trials_estimation_width,
            n_trials_factor,
            paralleling_params
        )
        records = sorted(records, key=lambda x: x.reward, reverse=True)
        print(
            f"Current best reward: {records[0].reward:.5f}, "
            f"achieved with: {records[0].actions}."
        )
        stub_length += 1
        stubs = create_stubs(records, beam_width, stub_length)
        records = select_distinct_best_records(records, n_records_to_keep)
    results = [past_actions for past_actions, reward in records[:beam_width]]
    return results
