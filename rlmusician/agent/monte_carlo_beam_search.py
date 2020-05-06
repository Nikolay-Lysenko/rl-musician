"""
Find optimum sequence of actions with beam search over random trials.

Author: Nikolay Lysenko
"""


import random
from typing import Any, Dict, List, Optional, NamedTuple

from rlmusician.environment import CounterpointEnv
from rlmusician.utils import generate_deep_copies, imap_in_parallel


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


def roll_out_randomly(env_with_actions: EnvWithActions) -> Record:
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


def add_records(
        env: CounterpointEnv,
        stubs: List[List[int]],
        records: List[Record],
        n_trials: int,
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
    :param n_trials:
        number of episodes to play per stub
    :param paralleling_params:
        settings of parallel playing of episodes
    :return:
        extended statistics of finished episodes as sequences of actions and
        corresponding to them rewards
    """
    for stub in stubs:
        env_with_actions = roll_in(env, stub)
        records_for_stub = imap_in_parallel(
            roll_out_randomly,
            generate_deep_copies(env_with_actions, n_trials),
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
        n_trials_schedule: List[int],
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
    :param n_trials_schedule:
        numbers of random continuations of each stub at
        corresponding iteration; the last element is used for all further
        iterations if there are any
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
        n_trials_index = min(stub_length, len(n_trials_schedule) - 1)
        n_trials = n_trials_schedule[n_trials_index]
        records = add_records(
            env, stubs, records, n_trials, paralleling_params
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
