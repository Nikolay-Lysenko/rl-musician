"""
Test `rlmusician.agent.beam_search_monte_carlo` module.

Author: Nikolay Lysenko
"""


from typing import Any, Dict, List

import pytest

from rlmusician.agent.monte_carlo_beam_search import (
    Record,
    create_stubs,
    optimize_with_monte_carlo_beam_search,
    select_distinct_best_records
)
from rlmusician.environment import CounterpointEnv, Piece


@pytest.mark.parametrize(
    "records, n_stubs, stub_length, include_finalized_sequences, expected",
    [
        (
            # `records`
            [
                Record(actions=[1, 2, 3], reward=5),
                Record(actions=[1, 3], reward=5),
                Record(actions=[1, 2, 2], reward=4),
                Record(actions=[3, 2, 1], reward=3),
                Record(actions=[2, 3, 2], reward=2),
            ],
            # `n_stubs`
            2,
            # `stub_length`
            2,
            # `include_finalized_sequences`
            True,
            # `expected`
            [
                [1, 2],
            ]
        ),
        (
            # `records`
            [
                Record(actions=[1, 2, 3], reward=5),
                Record(actions=[1, 3], reward=5),
                Record(actions=[1, 2, 2], reward=4),
                Record(actions=[3, 2, 1], reward=3),
                Record(actions=[2, 3, 2], reward=2),
            ],
            # `n_stubs`
            2,
            # `stub_length`
            2,
            # `include_finalized_sequences`
            False,
            # `expected`
            [
                [1, 2],
                [3, 2],
            ]
        ),
        (
            # `records`
            [],
            # `n_stubs`
            2,
            # `stub_length`
            3,
            # `include_finalized_sequences`
            False,
            # `expected`
            []
        ),
    ]
)
def test_create_stubs(
        records: List[Record], n_stubs: int, stub_length: int,
        include_finalized_sequences: bool, expected: List[List[int]]
) -> None:
    """Test `create_stubs` function."""
    result = create_stubs(
        records, n_stubs, stub_length, include_finalized_sequences
    )
    assert result == expected


@pytest.mark.parametrize(
    "env, beam_width, n_records_to_keep, n_trials_schedule, "
    "paralleling_params",
    [
        (
            # `env`
            CounterpointEnv(
                piece=Piece(
                    tonic='C',
                    scale_type='major',
                    cantus_firmus=['C4', 'D4', 'E4', 'D4', 'C4'],
                    counterpoint_specifications={
                        'start_note': 'E4',
                        'end_note': 'E4',
                        'lowest_note': 'G3',
                        'highest_note': 'G4',
                        'start_pause_in_eighths': 4,
                        'max_skip_in_degrees': 2,
                    },
                    rules={
                        'names': ['rearticulation_stability'],
                        'params': {}
                    },
                    rendering_params={}
                ),
                reward_for_dead_end=-100,
                scoring_coefs={'entropy': 1},
                scoring_fn_params={},
            ),
            # `beam_width`
            1,
            # `n_records_to_keep`
            100,
            # `n_trials_schedule`
            [10],
            # `paralleling_params`
            {'n_processes': 1}
        ),
    ]
)
def test_optimize_with_monte_carlo_beam_search(
        env: CounterpointEnv,
        beam_width: int,
        n_records_to_keep: int,
        n_trials_schedule: List[int],
        paralleling_params: Dict[str, Any]
) -> None:
    """Test that `optimize_with_monte_carlo_beam_search` has no failures."""
    results = optimize_with_monte_carlo_beam_search(
        env, beam_width, n_records_to_keep, n_trials_schedule,
        paralleling_params
    )
    assert len(results) == beam_width


@pytest.mark.parametrize(
    "records, n_records, expected",
    [
        (
            # `records`
            [
                Record(actions=[1, 2, 3], reward=5),
                Record(actions=[1, 2, 3], reward=5),
                Record(actions=[1, 3, 2], reward=4),
                Record(actions=[1, 1, 1], reward=3),
                Record(actions=[1, 3, 3], reward=2),
            ],
            # `n_records`
            3,
            # `expected`
            [
                Record(actions=[1, 2, 3], reward=5),
                Record(actions=[1, 3, 2], reward=4),
                Record(actions=[1, 1, 1], reward=3),
            ]
        ),
    ]
)
def test_select_distinct_best_records(
        records: List[Record], n_records: int, expected: List[Record]
) -> None:
    """Test `select_distinct_best_records` function."""
    result = select_distinct_best_records(records, n_records)
    assert result == expected
