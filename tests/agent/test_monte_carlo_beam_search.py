"""
Test `rlmusician.agent.beam_search_monte_carlo` module.

Author: Nikolay Lysenko
"""


from typing import Any, Dict, List, Tuple

import pytest

from rlmusician.agent.monte_carlo_beam_search import (
    create_stubs,
    optimize_with_monte_carlo_beam_search,
)
from rlmusician.environment import CounterpointEnv, Piece


@pytest.mark.parametrize(
    "records, n_stubs, stub_length, expected",
    [
        (
            # `records`
            [
                ([1, 2, 3], 5),
                ([1, 2, 2], 4),
                ([3, 2, 1], 3),
                ([2, 3, 2], 2),
            ],
            # `n_stubs`
            2,
            # `stub_length`
            2,
            # `expected`
            [
                [1, 2],
                [3, 2],
            ]
        ),
    ]
)
def test_create_stubs(
        records: List[Tuple[List[int], float]], n_stubs: int, stub_length: int,
        expected: List[List[int]]
) -> None:
    """Test `create_stubs` function."""
    result = create_stubs(records, n_stubs, stub_length)
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
                    n_measures=10,
                    max_skip=2,
                    line_specifications=[
                        {
                            'lowest_note': 'C4',
                            'highest_note': 'C5',
                            'start_note': 'E4',
                            'end_note': 'C4'
                        },
                    ],
                    voice_leading_rules={
                        'names': [
                            'rearticulation',
                            'destination_of_skip',
                            'turn_after_skip',
                            'VI_VII_resolution',
                            'step_motion_to_end'
                        ],
                        'params': {
                            'turn_after_skip': {
                                'min_n_scale_degrees': 3
                            },
                            'step_motion_to_end': {
                                'prohibit_rearticulation': False
                            }
                        }
                    },
                    harmony_rules={
                        'names': [],
                        'params': {}
                    },
                    rendering_params={}
                ),
                observation_decay=0.75,
                reward_for_dead_end=-100,
                scoring_coefs={'climax_explicity': 1},
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
    assert len(results[0]) == env.piece.n_measures - 2
