"""
Test `rlmusician.utils.utils` module.

Author: Nikolay Lysenko
"""


from typing import Any, Dict, List

import pytest

from rlmusician.utils import (
    add_reference_size_for_repetitiveness,
    create_wav_from_events
)


@pytest.mark.parametrize(
    "settings, expected",
    [
        (
            # `settings`
            {
                'environment': {
                    'n_semitones': 2,
                    'n_roll_steps': 3,
                    'scoring_coefs': {
                        'vertical_variance': 1,
                        'repetitiveness': 1
                    },
                    'scoring_fn_params': {}
                }
            },
            # `expected`
            True
        ),
        (
            # `settings`
            {
                'environment': {
                    'n_semitones': 2,
                    'n_roll_steps': 3,
                    'scoring_coefs': {
                        'vertical_variance': 1
                    },
                    'scoring_fn_params': {}
                }
            },
            # `expected`
            False
        ),
    ]
)
def test_add_reference_size_for_repetitiveness(
        settings: Dict[str, Any], expected: bool
) -> None:
    """Test `add_reference_size_for_repetitiveness` function."""
    new_settings = add_reference_size_for_repetitiveness(settings)
    result = (
        'reference_size' in
        new_settings['environment']['scoring_fn_params']
        .get('repetitiveness', {})
    )
    assert result == expected


@pytest.mark.parametrize(
    "tsv_content",
    [
        (
            [
                "timbre\tstart_time\tduration\tfrequency\tvolume\tlocation\teffects",
                "fm_sine\t1\t1\tA0\t1\t0\t",
                'fm_sine\t2\t1\t1\t1\t0\t[{"name": "tremolo", "frequency": 1}]'
            ]
        )
    ]
)
def test_create_wav_from_events(
        path_to_tmp_file: str, path_to_another_tmp_file: str,
        tsv_content: List[str]
) -> None:
    """Test `create_wav_from_events` function."""
    with open(path_to_tmp_file, 'w') as tmp_tsv_file:
        for line in tsv_content:
            tmp_tsv_file.write(line + '\n')
    create_wav_from_events(path_to_tmp_file, path_to_another_tmp_file)
