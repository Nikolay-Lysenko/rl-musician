"""
Test `rlmusician.io.io` module.

Author: Nikolay Lysenko
"""


from typing import List

import pytest

from rlmusician.io import create_wav_from_events


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
