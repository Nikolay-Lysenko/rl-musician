"""
Test `rlmusician.utils.music_theory` module.

Author: Nikolay Lysenko
"""


from typing import List, Tuple

import pytest

from rlmusician.utils.music_theory import create_scale


@pytest.mark.parametrize(
    "tonic, scale_type, expected",
    [
        (
            # `tonic`
            'C',
            # `scale_type`
            'major',
            # `expected`
            [(0, 6), (2, 7), (3, 1), (5, 2), (7, 3), (8, 4), (10, 5)]
        ),
        (
            # `tonic`
            'C',
            # `scale_type`
            'natural_minor',
            # `expected`
            [(1, 7), (3, 1), (5, 2), (6, 3), (8, 4), (10, 5), (11, 6)]
        ),
    ]
)
def test_create_scale(
        tonic: str, scale_type: str, expected: List[Tuple[int, int]]
) -> None:
    """Test `create_scale` function."""
    scale = create_scale(tonic, scale_type)
    result = [(x.absolute_position, x.degree) for x in scale[:7]]
    assert result == expected
