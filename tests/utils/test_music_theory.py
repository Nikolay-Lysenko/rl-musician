"""
Test `rlmusician.utils.music_theory` module.

Author: Nikolay Lysenko
"""


from typing import List

import pytest

from rlmusician.utils.music_theory import get_positions_from_scale


@pytest.mark.parametrize(
    "tonic, scale, expected",
    [
        ('C', 'major', [0, 2, 3, 5, 7, 8, 10]),
        ('C', 'natural_minor', [1, 3, 5, 6, 8, 10, 11]),
    ]
)
def test_get_positions_from_scale(
        tonic: str, scale: str, expected: List[int]
) -> None:
    """Test `get_positions_from_scale` function."""
    result = get_positions_from_scale(tonic, scale)[:7]
    assert result == expected
