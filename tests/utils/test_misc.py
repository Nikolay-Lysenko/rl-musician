"""
Test `rlmusician.utils.misc` module.

Author: Nikolay Lysenko
"""


from typing import List

import pytest

from rlmusician.utils.misc import convert_to_base


@pytest.mark.parametrize(
    "number, base, expected",
    [
        (12, 2, [1, 1, 0, 0]),
        (0, 5, [0])
    ]
)
def test_convert_to_base(number: int, base: int, expected: List[int]) -> None:
    """Test `convert_to_base` function."""
    result = convert_to_base(number, base)
    assert result == expected
