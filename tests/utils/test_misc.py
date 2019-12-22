"""
Test `rlmusician.utils.misc` module.

Author: Nikolay Lysenko
"""


from typing import List, Optional

import pytest

from rlmusician.utils.misc import convert_to_base


@pytest.mark.parametrize(
    "number, base, min_length, expected",
    [
        (12, 2, None, [1, 1, 0, 0]),
        (0, 5, None, [0]),
        (0, 5, 2, [0, 0]),
        (12, 8, 6, [0, 0, 0, 0, 1, 4]),
    ]
)
def test_convert_to_base(
        number: int, base: int, min_length: Optional, expected: List[int]
) -> None:
    """Test `convert_to_base` function."""
    result = convert_to_base(number, base, min_length)
    assert result == expected
