"""
Test `rlmusician.environment.piece` module.

Author: Nikolay Lysenko
"""


from typing import Any, Dict, List

import pytest

from rlmusician.environment import Piece


class TestPianoRollEnv:
    """Tests for `Piece` class."""

    @pytest.mark.parametrize(
        "tonic, scale, n_measures, max_skip, line_specifications, match",
        [
            (
                 # `tonic`
                'C',
                # `scale`
                'major',
                # `n_measures`
                10,
                # `max_skip`
                2,
                # `line_specifications`
                [
                    {
                        'lowest_note': 'G4',
                        'highest_note': 'G3',
                        'start_note': 'C4',
                        'end_note': 'C4'
                    }
                ],
                # `match`
                "No pitches from .*"
            ),
            (
                # `tonic`
                'C',
                # `scale`
                'major',
                # `n_measures`
                10,
                # `max_skip`
                2,
                # `line_specifications`
                [
                    {
                        'lowest_note': 'G3',
                        'highest_note': 'G4',
                        'start_note': 'C#4',
                        'end_note': 'C4'
                    }
                ],
                # `match`
                ".* does not belong to .*"
            ),
            (
                # `tonic`
                'C',
                # `scale`
                'major',
                # `n_measures`
                10,
                # `max_skip`
                2,
                # `line_specifications`
                [
                    {
                        'lowest_note': 'G3',
                        'highest_note': 'G4',
                        'start_note': 'C4',
                        'end_note': 'D4'
                    }
                ],
                # `match`
                ".* is not a tonic triad member for .*"
            ),
        ]
    )
    def test_improper_initialization(
            self, tonic: str, scale: str, n_measures: int, max_skip: int,
            line_specifications: List[Dict[str, Any]], match: str
    ) -> None:
        """Test that initialization with invalid values is prohibited."""
        with pytest.raises(ValueError, match=match):
            _ = Piece(
                tonic, scale, n_measures, max_skip, line_specifications, {}
            )
