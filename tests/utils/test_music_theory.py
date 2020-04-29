"""
Test `rlmusician.utils.music_theory` module.

Author: Nikolay Lysenko
"""


from typing import List, Tuple

import pytest

from rlmusician.utils.music_theory import Scale, ScaleElement, check_consonance


class TestScale:
    """Tests for `Scale` class."""

    @pytest.mark.parametrize(
        "tonic, scale_type, n_elements_to_take, expected",
        [
            (
                'C',
                'major',
                7,
                [(0, 6), (2, 7), (3, 1), (5, 2), (7, 3), (8, 4), (10, 5)]
            ),
            (
                'C',
                'natural_minor',
                7,
                [(1, 7), (3, 1), (5, 2), (6, 3), (8, 4), (10, 5), (11, 6)]
            ),
        ]
    )
    def test_elements(
            self, tonic: str, scale_type: str, n_elements_to_take: int,
            expected: List[Tuple[int, int]]
    ) -> None:
        """Test that `elements` attribute is properly filled."""
        scale = Scale(tonic, scale_type)
        result = [
            (x.position_in_semitones, x.degree)
            for x in scale.elements[:n_elements_to_take]
        ]
        assert result == expected

    @pytest.mark.parametrize(
        "tonic, scale_type, note, expected",
        [
            ('C', 'major', 'G4', ScaleElement('G4', 46, 27, 5, True)),
            ('E', 'natural_minor', 'F#1', ScaleElement('F#1', 9, 5, 2, False)),
        ]
    )
    def test_get_element_by_note_with_correct_input(
            self, tonic: str, scale_type: str, note: str,
            expected: ScaleElement
    ) -> None:
        """Test `get_element_by_note` method."""
        scale = Scale(tonic, scale_type)
        result = scale.get_element_by_note(note)
        assert result == expected

    @pytest.mark.parametrize(
        "tonic, scale_type, note",
        [
            ('C', 'major', 'G#4'),
            ('E', 'natural_minor', 'F1'),
        ]
    )
    def test_get_element_by_note_with_out_of_scale_notes(
            self, tonic: str, scale_type: str, note: str
    ) -> None:
        """Test `get_element_by_note` method."""
        scale = Scale(tonic, scale_type)
        with pytest.raises(ValueError, match=f"Note {note} is not from"):
            scale.get_element_by_note(note)

    @pytest.mark.parametrize(
        "tonic, scale_type, position, expected",
        [
            ('C', 'major', 29, ScaleElement('D3', 29, 17, 2, False)),
            ('E', 'natural_minor', 21, ScaleElement('F#2', 21, 12, 2, False)),
        ]
    )
    def test_get_element_by_position_in_semitones_with_correct_input(
            self, tonic: str, scale_type: str, position: int,
            expected: ScaleElement
    ) -> None:
        """Test get_element_by_position_in_semitones` method."""
        scale = Scale(tonic, scale_type)
        result = scale.get_element_by_position_in_semitones(position)
        assert result == expected

    @pytest.mark.parametrize(
        "tonic, scale_type, position",
        [
            ('C', 'major', 1),
            ('E', 'natural_minor', 8),
        ]
    )
    def test_get_element_by_position_in_semitones_with_out_of_scale_pitches(
            self, tonic: str, scale_type: str, position: int
    ) -> None:
        """Test `get_element_by_position_in_semitones` method."""
        scale = Scale(tonic, scale_type)
        with pytest.raises(ValueError, match=f"Position {position}"):
            scale.get_element_by_position_in_semitones(position)

    @pytest.mark.parametrize(
        "tonic, scale_type, position, expected",
        [
            ('C', 'major', 18, ScaleElement('E3', 31, 18, 3, True)),
            ('E', 'natural_minor', 6, ScaleElement('G1', 10, 6, 3, True)),
        ]
    )
    def test_get_element_by_position_in_degrees_with_correct_input(
            self, tonic: str, scale_type: str, position: int,
            expected: ScaleElement
    ) -> None:
        """Test get_element_by_position_in_degrees` method."""
        scale = Scale(tonic, scale_type)
        result = scale.get_element_by_position_in_degrees(position)
        assert result == expected

    @pytest.mark.parametrize(
        "tonic, scale_type, position",
        [
            ('C', 'major', -1),
            ('E', 'natural_minor', 52),
        ]
    )
    def test_get_element_by_position_in_degrees_with_out_of_scale_degree(
            self, tonic: str, scale_type: str, position: int
    ) -> None:
        """Test `get_element_by_position_in_degrees` method."""
        scale = Scale(tonic, scale_type)
        with pytest.raises(ValueError, match=f"Position {position}"):
            scale.get_element_by_position_in_degrees(position)


@pytest.mark.parametrize(
    "first, second, is_perfect_fourth_consonant, expected",
    [
        (
            ScaleElement('B3', 38, 22, 7, False),
            ScaleElement('E4', 43, 25, 3, True),
            True,
            True
        ),
        (
            ScaleElement('B3', 38, 22, 7, False),
            ScaleElement('E4', 43, 25, 3, True),
            False,
            False
        ),
        (
            ScaleElement('B3', 38, 22, 7, False),
            ScaleElement('D4', 41, 24, 2, False),
            False,
            True
        ),
    ]
)
def test_check_consonance(
        first: ScaleElement, second: ScaleElement,
        is_perfect_fourth_consonant: bool, expected: bool
) -> None:
    """Test `check_consonance` function."""
    result = check_consonance(first, second, is_perfect_fourth_consonant)
    assert result == expected
