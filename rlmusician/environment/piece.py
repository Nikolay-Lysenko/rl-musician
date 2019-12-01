"""
Author: Nikolay Lysenko
"""


from typing import Any, Dict, List

from sinethesizer.io.utils import get_note_to_position_mapping


NOTE_TO_POSITION = get_note_to_position_mapping()


class Piece:
    """"""

    def __init__(
            self,
            tonic: str,
            scale: str,
            n_measures: int,
            line_params: Dict[str, Any],
            rendering_params: Dict[str, Any]
    ):
        """"""
        self.tonic = tonic
        self.scale = scale
        self.n_measures = n_measures
        self.line_params = line_params
        self.rendering_params = rendering_params

        self.__validate_line_params()

        self.lines = None
        self.piano_roll = None
        self.__create_lines()
        self.__create_piano_roll()

    def __validate_line_params(self) -> None:
        """"""

    def __create_lines(self) -> None:
        """"""

    def __create_piano_roll(self) -> None:
        """"""

    def validate_movements(self, movements: List[int]) -> bool:
        """"""

    def add_measure(self, movements: List[int]) -> None:
        """"""
        if not self.validate_movements(movements):
            raise ValueError('Passed movements are not permitted.')
