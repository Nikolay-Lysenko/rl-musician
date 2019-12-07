"""
Create environment such that RL agent can interact with it.

Author: Nikolay Lysenko
"""


from . import environment, piece, scoring
from .environment import PianoRollEnv
from .piece import Piece


__all__ = ['PianoRollEnv', 'Piece', 'environment', 'piece', 'scoring']
