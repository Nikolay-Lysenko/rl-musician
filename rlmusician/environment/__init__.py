"""
Create environment such that RL agent can interact with it.

Author: Nikolay Lysenko
"""


from . import environment, piece, scoring
from .environment import CounterpointEnv
from .piece import Piece


__all__ = ['CounterpointEnv', 'Piece', 'environment', 'piece', 'scoring']
