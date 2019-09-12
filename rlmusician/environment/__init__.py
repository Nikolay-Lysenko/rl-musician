"""
Create environment such that RL agent can interact with it.

Author: Nikolay Lysenko
"""


from . import environment, scoring
from .environment import PianoRollEnv


__all__ = ['PianoRollEnv', 'environment', 'scoring']
