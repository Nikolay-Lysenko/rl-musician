"""
Create trainable agent and define methods to train it.

Author: Nikolay Lysenko
"""


from . import monte_carlo_beam_search
from .monte_carlo_beam_search import optimize_with_mcbs


__all__ = ['monte_carlo_beam_search.py', 'optimize_with_mcbs']
