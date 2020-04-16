"""
Create trainable agent and define methods to train it.

Author: Nikolay Lysenko
"""


from . import monte_carlo_beam_search
from .monte_carlo_beam_search import optimize_with_monte_carlo_beam_search


__all__ = [
    'monte_carlo_beam_search',
    'optimize_with_monte_carlo_beam_search'
]
