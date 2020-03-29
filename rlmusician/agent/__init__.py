"""
Create trainable agent and define methods to train it.

Author: Nikolay Lysenko
"""


from . import beam_search_monte_carlo
from .beam_search_monte_carlo import optimize_with_bsmc


__all__ = ['beam_search_monte_carlo', 'optimize_with_bsmc']
