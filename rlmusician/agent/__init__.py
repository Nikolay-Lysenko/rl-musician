"""
Create trainable agent and define methods to train it.

Author: Nikolay Lysenko
"""


from . import actor_model, agent, crossentropy
from .actor_model import create_actor_model
from .agent import CounterpointEnvAgent, find_n_weights_by_params
from .crossentropy import optimize_with_cem


__all__ = [
    'CounterpointEnvAgent',
    'actor_model',
    'agent',
    'create_actor_model',
    'crossentropy',
    'find_n_weights_by_params',
    'optimize_with_cem'
]
