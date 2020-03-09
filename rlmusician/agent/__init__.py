"""
Create trainable agent and define methods to train it.

Author: Nikolay Lysenko
"""


from . import agent, crossentropy, policy
from .agent import CounterpointEnvAgent, extract_initial_weights
from .crossentropy import optimize_with_cem
from .policy import create_policy


__all__ = [
    'CounterpointEnvAgent',
    'agent',
    'create_policy',
    'crossentropy',
    'extract_initial_weights',
    'optimize_with_cem',
    'policy'
]
