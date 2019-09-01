"""
Create trainable agent.

Author: Nikolay Lysenko
"""


from .actor_model import create_actor_model
from .crossentropy import CrossEntropyAgent


__all__ = ['create_actor_model', 'CrossEntropyAgent']
