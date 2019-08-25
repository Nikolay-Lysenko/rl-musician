"""
Do auxiliary tasks.

Author: Nikolay Lysenko
"""


from .utils import (
    add_reference_size_for_repetitiveness,
    estimate_max_compressed_size_given_shape,
    measure_compressed_size
)


__all__ = [
    'add_reference_size_for_repetitiveness',
    'estimate_max_compressed_size_given_shape',
    'measure_compressed_size'
]
