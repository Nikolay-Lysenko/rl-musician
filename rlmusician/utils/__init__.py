"""
Do auxiliary tasks.

Author: Nikolay Lysenko
"""


from .io import write_to_sinethesizer_format
from .misc import (
    add_reference_size_for_repetitiveness,
    estimate_max_compressed_size_given_shape,
    measure_compressed_size
)


__all__ = [
    'add_reference_size_for_repetitiveness',
    'estimate_max_compressed_size_given_shape',
    'measure_compressed_size',
    'write_to_sinethesizer_format'
]
