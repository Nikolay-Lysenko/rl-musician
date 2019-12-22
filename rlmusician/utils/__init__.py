"""
Do auxiliary tasks.

Author: Nikolay Lysenko
"""


from .io import (
    create_events_from_piece,
    create_midi_from_piece,
    create_wav_from_events
)
from .misc import convert_to_base, map_in_parallel
from .music_theory import (
    get_positions_from_scale,
    get_tonic_triad_positions,
    slice_positions
)


__all__ = [
    'convert_to_base',
    'create_events_from_piece',
    'create_midi_from_piece',
    'create_wav_from_events',
    'get_positions_from_scale',
    'get_tonic_triad_positions',
    'map_in_parallel',
    'slice_positions'
]
