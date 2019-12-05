"""
Do auxiliary tasks.

Author: Nikolay Lysenko
"""


from .io import create_midi_from_piano_roll, create_wav_from_events
from .misc import (
    apply_rolling_aggregation,
    map_in_parallel,
    shift_horizontally,
    shift_vertically
)
from .music_theory import (
    get_positions_from_scale,
    get_tonic_triad_positions,
    slice_positions
)


__all__ = [
    'apply_rolling_aggregation',
    'create_midi_from_piano_roll',
    'create_wav_from_events',
    'get_positions_from_scale',
    'get_tonic_triad_positions',
    'map_in_parallel',
    'shift_horizontally',
    'shift_vertically',
    'slice_positions'
]
