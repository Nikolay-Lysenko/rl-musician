"""
Do auxiliary tasks.

Author: Nikolay Lysenko
"""


from .io import (
    create_events_from_piece,
    create_lilypond_file_from_piece,
    create_midi_from_piece,
    create_pdf_sheet_music_with_lilypond,
    create_wav_from_events,
)
from .misc import (
    convert_to_base,
    imap_in_parallel,
    generate_copies,
    rolling_aggregate,
)
from .music_theory import Scale, ScaleElement, check_consonance


__all__ = [
    'Scale',
    'ScaleElement',
    'check_consonance',
    'convert_to_base',
    'create_events_from_piece',
    'create_lilypond_file_from_piece',
    'create_midi_from_piece',
    'create_pdf_sheet_music_with_lilypond',
    'create_wav_from_events',
    'imap_in_parallel',
    'generate_copies',
    'rolling_aggregate',
]
