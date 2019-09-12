"""
Read from some formats and write to some formats.

Author: Nikolay Lysenko
"""


from pkg_resources import resource_filename

from sinethesizer.io import (
    convert_tsv_to_timeline, create_timbres_registry, write_timeline_to_wav
)


def create_wav_from_events(events_path: str, output_path: str) -> None:
    """
    Create WAV file.

    :param events_path:
        path to TSV file with track represented as `sinethesizer` events
    :param output_path:
        path where resulting WAV file is going to be saved
    :return:
        None
    """
    presets_path = resource_filename(__name__, 'sinethesizer_presets.yml')
    settings = {
        'frame_rate': 44100,
        'trailing_silence': 2,
        'max_channel_delay': 0.02,
        'timbres_registry': create_timbres_registry(presets_path)
    }
    timeline = convert_tsv_to_timeline(events_path, settings)
    write_timeline_to_wav(output_path, timeline, settings['frame_rate'])
