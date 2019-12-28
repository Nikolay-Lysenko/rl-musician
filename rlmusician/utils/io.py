"""
Read from some formats and write to some formats.

Author: Nikolay Lysenko
"""


from pkg_resources import resource_filename

import pretty_midi
from sinethesizer.io import (
    convert_tsv_to_timeline, create_timbres_registry, write_timeline_to_wav
)
from sinethesizer.io.utils import get_list_of_notes


def create_midi_from_piece(
        piece: 'rlmusician.environment.Piece',
        midi_path: str, measure_in_seconds: float,
        instrument: int, velocity: int, trailing_silence_in_measures: int = 2
) -> None:
    """
    Create MIDI file from a piece created by this package.

    :param piece:
        `Piece` instance
    :param midi_path:
        path where resulting MIDI file is going to be saved
    :param measure_in_seconds:
        duration of one measure in seconds
    :param instrument:
        ID (number) of instrument according to General MIDI specification
    :param velocity:
        one common velocity for all notes
    :param trailing_silence_in_measures:
        number of measures with silence to add at the end of the composition
    :return:
        None
    """
    numeration_shift = pretty_midi.note_name_to_number('A0')
    pretty_midi_instrument = pretty_midi.Instrument(program=instrument)
    for line in piece.lines:
        for measure, element in enumerate(line):
            if element is None:  # Dead end occurred during piece creation.
                continue
            start_time = measure * measure_in_seconds
            end_time = start_time + measure_in_seconds
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=element.absolute_position + numeration_shift,
                start=start_time,
                end=end_time
            )
            pretty_midi_instrument.notes.append(note)
    for i in range(trailing_silence_in_measures):
        start_time = (piece.n_measures + i) * measure_in_seconds
        end_time = start_time + measure_in_seconds
        note = pretty_midi.Note(
            velocity=0,
            pitch=1,  # Arbitrary value that affects nothing.
            start=start_time,
            end=end_time
        )
        pretty_midi_instrument.notes.append(note)
    pretty_midi_instrument.notes.sort(key=lambda x: x.start)
    composition = pretty_midi.PrettyMIDI()
    composition.instruments.append(pretty_midi_instrument)
    composition.write(midi_path)


def create_events_from_piece(
        piece: 'rlmusician.environment.Piece',
        events_path: str, measure_in_seconds: float,
        timbre: str, volume: float, location: int = 0, effects: str = ''
) -> None:
    """
    Create TSV file with `sinethesizer` events from a piece.

    :param piece:
        `Piece` instance
    :param events_path:
        path to a file where result is going to be saved
    :param measure_in_seconds:
        duration of one measure in seconds
    :param timbre:
        timbre to be used
    :param volume:
        relative volume of sound to be played
    :param location:
        position of imaginary sound source
    :param effects:
        sound effects to be applied to the resulting event
    :return:
        None
    """
    all_notes = get_list_of_notes()
    events = []
    for line in piece.lines:
        for measure, element in enumerate(line):
            if element is None:  # Dead end occurred during piece creation.
                continue
            start_time = measure * measure_in_seconds
            duration = measure_in_seconds
            note = all_notes[element.absolute_position]
            event = (start_time, duration, note, element.absolute_position)
            events.append(event)
    events = sorted(events, key=lambda x: (x[0], x[3], x[1]))
    events = [
        f"{timbre}\t{x[0]}\t{x[1]}\t{x[2]}\t{volume}\t{location}\t{effects}"
        for x in events
    ]

    columns = [
        'timbre', 'start_time', 'duration', 'frequency', 'volume',
        'location', 'effects'
    ]
    header = '\t'.join(columns)
    results = [header] + events
    with open(events_path, 'w') as out_file:
        for line in results:
            out_file.write(line + '\n')


def create_wav_from_events(events_path: str, output_path: str) -> None:
    """
    Create WAV file based on `sinethesizer` TSV file.

    :param events_path:
        path to TSV file with track represented as `sinethesizer` events
    :param output_path:
        path where resulting WAV file is going to be saved
    :return:
        None
    """
    presets_path = resource_filename(
        'rlmusician',
        'configs/sinethesizer_presets.yml'
    )
    settings = {
        'frame_rate': 44100,
        'trailing_silence': 2,
        'max_channel_delay': 0.02,
        'timbres_registry': create_timbres_registry(presets_path)
    }
    timeline = convert_tsv_to_timeline(events_path, settings)
    write_timeline_to_wav(output_path, timeline, settings['frame_rate'])
