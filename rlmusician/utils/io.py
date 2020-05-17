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


N_EIGHTHS_PER_MEASURE = 8


def create_midi_from_piece(
        piece: 'rlmusician.environment.Piece',
        midi_path: str,
        measure_in_seconds: float,
        cantus_firmus_instrument: int,
        counterpoint_instrument: int,
        velocity: int,
        trailing_silence_in_measures: int = 2
) -> None:
    """
    Create MIDI file from a piece created by this package.

    :param piece:
        `Piece` instance
    :param midi_path:
        path where resulting MIDI file is going to be saved
    :param measure_in_seconds:
        duration of one measure in seconds
    :param cantus_firmus_instrument:
        for an instrument that plays cantus firmus, its ID (number)
        according to General MIDI specification
    :param counterpoint_instrument:
        for an instrument that plays counterpoint line, its ID (number)
        according to General MIDI specification
    :param velocity:
        one common velocity for all notes
    :param trailing_silence_in_measures:
        number of measures with silence to add at the end of the composition
    :return:
        None
    """
    numeration_shift = pretty_midi.note_name_to_number('A0')
    lines = [
        piece.cantus_firmus,
        piece.counterpoint
    ]
    pretty_midi_instruments = [
        pretty_midi.Instrument(program=cantus_firmus_instrument),
        pretty_midi.Instrument(program=counterpoint_instrument)
    ]
    for line, pretty_midi_instrument in zip(lines, pretty_midi_instruments):
        for element in line:
            pitch = (
                element.scale_element.position_in_semitones
                + numeration_shift
            )
            start_time = (
                element.start_time_in_eighths
                / N_EIGHTHS_PER_MEASURE
                * measure_in_seconds
            )
            end_time = (
                element.end_time_in_eighths
                / N_EIGHTHS_PER_MEASURE
                * measure_in_seconds
            )
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start_time,
                end=end_time
            )
            pretty_midi_instrument.notes.append(note)
        pretty_midi_instrument.notes.sort(key=lambda x: x.start)

    start_time = piece.n_measures * measure_in_seconds
    end_time = start_time + trailing_silence_in_measures * measure_in_seconds
    note = pretty_midi.Note(
        velocity=0,
        pitch=1,  # Arbitrary value that affects nothing.
        start=start_time,
        end=end_time
    )
    pretty_midi_instruments[0].notes.append(note)

    composition = pretty_midi.PrettyMIDI()
    for pretty_midi_instrument in pretty_midi_instruments:
        composition.instruments.append(pretty_midi_instrument)
    composition.write(midi_path)


def create_events_from_piece(
        piece: 'rlmusician.environment.Piece',
        events_path: str,
        measure_in_seconds: float,
        cantus_firmus_timbre: str,
        counterpoint_timbre: str,
        volume: float,
        location: int = 0,
        effects: str = ''
) -> None:
    """
    Create TSV file with `sinethesizer` events from a piece.

    :param piece:
        `Piece` instance
    :param events_path:
        path to a file where result is going to be saved
    :param measure_in_seconds:
        duration of one measure in seconds
    :param cantus_firmus_timbre:
        timbre to be used to play cantus firmus
    :param counterpoint_timbre:
        timbre to be used to play counterpoint line
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
    lines = [piece.cantus_firmus, piece.counterpoint]
    timbres = [cantus_firmus_timbre, counterpoint_timbre]
    for line, timbre in zip(lines, timbres):
        for element in line:
            start_time = (
                element.start_time_in_eighths
                / N_EIGHTHS_PER_MEASURE
                * measure_in_seconds
            )
            duration = (
                (element.end_time_in_eighths - element.start_time_in_eighths)
                / N_EIGHTHS_PER_MEASURE
                * measure_in_seconds
            )
            pitch_id = element.scale_element.position_in_semitones
            note = all_notes[pitch_id]
            event = (timbre, start_time, duration, note, pitch_id)
            events.append(event)
    events = sorted(events, key=lambda x: (x[1], x[4], x[2]))
    events = [
        f"{x[0]}\t{x[1]}\t{x[2]}\t{x[3]}\t{volume}\t{location}\t{effects}"
        for x in events
    ]

    columns = [
        'timbre', 'start_time', 'duration', 'frequency',
        'volume', 'location', 'effects'
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
