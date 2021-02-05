"""
Read from some formats and write to some formats.

Author: Nikolay Lysenko
"""


import os
import subprocess
import traceback
from pkg_resources import resource_filename
from typing import List

import pretty_midi
from sinethesizer.io import (
    convert_events_to_timeline,
    convert_tsv_to_events,
    create_instruments_registry,
    write_timeline_to_wav
)
from sinethesizer.utils.music_theory import get_list_of_notes


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
        cantus_firmus_instrument: str,
        counterpoint_instrument: str,
        velocity: float,
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
    :param cantus_firmus_instrument:
        instrument to be used to play cantus firmus
    :param counterpoint_instrument:
        instrument to be used to play counterpoint line
    :param velocity:
        one common velocity for all notes
    :param effects:
        sound effects to be applied to the resulting event
    :return:
        None
    """
    all_notes = get_list_of_notes()
    eight_in_seconds = measure_in_seconds / N_EIGHTHS_PER_MEASURE
    events = []
    lines = [piece.cantus_firmus, piece.counterpoint]
    line_ids = ['cantus_firmus', 'counterpoint']
    instruments = [cantus_firmus_instrument, counterpoint_instrument]
    for line, line_id, instrument in zip(lines, line_ids, instruments):
        for element in line:
            start_time = element.start_time_in_eighths * eight_in_seconds
            duration = (
                (element.end_time_in_eighths - element.start_time_in_eighths)
                * eight_in_seconds
            )
            pitch_id = element.scale_element.position_in_semitones
            note = all_notes[pitch_id]
            event = (instrument, start_time, duration, note, pitch_id, line_id)
            events.append(event)
    events = sorted(events, key=lambda x: (x[1], x[4], x[2]))
    events = [
        f"{x[0]}\t{x[1]}\t{x[2]}\t{x[3]}\t{velocity}\t{effects}\t{x[5]}"
        for x in events
    ]

    columns = [
        'instrument', 'start_time', 'duration', 'frequency',
        'velocity', 'effects', 'line_id'
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
        'frame_rate': 48000,
        'trailing_silence': 2,
        'instruments_registry': create_instruments_registry(presets_path)
    }
    events = convert_tsv_to_events(events_path, settings)
    timeline = convert_events_to_timeline(events, settings)
    write_timeline_to_wav(output_path, timeline, settings['frame_rate'])


def make_lilypond_template(tonic: str, scale_type: str) -> str:
    """
    Make template of Lilypond text file.

    :param tonic:
        tonic pitch class represented by letter (like C or A#)
    :param scale_type:
        type of scale (e.g., 'major', 'natural_minor', or 'harmonic_minor')
    :return:
        template
    """
    raw_template = (
        "\\version \"2.18.2\"\n"
        "\\layout {{{{\n"
        "    indent = #0\n"
        "}}}}\n"
        "\\new StaffGroup <<\n"
        "    \\new Staff <<\n"
        "        \\clef treble\n"
        "        \\time 4/4\n"
        "        \\key {} \\{}\n"
        "        {{{{{{}}}}}}\n"
        "        \\\\\n"
        "        {{{{{{}}}}}}\n"
        "    >>\n"
        ">>"
    )
    tonic = tonic.replace('#', 'is').replace('b', 'es').lower()
    scale_type = scale_type.split('_')[-1]
    template = raw_template.format(tonic, scale_type)
    return template


def convert_to_lilypond_note(
        line_element: 'rlmusician.environment.piece.LineElement'
) -> str:
    """
    Convert `LineElement` instance to note in Lilypond absolute notation.

    :param line_element:
        element of a melodic line
    :return:
        note in Lilypond absolute notation
    """
    pitch_class = line_element.scale_element.note[:-1]
    pitch_class = pitch_class.replace('#', 'is').replace('b', 'es')
    pitch_class = pitch_class.lower()

    octave_id = int(line_element.scale_element.note[-1])
    lilypond_default_octave_id = 3
    octave_diff = octave_id - lilypond_default_octave_id
    octave_sign = "'" if octave_diff >= 0 else ','
    octave_info = "".join(octave_sign for _ in range(abs(octave_diff)))

    start_time = line_element.start_time_in_eighths
    end_time = line_element.end_time_in_eighths
    time_from_measure_start = start_time % N_EIGHTHS_PER_MEASURE
    duration_in_measures = (end_time - start_time) / N_EIGHTHS_PER_MEASURE
    if duration_in_measures == 1.0 and time_from_measure_start > 0:
        filled_measure_share = time_from_measure_start / N_EIGHTHS_PER_MEASURE
        remaining_duration = int(round(1 / (1 - filled_measure_share)))
        remaining_note = f"{pitch_class}{octave_info}{remaining_duration}~"
        left_over_bar_duration = int(round(1 / filled_measure_share))
        left_over_note = f"{pitch_class}{octave_info}{left_over_bar_duration}"
        return f"{remaining_note} {left_over_note}"
    else:
        duration = int(round((1 / duration_in_measures)))
        note = f"{pitch_class}{octave_info}{duration}"
        return note


def combine_lilypond_voices(
        counterpoint_voice: str,
        cantus_firmus_voice: str,
        is_counterpoint_above: bool,
        counterpoint_start_pause_in_eighths: int
) -> List[str]:
    """
    Sort Lilypond voices and add delay to counterpoint voice if needed.

    :param counterpoint_voice:
        Lilypond representation of counterpoint line (without pauses)
    :param cantus_firmus_voice:
        Lilypond representation of cantus firmus line
    :param is_counterpoint_above:
        indicator whether counterpoint is written above cantus firmus
    :param counterpoint_start_pause_in_eighths:
        duration of pause that opens counterpoint line (in eighths of measure)
    :return:
        combined Lilypond representations
    """
    if counterpoint_start_pause_in_eighths > 0:
        pause_duration = int(round(
            N_EIGHTHS_PER_MEASURE / counterpoint_start_pause_in_eighths
        ))
        pause = f'r{pause_duration}'
        counterpoint_voice = pause + ' ' + counterpoint_voice
    if is_counterpoint_above:
        return [counterpoint_voice, cantus_firmus_voice]
    else:
        return [cantus_firmus_voice, counterpoint_voice]


def create_lilypond_file_from_piece(
        piece: 'rlmusician.environment.Piece',
        output_path: str
) -> None:
    """
    Create text file in format of Lilypond sheet music editor.

    :param piece:
        musical piece
    :param output_path:
        path where resulting file is going to be saved
    :return:
        None
    """
    template = make_lilypond_template(piece.tonic, piece.scale_type)
    lilypond_voices = {}
    melodic_lines = {
        'counterpoint': piece.counterpoint,
        'cantus_firmus': piece.cantus_firmus
    }
    for line_id, melodic_line in melodic_lines.items():
        lilypond_voice = []
        for line_element in melodic_line:
            note = convert_to_lilypond_note(line_element)
            lilypond_voice.append(note)
        lilypond_voice = " ".join(lilypond_voice)
        lilypond_voices[line_id] = lilypond_voice
    lilypond_voices = combine_lilypond_voices(
        lilypond_voices['counterpoint'],
        lilypond_voices['cantus_firmus'],
        piece.is_counterpoint_above,
        piece.counterpoint_specifications['start_pause_in_eighths']
    )
    result = template.format(*lilypond_voices)
    with open(output_path, 'w') as out_file:
        out_file.write(result)


def create_pdf_sheet_music_with_lilypond(
        lilypond_path: str
) -> None:  # pragma: no cover
    """
    Create PDF file with sheet music.

    :param lilypond_path:
        path to a text file in Lilypond format
    :return:
        None:
    """
    dir_path, filename = os.path.split(lilypond_path)
    bash_command = f"lilypond {filename}"
    try:
        process = subprocess.Popen(
            bash_command.split(),
            cwd=dir_path,
            stdout=subprocess.PIPE
        )
        process.communicate()
    except Exception:
        print("Rendering sheet music to PDF failed. Do you have Lilypond?")
        print(traceback.format_exc())
