"""MIDI output utilities."""

from pathlib import Path

from muscriptor.tokenizer.notes import Note, note2note_event, note_event2midi


def notes_to_midi(
    notes: list[Note],
    velocity: int = 100,
    tempo_bpm: float = 120,
    program_names: dict[int, str] | None = None,
):
    """Convert a list of Note objects to a mido MidiFile.

    `tempo_bpm` is stamped into the file and used to convert the notes'
    wall-clock seconds to ticks, so playback timing is identical at any
    value — it only decides where the beat grid falls.

    `program_names` maps program numbers to human-readable track names
    (see note_event2midi).
    """
    note_events = note2note_event(notes)
    tempo_us = int(60_000_000 / tempo_bpm)
    return note_event2midi(
        note_events,
        output_file=None,
        velocity=velocity,
        tempo=tempo_us,
        program_names=program_names,
    )


def save_midi(
    notes: list[Note],
    path: str | Path,
    velocity: int = 100,
    tempo_bpm: float = 120,
    program_names: dict[int, str] | None = None,
) -> None:
    """Save a list of Note objects as a MIDI file."""
    midi = notes_to_midi(
        notes, velocity=velocity, tempo_bpm=tempo_bpm, program_names=program_names
    )
    midi.save(str(path))
