"""Tests for muscriptor/utils/midi.py."""

import tempfile
from pathlib import Path

from mido import MidiFile

from muscriptor.tokenizer.notes import Note
from muscriptor.utils.midi import notes_to_midi, save_midi


def _sample_notes():
    return [
        Note(is_drum=False, program=0, onset=0.0, offset=0.5, pitch=60),
        Note(is_drum=False, program=0, onset=0.5, offset=1.0, pitch=64),
        Note(is_drum=True, program=128, onset=0.0, offset=0.01, pitch=36),
    ]


def test_notes_to_midi_returns_midi_file():
    midi = notes_to_midi(_sample_notes())
    assert isinstance(midi, MidiFile)


def test_notes_to_midi_has_tracks():
    midi = notes_to_midi(_sample_notes())
    assert len(midi.tracks) > 0


def test_notes_to_midi_custom_tempo():
    midi = notes_to_midi(_sample_notes(), tempo_bpm=90)
    assert isinstance(midi, MidiFile)


def test_notes_to_midi_empty_notes():
    midi = notes_to_midi([])
    assert isinstance(midi, MidiFile)


def test_save_midi_creates_file():
    notes = _sample_notes()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "out.mid"
        save_midi(notes, path)
        assert path.exists()
        assert path.stat().st_size > 0


def test_save_midi_is_valid_midi():
    notes = _sample_notes()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "out.mid"
        save_midi(notes, path)
        loaded = MidiFile(str(path))
        assert len(loaded.tracks) > 0


def test_save_midi_string_path():
    notes = _sample_notes()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "out.mid")
        save_midi(notes, path)
        assert Path(path).exists()


def test_tempo_preserves_wall_clock_time():
    """The stamped tempo shifts the beat grid, not the notes' seconds.

    A note onset at 0.5 s must land at 0.5 s regardless of tempo_bpm: the
    tick position doubles when the tempo halves.
    """
    from mido import second2tick

    for bpm in (60, 120, 174.5):
        midi = notes_to_midi(_sample_notes(), tempo_bpm=bpm)
        tempo_us = int(60_000_000 / bpm)
        tempos = [
            msg.tempo
            for track in midi.tracks
            for msg in track
            if msg.type == "set_tempo"
        ]
        assert tempos == [tempo_us]
        # First pitch-60 note_on sits at onset 0.0, the pitch-64 one at 0.5 s.
        track = next(
            t
            for t in midi.tracks
            if any(m.type == "note_on" and m.note == 64 for m in t)
        )
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "note_on" and msg.note == 64 and msg.velocity > 0:
                break
        assert tick == round(second2tick(0.5, midi.ticks_per_beat, tempo_us))


def test_events_to_midi_bytes_forwards_tempo():
    """TranscriptionModel.events_to_midi_bytes stamps the requested tempo."""
    import io
    from types import SimpleNamespace

    from muscriptor.events import NoteEndEvent, NoteStartEvent
    from muscriptor.transcription_model import TranscriptionModel

    # Drum-only events avoid the program lookup, so a bare fake self works.
    start = NoteStartEvent(pitch=36, start_time=1.0, index=0, instrument="drums")
    events = [start, NoteEndEvent(end_time=1.01, start_event=start)]
    midi_bytes = TranscriptionModel.events_to_midi_bytes(
        SimpleNamespace(), iter(events), tempo_bpm=60
    )
    midi = MidiFile(file=io.BytesIO(midi_bytes))
    tempos = [
        msg.tempo for track in midi.tracks for msg in track if msg.type == "set_tempo"
    ]
    assert tempos == [1_000_000]
