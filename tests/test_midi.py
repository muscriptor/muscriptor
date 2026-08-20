"""Tests for muscriptor/utils/midi.py."""

import dataclasses

import numpy as np
import pytest
from mido import MidiFile

from muscriptor.tokenizer.notes import Note
from muscriptor.utils.beats import BAR_OFFSET_MARKER, BeatGrid, read_bar_offset
from muscriptor.utils.midi import notes_to_midi


def _metas(midi, msg_type):
    return [m for track in midi.tracks for m in track if m.type == msg_type]


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
    grid = BeatGrid(bpm=90, beats_per_bar=None, first_downbeat=0.0)
    midi = notes_to_midi(_sample_notes(), beat_grid=grid)
    assert _metas(midi, "set_tempo")[0].tempo == round(60_000_000 / 90)


def test_every_note_track_repeats_the_tempo():
    """MuseScore ignores set_tempo in a note-less conductor track."""
    grid = BeatGrid(bpm=90, beats_per_bar=None, first_downbeat=0.0)
    midi = notes_to_midi(_sample_notes(), beat_grid=grid)
    for track in midi.tracks[1:]:
        tempos = [m for m in track if m.type == "set_tempo"]
        assert [m.tempo for m in tempos] == [round(60_000_000 / 90)]


def test_notes_to_midi_empty_notes():
    midi = notes_to_midi([])
    assert isinstance(midi, MidiFile)


def test_no_grid_writes_no_time_signature():
    """Default output must stay as it was: placeholder tempo, meter unstated."""
    midi = notes_to_midi(_sample_notes())
    assert _metas(midi, "time_signature") == []
    assert _metas(midi, "marker") == []
    assert _metas(midi, "set_tempo")[0].tempo == 500000


def test_grid_writes_tempo_time_signature_and_marker():
    grid = BeatGrid(bpm=103.437, beats_per_bar=4, first_downbeat=1.560)
    midi = notes_to_midi(_sample_notes(), beat_grid=grid)
    tempo = _metas(midi, "set_tempo")[0].tempo
    assert round(60_000_000 / tempo) == 103  # 580063 us/beat
    sig = _metas(midi, "time_signature")[0]
    assert (sig.numerator, sig.denominator) == (4, 4)
    assert read_bar_offset(midi) == round(grid.bar_offset(), 4)


def test_grid_without_meter_writes_tempo_only():
    grid = BeatGrid(bpm=98.5, beats_per_bar=None, first_downbeat=1.2)
    midi = notes_to_midi(_sample_notes(), beat_grid=grid)
    assert _metas(midi, "set_tempo")
    assert _metas(midi, "time_signature") == []
    assert _metas(midi, "marker") == []


def test_bar_alignment_shifts_notes_without_negative_ticks():
    grid = BeatGrid(bpm=103.437, beats_per_bar=4, first_downbeat=1.560)
    midi = notes_to_midi(_sample_notes(), beat_grid=grid)  # earliest onset is 0.0
    assert all(m.time >= 0 for track in midi.tracks for m in track)
    # The note at t=0 is delayed by exactly the recorded offset.
    played = 0.0
    for msg in midi:
        played += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            break
    assert played == pytest.approx(grid.bar_offset(), abs=0.01)


def _late_notes(beats, delay, subdivision=4):
    """Sixteenth notes on `beats`, every one `delay` seconds late."""
    fine = np.linspace(beats[0], beats[-1], (len(beats) - 1) * subdivision + 1)
    return [
        Note(
            is_drum=False,
            program=0,
            onset=float(onset + delay),
            offset=float(onset + delay + 0.05),
            pitch=60,
        )
        for onset in fine
    ]


def _first_note_time(midi):
    played = 0.0
    for msg in midi:
        played += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            return played
    raise AssertionError("no note in the MIDI")


def test_late_notes_are_moved_onto_a_detected_grid():
    """The notes follow the tracked beats, which locate them better than the model.

    Real onsets land a few milliseconds after those beats, so they are moved back
    onto them. The grid itself is written as detected: `bar_offset` stays a pure
    bar-alignment shift, which is what /auralize undoes to line the synthesis up
    with the original audio.
    """
    delay = 0.012
    beats = 0.31 + np.arange(64) * (60.0 / 120.0)
    grid = BeatGrid(bpm=120.0, beats_per_bar=4, first_downbeat=0.31, beats=beats)
    midi = notes_to_midi(_late_notes(beats, delay), beat_grid=grid)

    assert read_bar_offset(midi) == round(grid.bar_offset(), 4)
    # First onset was 0.31 + delay; corrected to 0.31, the shifted downbeat lands
    # it on a bar line rather than `delay` after one.
    played = _first_note_time(midi)
    assert played == pytest.approx(grid.bar_offset() + 0.31, abs=0.002)
    assert played % grid.bar_seconds == pytest.approx(0.0, abs=0.002)


@pytest.mark.parametrize("beats_per_bar", [4, None])
def test_correction_buys_headroom_rather_than_squashing_the_start(beats_per_bar):
    """The bar-alignment shift grows by a whole bar (or beat) to make room.

    A grid whose downbeat already sits on a bar line has nothing to absorb the
    correction, and the first note is exactly `delay` in.
    """
    delay = 0.012
    beats = np.arange(64) * (60.0 / 120.0)
    grid = BeatGrid(
        bpm=120.0, beats_per_bar=beats_per_bar, first_downbeat=0.0, beats=beats
    )
    step = grid.bar_seconds or 60.0 / grid.bpm
    midi = notes_to_midi(_late_notes(beats, delay), beat_grid=grid)

    assert read_bar_offset(midi) == pytest.approx(step, abs=0.001)
    # The first onset (delay) is corrected to 0, then shifted a whole step in, so
    # it still lands on a bar line — with room to spare instead of a clamp.
    assert _first_note_time(midi) == pytest.approx(step, abs=0.002)


def test_notes_that_ignore_the_beat_are_left_alone():
    """Nothing to measure a correction from means the notes are written as-is."""
    beats = np.arange(64) * (60.0 / 120.0)
    grid = BeatGrid(bpm=120.0, beats_per_bar=4, first_downbeat=0.31, beats=beats)
    midi = notes_to_midi(_sample_notes(), beat_grid=grid)
    assert read_bar_offset(midi) == round(grid.bar_offset(), 4)
    # The earliest onset is 0.0, so it plays at exactly the bar-alignment shift.
    assert _first_note_time(midi) == pytest.approx(grid.bar_offset(), abs=0.002)


def test_bar_offset_marker_is_machine_readable():
    grid = BeatGrid(bpm=120.0, beats_per_bar=3, first_downbeat=0.4)
    midi = notes_to_midi(_sample_notes(), beat_grid=grid)
    text = _metas(midi, "marker")[0].text
    assert text.startswith(BAR_OFFSET_MARKER)
    assert read_bar_offset(midi) > 0


# --- quantization -----------------------------------------------------------


def _sixteenth(bpm: float = 120.0) -> float:
    return 60.0 / bpm / 4


def _jittered(times, jitter: float = 0.008, duration: float = 0.1) -> list[Note]:
    """One note per time, each pushed off it by an alternating `jitter`."""
    return [
        Note(
            is_drum=False,
            program=0,
            onset=t + (-1) ** i * jitter,
            offset=t + duration,
            pitch=60 + i % 12,
        )
        for i, t in enumerate(times)
    ]


def _quantizing_grid(bpm: float = 120.0, subdivision: int = 4) -> BeatGrid:
    """A grid as detection leaves it: delay and subdivision both measured."""
    return BeatGrid(
        bpm=bpm,
        beats_per_bar=4,
        first_downbeat=0.0,
        onset_delay=0.0,
        beat_subdivision=subdivision,
    )


def _off_grid(time: float, step: float) -> float:
    """How far `time` is from the nearest multiple of `step`, in seconds."""
    return abs(time / step - round(time / step)) * step


def _note_times(midi) -> list[float]:
    """Onset seconds of every note in the file, in order."""
    times, time = [], 0.0
    for msg in midi:
        time += msg.time
        if msg.type == "note_on" and msg.velocity:
            times.append(time)
    return sorted(times)


def test_quantize_snaps_the_jitter_out():
    step = _sixteenth()
    grid = _quantizing_grid()
    notes = _jittered([i * step for i in range(64)])
    midi = notes_to_midi(notes, beat_grid=grid, quantize=True)
    offset = grid.bar_offset()
    for time in _note_times(midi):
        assert _off_grid(time - offset, step) < 0.002


def test_quantize_snaps_onto_a_triplet_grid():
    """Triplet eighths are a grid of their own; snapping them to 1/16 would ruin them."""
    grid = _quantizing_grid(subdivision=3)
    third = 60.0 / grid.bpm / 3
    notes = _jittered([i * third for i in range(64)], jitter=0.005)
    midi = notes_to_midi(notes, beat_grid=grid, quantize=True)
    for time in _note_times(midi):
        assert _off_grid(time - grid.bar_offset(), third) < 0.002


def test_quantize_keeps_notes_too_short_to_survive_rounding():
    """A hit shorter than half a grid step would otherwise round away to nothing."""
    step = _sixteenth()
    notes = _jittered([i * step for i in range(64)], jitter=0.0, duration=step / 10)
    midi = notes_to_midi(notes, beat_grid=_quantizing_grid(), quantize=True)
    assert len(_note_times(midi)) == 64


def test_no_measured_subdivision_is_nothing_to_snap_to():
    """A placeholder tempo, mostly: the notes are written as they came in."""
    notes = _jittered([i * _sixteenth() for i in range(64)])
    grid = dataclasses.replace(_quantizing_grid(), beat_subdivision=None)
    assert _note_times(
        notes_to_midi(notes, beat_grid=grid, quantize=True)
    ) == _note_times(notes_to_midi(notes, beat_grid=grid))


def test_the_grid_carries_the_subdivision_it_measured():
    """`quantize` snaps to this, so detection has to leave it on the grid."""
    beats = np.arange(64) * (60.0 / 120.0)
    grid = BeatGrid(bpm=120.0, beats_per_bar=4, first_downbeat=0.0, beats=beats)
    eighths = [
        Note(is_drum=False, program=0, onset=t, offset=t + 0.1, pitch=60)
        for t in np.arange(0, 30, 0.25) + 0.01
    ]
    measured = grid.with_onset_delay([n.onset for n in eighths])
    assert measured.beat_subdivision == 2
