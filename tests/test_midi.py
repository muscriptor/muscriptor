"""Tests for muscriptor/utils/midi.py."""

import dataclasses

import numpy as np
import pytest
import mido
from mido import MidiFile

from muscriptor.tokenizer.notes import Note
from muscriptor.utils.beats import BAR_OFFSET_MARKER, BeatGrid, read_bar_offset
from muscriptor.utils.midi import ExportOptions, notes_to_midi


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
    midi = notes_to_midi(notes, beat_grid=grid, options=ExportOptions(quantize=True))
    offset = grid.bar_offset()
    for time in _note_times(midi):
        assert _off_grid(time - offset, step) < 0.002


def test_quantize_snaps_onto_a_triplet_grid():
    """Triplet eighths are a grid of their own; snapping them to 1/16 would ruin them."""
    grid = _quantizing_grid(subdivision=3)
    third = 60.0 / grid.bpm / 3
    notes = _jittered([i * third for i in range(64)], jitter=0.005)
    midi = notes_to_midi(notes, beat_grid=grid, options=ExportOptions(quantize=True))
    for time in _note_times(midi):
        assert _off_grid(time - grid.bar_offset(), third) < 0.002


def test_quantize_keeps_notes_too_short_to_survive_rounding():
    """A hit shorter than half a grid step would otherwise round away to nothing."""
    step = _sixteenth()
    notes = _jittered([i * step for i in range(64)], jitter=0.0, duration=step / 10)
    midi = notes_to_midi(
        notes, beat_grid=_quantizing_grid(), options=ExportOptions(quantize=True)
    )
    assert len(_note_times(midi)) == 64


def test_a_placeholder_tempo_is_nothing_to_snap_to():
    """Without a tempo, detected or given, the notes are written as they came in."""
    notes = _jittered([i * _sixteenth() for i in range(64)])
    assert _note_times(
        notes_to_midi(notes, options=ExportOptions(quantize=True, subdivision=4))
    ) == _note_times(notes_to_midi(notes))


def test_unaligned_onsets_are_not_snapped_to_a_guessed_subdivision():
    """Onsets on no subdivision of the beat give quantizing nothing to go by."""
    rng = np.random.default_rng(0)
    notes = _jittered(np.sort(rng.uniform(0, 30, 64)), jitter=0.0)
    grid = dataclasses.replace(_quantizing_grid(), beat_subdivision=None)
    assert _note_times(
        notes_to_midi(notes, beat_grid=grid, options=ExportOptions(quantize=True))
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


# --- export options -------------------------------------------------------


def test_a_given_tempo_replaces_the_detected_one():
    grid = BeatGrid(bpm=103.4, beats_per_bar=4, first_downbeat=0.0)
    midi = notes_to_midi(_sample_notes(), beat_grid=grid, options=ExportOptions(bpm=90))
    assert _metas(midi, "set_tempo")[0].tempo == round(60_000_000 / 90)


def test_a_given_tempo_needs_no_detected_grid():
    midi = notes_to_midi(
        _sample_notes(), options=ExportOptions(bpm=90, beats_per_bar=3)
    )
    assert _metas(midi, "set_tempo")[0].tempo == round(60_000_000 / 90)
    sig = _metas(midi, "time_signature")[0]
    assert (sig.numerator, sig.denominator) == (3, 4)


def test_retiming_keeps_the_notes_where_they_were_in_seconds():
    """A tempo only moves bar lines: the file plays the same either way."""
    grid = _quantizing_grid()
    notes = _jittered([i * _sixteenth() for i in range(16)])
    times = _note_times(notes_to_midi(notes, beat_grid=grid))
    retimed = _note_times(
        notes_to_midi(notes, beat_grid=grid, options=ExportOptions(bpm=97))
    )
    assert retimed == pytest.approx(times, abs=0.002)


def test_a_given_downbeat_moves_the_bar_lines():
    """Bar 1 starts at tick 0, so the downbeat decides how far the notes are delayed."""
    grid = BeatGrid(bpm=120, beats_per_bar=4, first_downbeat=0.0, onset_delay=0.0)
    notes = [Note(is_drum=False, program=0, onset=1.5, offset=1.9, pitch=60)]
    midi = notes_to_midi(
        notes, beat_grid=grid, options=ExportOptions(first_downbeat=0.5)
    )
    # Bars are 2 s, with one at 0.5 s: delaying everything by 1.5 s puts it at 2 s.
    assert _first_note_time(midi) == pytest.approx(3.0, abs=0.002)


def test_a_given_subdivision_is_snapped_to():
    grid = _quantizing_grid(subdivision=4)
    eighth = 60.0 / grid.bpm / 2
    notes = _jittered([i * _sixteenth() for i in range(64)])
    midi = notes_to_midi(
        notes, beat_grid=grid, options=ExportOptions(quantize=True, subdivision=2)
    )
    for time in _note_times(midi):
        assert _off_grid(time - grid.bar_offset(), eighth) < 0.002


def test_half_strength_moves_notes_halfway():
    grid = _quantizing_grid()
    notes = [Note(is_drum=False, program=0, onset=0.51, offset=0.6, pitch=60)]
    midi = notes_to_midi(
        notes, beat_grid=grid, options=ExportOptions(quantize=True, strength=0.5)
    )
    assert _first_note_time(midi) == pytest.approx(0.505, abs=0.002)


def test_a_typed_tempo_finds_its_own_subdivision():
    """The detected subdivision belongs to the detected tempo; a new one is measured."""
    grid = _quantizing_grid(bpm=100, subdivision=3)
    step = 60.0 / 120 / 4
    notes = _jittered([i * step for i in range(96)], jitter=0.004)
    midi = notes_to_midi(
        notes, beat_grid=grid, options=ExportOptions(bpm=120, quantize=True)
    )
    for time in _note_times(midi):
        assert _off_grid(time - grid.bar_offset(), step) < 0.002


def _note_spans(midi) -> list[tuple[float, float, int]]:
    """(start, end, pitch) of every note in the file, in seconds."""
    spans, open_at, time = [], {}, 0.0
    for msg in midi:
        time += msg.time
        if msg.type == "note_on" and msg.velocity:
            open_at[msg.note] = time
        elif msg.type in ("note_on", "note_off"):
            spans.append((open_at.pop(msg.note), time, msg.note))
    return sorted(spans)


def test_fill_gaps_holds_notes_until_the_next_one():
    grid = _quantizing_grid()  # 0.5 s beats
    notes = [
        Note(is_drum=False, program=0, onset=0.0, offset=0.2, pitch=60),
        Note(is_drum=False, program=0, onset=0.5, offset=0.6, pitch=62),
        # 1.4 s of silence: a real rest, longer than the one beat allowed.
        Note(is_drum=False, program=0, onset=2.0, offset=2.1, pitch=64),
    ]
    midi = notes_to_midi(notes, beat_grid=grid, options=ExportOptions(fill_gaps=1))
    spans = [(round(a, 2), round(b, 2)) for a, b, _ in _note_spans(midi)]
    assert spans == [(0.0, 0.5), (0.5, 0.6), (2.0, 2.1)]


def test_fill_gaps_keeps_instruments_apart_and_leaves_drums_alone():
    grid = _quantizing_grid()
    notes = [
        Note(is_drum=False, program=0, onset=0.0, offset=0.2, pitch=60),
        Note(is_drum=False, program=33, onset=0.3, offset=0.4, pitch=40),
        Note(is_drum=True, program=128, onset=0.4, offset=0.4, pitch=36),
    ]
    midi = notes_to_midi(notes, beat_grid=grid, options=ExportOptions(fill_gaps=1))
    ends = {pitch: round(b - a, 2) for a, b, pitch in _note_spans(midi)}
    assert ends[60] == 0.2
    assert ends[40] == 0.1
    # Drum hits are written 1/32 beat long.
    assert ends[36] == round(0.5 / 32, 2)


def test_trim_leading_bars_drops_the_silence_before_the_first_note():
    grid = BeatGrid(bpm=120, beats_per_bar=4, first_downbeat=0.0, onset_delay=0.0)
    notes = [Note(is_drum=False, program=0, onset=9.0, offset=9.5, pitch=60)]
    midi = notes_to_midi(
        notes, beat_grid=grid, options=ExportOptions(trim_leading_bars=True)
    )
    # 2 s bars: the four empty ones go, the note keeps its place in its bar.
    assert _first_note_time(midi) == pytest.approx(1.0, abs=0.002)
    assert read_bar_offset(midi) == pytest.approx(-8.0)


# --- tempo map --------------------------------------------------------------


def _accelerating_beats(n: int = 48, start: float = 0.5) -> np.ndarray:
    """Beats speeding up from 100 to about 140 BPM: no constant tempo fits."""
    intervals = 60.0 / np.linspace(100, 140, n - 1)
    return start + np.concatenate([[0.0], np.cumsum(intervals)])


def _rubato_grid(beats: np.ndarray, beats_per_bar: int | None = 4) -> BeatGrid:
    return BeatGrid(
        bpm=120.0,
        beats_per_bar=beats_per_bar,
        first_downbeat=float(beats[0]),
        beats=beats,
        onset_delay=0.0,
        beat_subdivision=4,
        steady=False,
    )


def _on_beats(beats, every: float = 1.0) -> list[Note]:
    """A note on every `every`-th beat (fractions interpolate between beats)."""
    positions = np.arange(0, len(beats) - 1, every)
    times = np.interp(positions, np.arange(len(beats)), beats)
    return [
        Note(is_drum=False, program=0, onset=float(t), offset=float(t) + 0.05, pitch=60)
        for t in times
    ]


def _note_ticks(midi) -> list[int]:
    ticks = []
    for track in midi.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "note_on" and msg.velocity:
                ticks.append(tick)
    return sorted(ticks)


def test_an_unsteady_grid_is_written_as_a_tempo_map():
    beats = _accelerating_beats()
    midi = notes_to_midi(_on_beats(beats), beat_grid=_rubato_grid(beats))
    tempi = [m.tempo for m in midi.tracks[0] if m.type == "set_tempo"]
    assert len(tempi) > 20
    assert tempi[0] > tempi[-1]  # speeding up: fewer microseconds per beat
    # Every note lands on a beat of the score, and the first on a bar line.
    ticks = _note_ticks(midi)
    assert all(t % midi.ticks_per_beat == 0 for t in ticks)
    assert ticks[0] % (4 * midi.ticks_per_beat) == 0


def test_a_tempo_map_keeps_the_notes_where_they_were_in_seconds():
    """Played back, the file sounds like the recording, delayed by the marker."""
    beats = _accelerating_beats()
    notes = _on_beats(beats, every=0.5)
    midi = notes_to_midi(notes, beat_grid=_rubato_grid(beats))
    played = np.array(_note_times(midi)) - read_bar_offset(midi)
    assert played == pytest.approx([n.onset for n in notes], abs=0.003)


def test_a_steady_or_unwanted_tempo_map_is_one_tempo():
    beats = _accelerating_beats()
    for grid, options in [
        (dataclasses.replace(_rubato_grid(beats), steady=True), ExportOptions()),
        (_rubato_grid(beats), ExportOptions(tempo_map=False)),
        (_rubato_grid(beats), ExportOptions(bpm=110)),
    ]:
        midi = notes_to_midi(_on_beats(beats), beat_grid=grid, options=options)
        assert len(_metas(midi, "set_tempo")) == len(midi.tracks)  # once per track


def test_a_doubled_beat_puts_notes_on_every_other_beat_and_doubles_the_meter():
    beats = _accelerating_beats()
    midi = notes_to_midi(
        _on_beats(beats),
        beat_grid=_rubato_grid(beats),
        options=ExportOptions(beat_scale=2),
    )
    assert all(t % (2 * midi.ticks_per_beat) == 0 for t in _note_ticks(midi))
    assert _metas(midi, "time_signature")[0].numerator == 8


def test_a_halved_beat_puts_notes_between_beats_and_halves_the_meter():
    beats = _accelerating_beats()
    midi = notes_to_midi(
        _on_beats(beats),
        beat_grid=_rubato_grid(beats),
        options=ExportOptions(beat_scale=0.5),
    )
    # The beats dropped from the tempo map were not quite halfway between the
    # ones kept, the tempo being on the move; they still land close to it.
    half = midi.ticks_per_beat // 2
    assert all(abs(t - half * round(t / half)) <= 5 for t in _note_ticks(midi))
    assert _metas(midi, "time_signature")[0].numerator == 2


def test_a_given_downbeat_snaps_to_the_nearest_tracked_beat():
    beats = _accelerating_beats()
    notes = _on_beats(beats)
    midi = notes_to_midi(
        notes,
        beat_grid=_rubato_grid(beats),
        # Just after the third beat: that beat starts a bar now.
        options=ExportOptions(first_downbeat=float(beats[2]) + 0.03),
    )
    ticks = _note_ticks(midi)
    bar = 4 * midi.ticks_per_beat
    assert [t % bar for t in ticks[:4]] == [2 * bar // 4, 3 * bar // 4, 0, bar // 4]


def test_quantizing_follows_the_tempo_map():
    beats = _accelerating_beats()
    sixteenths = _on_beats(beats, every=0.25)
    rng = np.random.default_rng(1)
    jittered = [
        dataclasses.replace(n, onset=n.onset + rng.uniform(-0.01, 0.01))
        for n in sixteenths
    ]
    midi = notes_to_midi(
        jittered,
        beat_grid=_rubato_grid(beats),
        options=ExportOptions(quantize=True),
    )
    assert all(t % (midi.ticks_per_beat // 4) == 0 for t in _note_ticks(midi))


def test_smoothing_takes_the_trackers_frame_jitter_out_of_the_tempo():
    """beat_this reports beats on 20 ms frames: that alone must not make a tempo map."""
    exact = 0.3 + np.arange(64) * (60.0 / 123.0)
    framed = np.round(exact / 0.02) * 0.02
    midi = notes_to_midi(
        _on_beats(framed),
        beat_grid=_rubato_grid(framed),
        options=ExportOptions(tempo_map=True),
    )
    bpms = [mido.tempo2bpm(m.tempo) for m in midi.tracks[0] if m.type == "set_tempo"]
    assert max(bpms) - min(bpms) < 0.04 * 123


def test_beat_scale_must_be_a_power_of_two():
    with pytest.raises(ValueError):
        ExportOptions(beat_scale=3)
