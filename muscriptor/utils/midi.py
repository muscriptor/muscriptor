"""MIDI output utilities."""

import bisect
import dataclasses
import logging
import math
from collections import defaultdict

import numpy as np

from muscriptor.tokenizer.notes import (
    Note,
    note2note_event,
    note_event2midi,
    trim_overlapping_notes,
)
from muscriptor.utils.beats import BeatGrid, Onsets, estimate_onset_delay, fit_tempo

logger = logging.getLogger(__name__)

# Written when no grid was detected: 120 BPM and no time signature, leaving the
# meter for notation software to guess.
PLACEHOLDER_GRID = BeatGrid(bpm=120, beats_per_bar=None, first_downbeat=0.0)


@dataclasses.dataclass(frozen=True)
class ExportOptions:
    """How to write a transcription out as MIDI.

    The first fields correct the detected beat grid (None keeps what was
    detected); they move bar lines, never notes. The rest tidy the notes up for
    notation software and DAWs, which turn every few milliseconds of slop into
    a 1/64 rest.
    """

    bpm: float | None = None
    beats_per_bar: int | None = None
    # A bar line, in seconds of the original audio.
    first_downbeat: float | None = None
    # Follow the tracked beats with a tempo map (True) or write one constant
    # tempo (False). None follows them when they don't fit a constant tempo.
    # A given `bpm` always means a constant tempo.
    tempo_map: bool | None = None
    # Beats per detected beat: 2 when the tracker counted half notes as beats,
    # 0.5 when it counted eighths. A power of two.
    beat_scale: float = 1.0
    # Snap note starts and ends onto a subdivision of the beat.
    quantize: bool = False
    # Grid steps per beat to snap to (4 = sixteenths, 3 = eighth triplets...);
    # None uses the one the onsets were measured to sit on.
    subdivision: int | None = None
    # How far to move the notes towards the grid: 1.0 all the way, 0.5 halfway.
    strength: float = 1.0
    # Extend a note to the next onset of its instrument when the silence in
    # between is at most this many beats. 0 leaves the note lengths alone.
    fill_gaps: float = 0.0
    # Drop the empty bars before the first note. The MIDI then no longer starts
    # where the audio does.
    trim_leading_bars: bool = False

    def __post_init__(self):
        if self.beat_scale <= 0 or not math.log2(self.beat_scale).is_integer():
            raise ValueError(
                f"beat_scale must be a power of two, not {self.beat_scale}"
            )


def resolve_grid(
    beat_grid: BeatGrid | None, options: ExportOptions, onsets: Onsets
) -> BeatGrid:
    """The grid to write: `beat_grid` with `options`' corrections applied.

    The onset delay is measured against the detected beats before any
    correction, so it is the same whatever the user does to the grid: it is a
    property of the transcription, not of the bar lines.

    The result's `steady` says how to write it: False means its `beats`
    (smoothed, and rescaled by `beat_scale`) are the grid, for a tempo map;
    True means a constant `bpm` from `first_downbeat`.

    `beat_subdivision` comes back None when there is nothing to quantize to:
    no tempo at all (detected or given), or onsets that sit on no subdivision of
    the beat.

    Not idempotent: pass it the detected grid, not one it returned.
    """
    tempo_known = beat_grid is not None or options.bpm is not None
    grid = beat_grid or PLACEHOLDER_GRID
    if grid.onset_delay is None:
        grid = grid.with_onset_delay(onsets)

    follow = (
        options.bpm is None
        and grid.beats is not None
        and len(grid.beats) >= 2
        and (options.tempo_map if options.tempo_map is not None else not grid.steady)
    )
    scale = options.beat_scale
    beats_per_bar = options.beats_per_bar or scaled_meter(grid.beats_per_bar, scale)
    first_downbeat = (
        options.first_downbeat
        if options.first_downbeat is not None
        else grid.first_downbeat
    )

    if follow:
        beats = rescaled_beats(smoothed_beats(grid.beats), scale, first_downbeat)
        grid = dataclasses.replace(
            grid,
            bpm=fit_tempo(beats)[0],
            beats_per_bar=beats_per_bar,
            # Bar lines go on beats, so on the tracked beat nearest the one asked for.
            first_downbeat=float(beats[nearest(beats, first_downbeat)]),
            beats=beats,
            beat_subdivision=grid.beat_subdivision if scale == 1 else None,
            steady=False,
        )
    else:
        bpm = options.bpm if options.bpm is not None else grid.bpm * scale
        retimed = not math.isclose(bpm, grid.bpm)
        grid = dataclasses.replace(
            grid,
            bpm=bpm,
            beats_per_bar=beats_per_bar,
            first_downbeat=first_downbeat,
            # Both belong to the detected tempo, and say nothing about another one.
            beats=None if retimed else grid.beats,
            beat_subdivision=None if retimed else grid.beat_subdivision,
            steady=True,
        )

    if not tempo_known:
        return dataclasses.replace(grid, beat_subdivision=None)
    if options.subdivision is not None:
        return dataclasses.replace(grid, beat_subdivision=options.subdivision)
    if options.quantize and grid.beat_subdivision is None:
        delay = grid.onset_delay
        return dataclasses.replace(
            grid,
            beat_subdivision=measure_subdivision(
                np.asarray(onsets, dtype=float) - delay, grid
            ),
        )
    return grid


def scaled_meter(beats_per_bar: int | None, scale: float) -> int | None:
    """The detected meter in rescaled beats: 4/4 read twice as fast is 8/4.

    Kept as it was when the bar doesn't hold a whole number of the new beats.
    """
    if beats_per_bar is None:
        return None
    scaled = beats_per_bar * scale
    return int(scaled) if scaled >= 1 and scaled.is_integer() else beats_per_bar


def nearest(beats: np.ndarray, time: float) -> int:
    """Index of the beat closest to `time`."""
    return int(np.abs(beats - time).argmin())


# Beats around each beat that `smoothed_beats` fits a curve through.
SMOOTHING_WINDOW = 9


def smoothed_beats(beats: np.ndarray) -> np.ndarray:
    """Tracked beat times with the tracker's jitter averaged out.

    beat_this reports beats on a 20 ms frame grid, which at 120 BPM alone makes
    consecutive beats look ±4% apart in tempo: a tempo map read straight off
    them changes tempo on every beat for no musical reason. Each beat is moved
    onto a parabola fitted through its neighbours (a Savitzky-Golay filter), which
    divides that noise down while keeping steady accelerations exact, where a
    plain average would pull the beats of a ritardando off their real times.
    """
    beats = np.asarray(beats, dtype=float)
    n = len(beats)
    if n < 4:
        return beats
    width = min(SMOOTHING_WINDOW, n)
    index = np.arange(n)
    smoothed = np.empty(n)
    for i in range(n):
        lo = min(max(0, i - width // 2), n - width)
        window = slice(lo, lo + width)
        coeffs = np.polyfit(index[window] - i, beats[window], 2)
        smoothed[i] = coeffs[-1]
    return smoothed


def rescaled_beats(beats: np.ndarray, scale: float, downbeat: float) -> np.ndarray:
    """`beats` with `scale` times as many beats per second.

    Doubling puts a beat halfway between each pair; halving keeps every other
    one, picking the half that has the beat nearest `downbeat` so the bar lines
    can stay where they were.
    """
    while scale > 1:
        doubled = np.empty(2 * len(beats) - 1)
        doubled[0::2] = beats
        doubled[1::2] = (beats[:-1] + beats[1:]) / 2
        beats, scale = doubled, scale / 2
    while scale < 1 and len(beats) >= 4:
        beats, scale = beats[nearest(beats, downbeat) % 2 :: 2], scale * 2
    return beats


def measure_subdivision(onsets: np.ndarray, grid: BeatGrid) -> int | None:
    """The subdivision of `grid`'s beats that `onsets` sit on, if any.

    A steady grid's beats are laid out at constant tempo from `first_downbeat`
    over the onsets' span, since it may not have tracked ones (a tempo typed in
    by the user) or they may be for another tempo.
    """
    if not len(onsets):
        return None
    if grid.steady:
        period = 60.0 / grid.bpm
        first = math.floor((onsets.min() - grid.first_downbeat) / period)
        last = math.ceil((onsets.max() - grid.first_downbeat) / period)
        grid = dataclasses.replace(
            grid, beats=grid.first_downbeat + period * np.arange(first, last + 1)
        )
    measured = estimate_onset_delay(onsets, grid)
    return measured.subdivision if measured else None


@dataclasses.dataclass(frozen=True)
class Timeline:
    """Where moments of the recording fall in the written score.

    `times` are seconds of the recording, increasing, at consecutive whole
    `positions` in beats from the start of the file (tick 0). Linear in
    between, and continued at the edge intervals' tempo beyond.
    """

    times: np.ndarray
    positions: np.ndarray

    def beats_at(self, seconds):
        return _extrapolated(seconds, self.times, self.positions)

    def seconds_at(self, beats):
        return _extrapolated(beats, self.positions, self.times)

    def shifted(self, beats: float) -> "Timeline":
        """This timeline with every position moved by `beats`."""
        return Timeline(self.times, self.positions + beats)

    def tempo_changes(self, ticks_per_beat: int) -> list[tuple[int, int]]:
        """(tick, microseconds per beat) at every beat where the tempo changes.

        Only from tick 0 on: whatever came before has no place in the file.
        """
        tempi = np.round(np.diff(self.times) * 1e6).astype(int)
        # The first interval's tempo also holds before it, back to tick 0.
        changes = [(0, int(tempi[0]))]
        for position, tempo in zip(self.positions[:-1], tempi):
            tick = max(0, round(position * ticks_per_beat))
            if tempo == changes[-1][1]:
                continue
            if tick == changes[-1][0]:
                changes[-1] = (tick, int(tempo))
            else:
                changes.append((tick, int(tempo)))
        return changes


def _extrapolated(x, xp: np.ndarray, fp: np.ndarray):
    """np.interp, but continuing the end segments' slopes instead of clamping."""
    x = np.asarray(x, dtype=float)
    lo = fp[0] + (x - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
    hi = fp[-1] + (x - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    inner = np.interp(x, xp, fp)
    out = np.where(x < xp[0], lo, np.where(x > xp[-1], hi, inner))
    return float(out) if out.ndim == 0 else out


def timeline_for(grid: BeatGrid, earliest: float) -> Timeline:
    """The timeline `grid` is written with, bar 1 starting at tick 0.

    MIDI has no pickup measure, so the music is delayed by whole bars until
    `earliest` (seconds) falls at or after tick 0, keeping a bar line on the
    downbeat. Bars are single beats when there is no meter.
    """
    if grid.steady:
        period = 60.0 / grid.bpm
        start = -grid.bar_offset(min_shift=-earliest)
        return Timeline(np.array([start, start + period]), np.array([0.0, 1.0]))

    beats = grid.beats
    # Positions with the downbeat at 0, then whole bars added to fit `earliest`.
    timeline = Timeline(
        beats, np.arange(len(beats)) - nearest(beats, grid.first_downbeat)
    )
    bar = grid.beats_per_bar or 1
    return timeline.shifted(bar * math.ceil(-timeline.beats_at(earliest) / bar))


def shifted_notes(notes: list[Note], delay_s: float) -> list[Note]:
    """`notes` moved by `delay_s`. May go negative; the bar offset covers that."""
    if not delay_s:
        return notes
    return [
        dataclasses.replace(
            note, onset=note.onset + delay_s, offset=note.offset + delay_s
        )
        for note in notes
    ]


TICKS_PER_BEAT = 480

# How long drum hits are written, in beats: they have no length of their own.
DRUM_HIT_BEATS = 1 / 32


def notes_to_midi(
    notes: list[Note],
    velocity: int = 100,
    program_names: dict[int, str] | None = None,
    beat_grid: BeatGrid | None = None,
    options: ExportOptions | None = None,
):
    """Convert a list of Note objects to a mido MidiFile.

    `program_names` maps program numbers to human-readable track names
    (see note_event2midi).

    `beat_grid` is a detected beat grid (see muscriptor.utils.beats): it supplies
    the tempo, the time signature and a delay that puts bar lines on real
    downbeats. Defaults to PLACEHOLDER_GRID. A grid whose beats don't fit a
    constant tempo is written as a tempo map following them, so the bar lines
    follow the performance.

    The notes are moved onto the beat grid based on its `onset_delay` to better match
    the grid.

    `options` corrects the grid and says how far to tidy the notes onto it; see
    ExportOptions. Quantizing needs a subdivision, which `resolve_grid` may not
    find; the notes are then written unquantized.
    """
    options = options or ExportOptions()
    grid = resolve_grid(beat_grid, options, [n.onset for n in notes])
    delay = grid.onset_delay
    notes = shifted_notes(notes, -delay)
    earliest = min([-delay] + [n.onset for n in notes])
    timeline = timeline_for(grid, earliest)
    # From here on, note times are in beats from tick 0.
    notes = [
        dataclasses.replace(
            n, onset=timeline.beats_at(n.onset), offset=timeline.beats_at(n.offset)
        )
        for n in notes
    ]

    if options.quantize:
        if grid.beat_subdivision is None:
            logger.warning("not quantizing: no beat subdivision to snap the notes to")
        else:
            notes = quantized_notes(notes, 1 / grid.beat_subdivision, options.strength)

    if options.fill_gaps > 0:
        notes = gaps_filled(notes, options.fill_gaps)

    if options.trim_leading_bars and notes:
        empty = leading_bars(notes, grid)
        timeline = timeline.shifted(-empty)
        notes = shifted_notes(notes, -empty)

    return note_event2midi(
        note2note_event(notes),
        output_file=None,
        velocity=velocity,
        ticks_per_beat=TICKS_PER_BEAT,
        program_names=program_names,
        beats_per_bar=grid.beats_per_bar,
        # Where the recording's 0 s plays: a tempo map is the recording's own
        # timing, so seconds after tick 0 are seconds of the recording.
        offset_s=-timeline.seconds_at(0.0),
        to_tick=lambda beats: round(beats * TICKS_PER_BEAT),
        tempo_changes=timeline.tempo_changes(TICKS_PER_BEAT),
        drum_length=DRUM_HIT_BEATS,
    )


def quantized_notes(
    notes: list[Note], step: float, strength: float = 1.0
) -> list[Note]:
    """`notes` (in beats from the start of the score) moved onto a `step`-beat grid.

    `strength` below 1 only moves them that fraction of the way, which keeps
    some of the feel of the performance.
    """

    def snap(t: float) -> float:
        return round(t / step) * step

    def pull(t: float, target: float) -> float:
        return t + strength * (target - t)

    snapped = []
    for note in notes:
        onset = snap(note.onset)
        # A note shorter than half a step rounds to nothing and would vanish
        # from the score; give it the shortest length the grid has.
        offset = max(snap(note.offset), onset + step)
        snapped.append(
            dataclasses.replace(
                note, onset=pull(note.onset, onset), offset=pull(note.offset, offset)
            )
        )
    # Bumping the short ones can make two notes of the same pitch overlap.
    return trim_overlapping_notes(snapped)


# Leeway for comparing times that went through the same float arithmetic.
_EPS = 1e-6


def gaps_filled(notes: list[Note], max_gap: float) -> list[Note]:
    """`notes` held on until their instrument's next onset, across short silences.

    In whatever unit the note times are; `notes_to_midi` passes beats.

    A transcription ends notes where the sound fades, which notation software
    writes as a rest before the next note: a staccato score where the player
    let the notes ring. Per instrument, a note that stops at most `max_gap`
    before the next onset (of any pitch) is extended to that onset.
    Drums have no length to extend.
    """
    by_instrument: dict[tuple[int, bool], list[Note]] = defaultdict(list)
    for note in notes:
        by_instrument[(note.program, note.is_drum)].append(note)

    filled = []
    for (_, is_drum), group in by_instrument.items():
        if is_drum:
            filled.extend(group)
            continue
        onsets = sorted({n.onset for n in group})
        for note in group:
            i = bisect.bisect_right(onsets, note.onset + _EPS)
            if i < len(onsets) and 0 < onsets[i] - note.offset <= max_gap + _EPS:
                note = dataclasses.replace(note, offset=onsets[i])
            filled.append(note)
    return filled


def leading_bars(notes: list[Note], grid: BeatGrid) -> float:
    """Beats of whole empty bars before the first of `notes` (in beats).

    Counted in beats when there is no meter, since that is all the grid has.
    """
    bar = grid.beats_per_bar or 1
    first = min(n.onset for n in notes)
    return max(0, math.floor((first + _EPS) / bar)) * bar
