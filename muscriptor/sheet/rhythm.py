"""Rhythm: beat-relative time, swing, and per-beat adaptive quantization.

Time is measured in the beats tracked from the audio (beat_this, see
`muscriptor.utils.beats`), not in seconds at an assumed tempo: a note's
position is "beat 37 and a bit" wherever the performance sped up or slowed
down, so bar lines stay on the music through rubato.

Quantization does not snap everything to one global 1/16 grid. Each beat of
each part picks its own subdivision — 1, 2, 4 (binary) or 3, 6 (triplet) —
trading how far the notes would have to move against how hard the result is
to read (finer grids and triplets cost more). The choices are made jointly
for the whole part with a Viterbi pass, which also charges for flipping
between binary and triplet feel, so an isolated sloppy beat does not turn
into a lone triplet. A swung performance (off-beat eighths late, near 2/3 of
the beat) is detected globally and written as straight eighths marked
"Swing", the way a musician would read it.
"""

from __future__ import annotations

import bisect
import dataclasses
import logging
import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from muscriptor.sheet.ir import PerformanceNote, QuantizedNote
from muscriptor.utils.beats import BeatGrid
from muscriptor.utils.midi import ExportOptions, Timeline, resolve_grid, timeline_for

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Meter map


@dataclass
class MeterMap:
    """Where seconds of the recording fall in the score (quarter-note beats).

    Position 0 is the downbeat of measure 1.
    """

    beats_per_bar: int
    timeline: Timeline
    onset_delay: float
    bpm: float
    steady: bool
    detected: bool  # False when no beat grid could be detected

    def beats_at(self, seconds):
        return self.timeline.beats_at(np.asarray(seconds) - self.onset_delay)

    def seconds_at(self, beats):
        return self.timeline.seconds_at(beats) + self.onset_delay

    def measure_tempo(self, measure_index: int) -> float:
        """Average BPM over a measure (0-based)."""
        a = measure_index * self.beats_per_bar
        b = a + self.beats_per_bar
        dt = float(self.seconds_at(b) - self.seconds_at(a))
        return 60.0 * self.beats_per_bar / dt if dt > 0 else self.bpm

    def shifted(self, beats: float) -> "MeterMap":
        return dataclasses.replace(self, timeline=self.timeline.shifted(beats))


def build_meter_map(
    grid: BeatGrid | None,
    onsets: list[float],
    beats_per_bar: int | None = None,
    beat_scale: float = 1.0,
    first_downbeat: float | None = None,
    bpm: float | None = None,
) -> MeterMap:
    """A MeterMap from a detected beat grid (or the placeholder 120 BPM).

    `beats_per_bar` overrides the detected meter; with none detected either,
    4/4 is assumed (and logged), since a score needs bar lines.
    `beat_scale` doubles (2) or halves (0.5) the tracked tempo.
    """
    detected_meter = grid.beats_per_bar if grid is not None else None
    meter = beats_per_bar or (
        int(detected_meter * beat_scale)
        if detected_meter and (detected_meter * beat_scale).is_integer()
        else None
    )
    if meter is None:
        meter = 4
        logger.warning("meter: none detected, assuming 4/4")
    options = ExportOptions(
        bpm=bpm,
        beats_per_bar=meter,
        first_downbeat=first_downbeat,
        beat_scale=beat_scale,
    )
    resolved = resolve_grid(grid, options, onsets)
    earliest = min([0.0] + [o - (resolved.onset_delay or 0.0) for o in onsets])
    timeline = timeline_for(resolved, earliest)
    logger.info(
        "meter: %d/4, %.1f BPM%s, onset delay %+.0f ms",
        meter,
        resolved.bpm,
        "" if resolved.steady else " (tempo map follows the performance)",
        1000 * (resolved.onset_delay or 0.0),
    )
    return MeterMap(
        beats_per_bar=meter,
        timeline=timeline,
        onset_delay=resolved.onset_delay or 0.0,
        bpm=resolved.bpm,
        steady=resolved.steady,
        detected=grid is not None,
    )


# ---------------------------------------------------------------------------
# Swing


@dataclass(frozen=True)
class SwingAnalysis:
    swing: bool
    ratio: float  # mean position of off-beat eighths within the beat
    n_offbeats: int


def detect_swing(positions: np.ndarray, min_offbeats: int = 40) -> SwingAnalysis:
    """Whether off-beat eighths sit late (near 2/3 of the beat) rather than at 1/2.

    `positions` are onset times in beats. Only onsets in the middle of the
    beat count: those near 1/4 or 3/4 (straight sixteenths) vote straight,
    those near 1/2 straight, those near 2/3 swung.
    """
    frac = np.mod(positions, 1.0)
    mid = frac[(frac > 0.38) & (frac < 0.78)]
    if len(mid) < min_offbeats:
        return SwingAnalysis(False, float(np.mean(mid)) if len(mid) else 0.5, len(mid))
    straight = np.sum(np.abs(mid - 0.5) < 0.07)
    swung = np.sum(np.abs(mid - 2 / 3) < 0.07)
    sixteenths = np.sum(np.abs(frac - 0.25) < 0.06) + np.sum(np.abs(frac - 0.75) < 0.06)
    swing = (
        swung > 1.5 * straight
        and swung > 0.5 * sixteenths
        and swung >= min_offbeats // 2
    )
    return SwingAnalysis(bool(swing), float(np.median(mid)), int(len(mid)))


# ---------------------------------------------------------------------------
# Per-beat grid selection


@dataclass(frozen=True)
class GridChoice:
    """One candidate way to divide a beat."""

    name: str
    # Snapping targets within the beat (as fractions of the beat), 0..1.
    targets: tuple[Fraction, ...]
    # Where each target is written (differs from `targets` only for swing).
    written: tuple[Fraction, ...]
    penalty: float
    family: str  # "neutral", "binary", "triplet"
    subdivision: int


def _even(n: int, penalty: float, family: str) -> GridChoice:
    pts = tuple(Fraction(i, n) for i in range(n + 1))
    return GridChoice(f"1/{n}", pts, pts, penalty, family, n)


def grid_choices(
    swing: bool, allow_triplets: bool = True, finest: int = 4
) -> list[GridChoice]:
    """Candidate beat divisions with their readability penalty.

    Penalties are in units of squared timing error / sigma², so a penalty of
    2 means "only pick this grid if it saves two notes being off by a full
    sigma".
    """
    choices = [_even(1, 0.0, "neutral"), _even(2, 0.6 if not swing else 2.5, "binary")]
    if finest >= 4:
        choices.append(_even(4, 2.0, "binary"))
    if allow_triplets:
        choices.append(_even(3, 3.0 if not swing else 1.5, "triplet"))
        if finest >= 4:
            choices.append(_even(6, 5.0, "triplet"))
    if swing:
        half, two_thirds = Fraction(1, 2), Fraction(2, 3)
        choices.append(
            GridChoice(
                "swing8",
                (Fraction(0), two_thirds, Fraction(1)),
                (Fraction(0), half, Fraction(1)),
                0.6,
                "binary",
                2,
            )
        )
    return choices


@dataclass(frozen=True)
class QuantizeOptions:
    # Timing tolerance, in beats, that a snap error is measured against.
    sigma: float = 0.07
    # Extra cost for switching between binary and triplet on adjacent beats.
    family_switch_penalty: float = 2.5
    allow_triplets: bool = True
    # Finest binary division (4 = sixteenths, 2 = eighths).
    finest: int = 4
    # An onset this close (in beats) before a beat belongs to that beat.
    lookahead: float = 0.12


def _snap(frac: float, choice: GridChoice) -> tuple[float, Fraction]:
    """(error in beats, written position) of snapping `frac` with `choice`."""
    best_err, best_written = math.inf, Fraction(0)
    for t, w in zip(choice.targets, choice.written):
        err = abs(frac - float(t))
        if err < best_err:
            best_err, best_written = err, w
    return best_err, best_written


def choose_grids(
    positions: np.ndarray,
    n_beats: int,
    choices: list[GridChoice],
    options: QuantizeOptions,
) -> list[GridChoice]:
    """The best GridChoice for every beat 0..n_beats-1 (Viterbi)."""
    beat_index = np.floor(positions + options.lookahead).astype(int)
    frac = positions - beat_index
    per_beat: list[list[float]] = [[] for _ in range(n_beats)]
    for b, f in zip(beat_index, frac):
        if 0 <= b < n_beats:
            per_beat[b].append(float(f))

    k = len(choices)
    # Distinct onset times per beat (a chord counts once).
    local = np.zeros((n_beats, k))
    for b, fr in enumerate(per_beat):
        if not fr:
            continue
        distinct = sorted(set(round(x, 3) for x in fr))
        for j, c in enumerate(choices):
            err = sum((_snap(x, c)[0] / options.sigma) ** 2 for x in distinct)
            # Two distinct onsets collapsing onto one grid point loses a note.
            snapped = [_snap(x, c)[1] for x in distinct]
            collisions = len(snapped) - len(set(snapped))
            local[b, j] = err + c.penalty + 4.0 * collisions

    def switch(a: GridChoice, b: GridChoice) -> float:
        if "neutral" in (a.family, b.family) or a.family == b.family:
            return 0.0
        return options.family_switch_penalty

    trans = np.array([[switch(a, b) for b in choices] for a in choices])
    cost = local[0].copy()
    back = np.zeros((n_beats, k), dtype=int)
    for b in range(1, n_beats):
        total = cost[:, None] + trans
        back[b] = total.argmin(axis=0)
        cost = total.min(axis=0) + local[b]
    path = [int(cost.argmin())]
    for b in range(n_beats - 1, 0, -1):
        path.append(int(back[b, path[-1]]))
    path.reverse()
    return [choices[j] for j in path]


def quantize_position(
    pos: float, grids: list[GridChoice], lookahead: float
) -> tuple[Fraction, GridChoice]:
    """Snap one position (beats) with the grid chosen for its beat."""
    b = int(math.floor(pos + lookahead))
    b_clamped = min(max(b, 0), len(grids) - 1)
    choice = grids[b_clamped]
    _, written = _snap(pos - b, choice)
    return Fraction(b) + written, choice


def quantize_part(
    notes: list[PerformanceNote],
    meter: MeterMap,
    n_beats: int,
    swing: bool,
    options: QuantizeOptions = QuantizeOptions(),
) -> list[QuantizedNote]:
    """Quantize one part's notes onto per-beat adaptive grids."""
    if not notes:
        return []
    onsets = np.asarray(meter.beats_at([n.onset for n in notes]), dtype=float)
    offsets = np.asarray(meter.beats_at([n.offset for n in notes]), dtype=float)
    choices = grid_choices(swing, options.allow_triplets, options.finest)
    grids = choose_grids(onsets, n_beats, choices, options)

    out = []
    for n, on, off in zip(notes, onsets, offsets):
        q_on, choice = quantize_position(float(on), grids, options.lookahead)
        if q_on < 0:
            continue
        step = Fraction(1, choice.subdivision)
        q_off, _ = quantize_position(float(off), grids, options.lookahead)
        if q_off <= q_on:
            q_off = q_on + step
        out.append(
            QuantizedNote(
                pitch=n.pitch,
                onset=q_on,
                offset=q_off,
                instrument=n.instrument,
                velocity=n.velocity,
                grid=choice.subdivision,
            )
        )
    return dedupe_quantized(out)


def dedupe_quantized(notes: list[QuantizedNote]) -> list[QuantizedNote]:
    """Merge notes that snapped onto the same (pitch, onset); trim same-pitch overlaps."""
    best: dict[tuple[int, Fraction], QuantizedNote] = {}
    for n in notes:
        key = (n.pitch, n.onset)
        if key not in best or n.offset > best[key].offset:
            best[key] = n
    out = sorted(best.values(), key=lambda n: (n.pitch, n.onset))
    for a, b in zip(out, out[1:]):
        if a.pitch == b.pitch and a.offset > b.onset:
            a.offset = b.onset
    return sorted(
        (n for n in out if n.offset > n.onset), key=lambda n: (n.onset, n.pitch)
    )


def fill_gaps(notes: list[QuantizedNote], max_gap: Fraction) -> list[QuantizedNote]:
    """Hold notes on to the next onset across short silences (sustain).

    The transcription ends a note where its sound fades; written out, that is
    a rest the player would never read ("let ring" / pedal). Notes are held
    on when the silence before the next onset of the same staff is at most
    `max_gap` beats — and never past that next onset.
    """
    if max_gap <= 0 or not notes:
        return notes
    onsets = sorted({n.onset for n in notes})
    out = []
    for n in notes:
        i = bisect.bisect_right(onsets, n.onset)
        if i < len(onsets):
            nxt = onsets[i]
            if n.offset < nxt and nxt - n.offset <= max_gap:
                n = dataclasses.replace(n, offset=nxt)
        out.append(n)
    return out


# ---------------------------------------------------------------------------
# Downbeat inference


@dataclass(frozen=True)
class MeterGuess:
    beats_per_bar: int
    phase: int  # index of the first downbeat among the tracked beats
    score: float  # share of the evidence weight that lands on bar lines
    runner_up: float


def infer_meter_phase(
    beats: np.ndarray,
    events: list[tuple[float, float]],
    meters: tuple[int, ...] = (4, 3),
    prefer: int = 4,
) -> MeterGuess | None:
    """Meter and bar phase from where musical events fall on the tracked beats.

    Used when the beat tracker's downbeats do not agree on a meter. `events`
    are (time, weight) pairs: chord changes, bass-note attacks and kick-drum
    hits all gravitate to the downbeat. Each is snapped to its nearest beat
    (if within a fifth of a beat), and the (meter, phase) putting the most
    weight on bar lines wins; `prefer` (4/4) is kept unless another meter is
    clearly better.
    """
    beats = np.asarray(beats, dtype=float)
    if len(beats) < 8 or not events:
        return None
    period = float(np.median(np.diff(beats)))
    idx_w: list[tuple[int, float]] = []
    for t, w in events:
        i = int(np.argmin(np.abs(beats - t)))
        if abs(beats[i] - t) <= 0.2 * period:
            idx_w.append((i, w))
    total = sum(w for _, w in idx_w)
    if total <= 0:
        return None
    results = []
    for m in meters:
        for p in range(m):
            on_bar = sum(w for i, w in idx_w if (i - p) % m == 0)
            # Normalise by the share a random phase would get (1/m), so
            # meters with fewer beats per bar are not favoured for free.
            results.append(((on_bar / total) * m, m, p))
    results.sort(reverse=True)
    best = results[0]
    preferred = max((r for r in results if r[1] == prefer), default=best)
    if best[1] != prefer and best[0] < 1.25 * preferred[0]:
        best = preferred
    runner = max(
        (r[0] for r in results if (r[1], r[2]) != (best[1], best[2])), default=0.0
    )
    return MeterGuess(best[1], best[2], best[0], runner)
