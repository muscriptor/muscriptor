"""Beat-grid detection, for writing real tempo and time signatures into MIDI."""

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Used to detect songs that don't have a constant tempo (don't use a metronome)
MAX_TEMPO_RESIDUAL = 0.05

# Fraction of bars that must agree on a beats-per-bar count to write a time
# signature. Trackers that lose the meter spread their downbeats across several
# spacings, and a wrong time signature is worse than none.
MIN_METER_AGREEMENT = 0.9

MIN_BEATS = 8

# An interval this many times the beat length around it is read as spanning
# several beats rather than as the beat length itself.
GAP_INTERVAL_RATIO = 1.5

# Half-width, in intervals, of the window the local beat length is measured
# over. Wide enough that dropouts landing next to each other don't move it, and
# it sets the shortest half-speed passage that stays visible as a tempo change
# rather than being read as gaps: around this many beats, so roughly three bars
# of 4/4. Shorter dips are absorbed into the fit, which is the better reading of
# a brief hesitation anyway.
GAP_BASELINE_WINDOW = 10

# Share of intervals that must span a single beat before the counted positions
# are used. Once most of the take is gaps, the local beat length is itself
# measured over gap-ridden intervals and the count stops meaning anything, so
# array position is the safer guess.
MIN_SINGLE_BEAT_SHARE = 0.6

# Marker text prefix recording how far notes were delayed to align bar lines,
# so `/auralize` can line the synthesis back up with the original audio.
BAR_OFFSET_MARKER = "muscriptor:bar_offset="


class BeatDetectionError(RuntimeError):
    """No usable beat grid in the audio (too short, or no constant tempo)."""


# What to do when the tempo can't be detected: True raises, False doesn't even
# try (the escape hatch for songs the detector gets wrong), and "best-effort"
# warns and falls back to the placeholder tempo.
TempoDetection = bool | Literal["best-effort"]


def read_bar_offset(midi) -> float:
    """Seconds of bar-alignment delay recorded in a MidiFile, 0.0 if absent."""
    for track in midi.tracks:
        for msg in track:
            if msg.type == "marker" and msg.text.startswith(BAR_OFFSET_MARKER):
                try:
                    return float(msg.text.removeprefix(BAR_OFFSET_MARKER))
                except ValueError:
                    return 0.0
    return 0.0


@dataclass
class BeatGrid:
    """A constant-tempo beat grid detected from audio."""

    bpm: float
    # None when the meter could not be determined; write no time signature.
    beats_per_bar: int | None
    # Time of the first detected bar line, in seconds.
    first_downbeat: float

    @property
    def bar_seconds(self) -> float | None:
        if self.beats_per_bar is None:
            return None
        return self.beats_per_bar * 60.0 / self.bpm

    def bar_offset(self) -> float:
        """Seconds to delay every note so bar lines land on downbeats.

        MIDI has no pickup measure: bar 1 starts at tick 0, so the only way to
        put a bar line on the first downbeat is to shift the music later. Always
        a forward shift, keeping ticks non-negative and dropping no notes; the
        leading partial bar holds whatever preceded the first downbeat.
        """
        bar = self.bar_seconds
        if bar is None:
            return 0.0
        return (bar - self.first_downbeat % bar) % bar


def beat_positions(beats: np.ndarray) -> np.ndarray | None:
    """Which beat of the piece each detected beat is, or None if not countable.

    A tracker that misses a beat leaves a gap, and the beats after it are still
    the same beats of the music — so the count has to skip a place there, or
    every later beat is credited to an earlier position than it holds.

    A gap is judged against the beat length *around* it rather than the whole
    take's: a passage that genuinely slows down carries its neighbours with it
    and is not a gap, while a dropout stands out from beats on either side.
    Measured against the whole take the two are indistinguishable — half-speed
    beats and every-other-beat-missing are the same timings — so a take that
    changes tempo by a factor of two would be counted as gaps, fit one line
    perfectly and pass the constant-tempo gate at a tempo it never holds.
    """
    intervals = np.diff(beats)
    if len(intervals) < 3:
        return None
    steps = np.ones(len(intervals), dtype=int)
    for i, interval in enumerate(intervals):
        window = intervals[
            max(0, i - GAP_BASELINE_WINDOW) : i + GAP_BASELINE_WINDOW + 1
        ]
        local = float(np.median(window))
        if local > 0 and interval >= local * GAP_INTERVAL_RATIO:
            steps[i] = max(1, int(round(interval / local)))
    if float((steps == 1).mean()) < MIN_SINGLE_BEAT_SHARE:
        return None
    return np.concatenate(([0], np.cumsum(steps)))


def fit_tempo(beats: np.ndarray) -> tuple[float, float]:
    """Least-squares tempo over the beat sequence.

    Returns (bpm, residual RMS in seconds). Fitting a line through beat time
    against musical position beats taking the median inter-beat interval:
    trackers quantise beats to a frame grid (50 Hz for beat_this), which alone
    limits median-IBI tempo resolution to a few BPM.

    Position is the beat's place in the music, from `beat_positions`, so that a
    missed beat costs a place in the count instead of tilting the line; it falls
    back to position in the array when the beats cannot be counted.
    """
    positions = beat_positions(beats)
    if positions is None:
        positions = np.arange(len(beats))
    slope, intercept = np.polyfit(positions, beats, 1)
    residual = beats - (intercept + slope * positions)
    return 60.0 / float(slope), float(residual.std())


def infer_beats_per_bar(
    beats: np.ndarray,
    downbeats: np.ndarray,
    min_agreement: float = MIN_METER_AGREEMENT,
) -> int | None:
    """Beats per bar from downbeat spacing, or None if the bars disagree.

    Only measures how far apart the downbeats are; it cannot tell whether the
    downbeats themselves are on the right beat. Note that a tracker that
    subdivides the bar wrongly (reporting two beats per bar for music in 3/4)
    can still be self-consistent here, which is why this stays conservative.
    """
    if len(downbeats) < 3 or len(beats) < 2:
        return None
    beat = float(np.median(np.diff(beats)))
    counts = np.round(np.diff(downbeats) / beat).astype(int)
    counts = counts[counts >= 2]
    if not len(counts):
        return None
    values, tally = np.unique(counts, return_counts=True)
    best = int(tally.argmax())
    if tally[best] / len(counts) < min_agreement:
        return None
    return int(values[best])


def detect_grid(
    wav: torch.Tensor, sr: int, checkpoint: str = "final0", device: str = "cpu"
) -> BeatGrid:
    """Detect a constant-tempo beat grid.

    Args:
        wav: Audio as [C, T] (this repo's convention), float32.
        sr: Sample rate of `wav`; beat_this resamples internally.
        checkpoint: beat_this checkpoint name. "final0" over "small0": the small
            model emits spurious beats before the first downbeat, which shifts
            the bar offset by a beat or two.
        device: Torch device for the beat model.

    Raises BeatDetectionError when the audio is too short or the beats do not
    fit a constant tempo. An unclear meter is not fatal: the BeatGrid comes back
    with beats_per_bar=None, since tempo alone is worth writing.
    """
    # Imported here, not at module scope: beat_this pulls in torchaudio and soxr,
    # which would slow every CLI invocation that never transcribes anything.
    from beat_this.inference import Audio2Beats

    # This triggers an error in beat_this so report as BeatDetectionError directly
    min_duration_s = 1.0
    if wav.shape[-1] < min_duration_s * sr:
        raise BeatDetectionError(
            f"Audio is {wav.shape[-1] / sr:.2f}s long, too short to detect a tempo"
        )

    signal = wav.mean(dim=0).detach().cpu().numpy()  # beat_this wants mono, 1-D
    # Returns (beats, downbeats) despite beat_this's own File2File unpacking
    # them the other way round.
    beats, downbeats = Audio2Beats(
        checkpoint_path=checkpoint, device=device, dbn=False
    )(signal, sr)

    beats = np.asarray(beats, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)
    if len(beats) < MIN_BEATS:
        raise BeatDetectionError(
            f"Only {len(beats)} beats detected, need at least {MIN_BEATS}"
        )

    bpm, residual = fit_tempo(beats)
    beat_seconds = 60.0 / bpm
    if residual > MAX_TEMPO_RESIDUAL * beat_seconds:
        raise BeatDetectionError(
            f"The recording has no fixed tempo (beats deviate {residual * 1000:.0f} ms "
            f"RMS from a constant {bpm:.1f} BPM)"
        )

    beats_per_bar = infer_beats_per_bar(beats, downbeats)
    first_downbeat = float(downbeats[0]) if len(downbeats) else float(beats[0])
    logger.info(
        "detected %.3f BPM, %s, first downbeat %.3fs (beat residual %.1f ms)",
        bpm,
        f"{beats_per_bar}/4" if beats_per_bar else "meter unknown",
        first_downbeat,
        residual * 1000,
    )
    return BeatGrid(bpm=bpm, beats_per_bar=beats_per_bar, first_downbeat=first_downbeat)
