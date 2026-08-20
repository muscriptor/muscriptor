"""Beat-grid detection, for writing real tempo and time signatures into MIDI."""

import dataclasses
import logging
import math
from collections.abc import Sequence
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

# Marker text prefix recording how far notes were delayed to align bar lines,
# so `/auralize` can line the synthesis back up with the original audio.
BAR_OFFSET_MARKER = "muscriptor:bar_offset="

### Constants for estimate_onset_delay()

# Candidate grids, binary and triplet divisions of the beat, simplest first.
ONSET_SUBDIVISIONS = (1, 2, 3, 4, 6, 8, 12, 16, 24)

# Discard detections with a concentration lower than this. On our test set, 92% songs
# are above 0.5
MIN_ONSET_CONCENTRATION = 0.5

# |R| for n random angles is about 1/sqrt(n). With MIN_ONSETS = 40 the chance of
# passing the bar with a random distribution is about 1 in 5,000.
MIN_ONSETS = 40

# The largest delay this is meant to correct.
# Trade-off: we want to correct larger offsets if they happen, but a smaller value
# allows us to use smaller grids like 1/16th notes (depends also on the tempo).
# This is because if a grid has 20ms, we can't distinguish between a +15ms and -5ms
# offset since things are cyclical, unless we have a max offset of <15ms.
# With 0.04 we can do 16th notes up to 187 BPM.
MAX_ONSET_DELAY_S = 0.04

# Note onset times in seconds, however the caller happens to hold them.
Onsets = Sequence[float] | np.ndarray


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


@dataclasses.dataclass
class BeatGrid:
    """A constant-tempo beat grid detected from audio."""

    bpm: float
    # None when the meter could not be determined; write no time signature.
    beats_per_bar: int | None
    # Time of the first detected bar line, in seconds.
    first_downbeat: float
    # The individual beat times the grid was fitted to, kept for
    # `with_onset_delay`, which needs them rather than the fitted tempo: the
    # tracked beats follow the recording's small tempo wobbles, and those are the
    # same size as the offset being measured. None for a hand-built grid; kept
    # out of equality and repr, being an array and an implementation detail.
    beats: np.ndarray | None = dataclasses.field(
        default=None, repr=False, compare=False
    )
    # Seconds by which the transcribed onsets are late against these beats, or None
    # until some notes have been measured against the grid (see `with_onset_delay`).
    # Whoever writes the notes out subtracts it from them.
    onset_delay: float | None = None
    # Subdivisions of the beat to use for quantizing the onsets.
    beat_subdivision: int | None = None

    @property
    def bar_seconds(self) -> float | None:
        if self.beats_per_bar is None:
            return None
        return self.beats_per_bar * 60.0 / self.bpm

    def bar_offset(self, min_shift: float = 0.0) -> float:
        """Seconds to delay every note so bar lines land on downbeats.

        MIDI has no pickup measure: bar 1 starts at tick 0, so the only way to
        put a bar line on the first downbeat is to shift the music later. Always
        a forward shift, keeping ticks non-negative and dropping no notes; the
        leading partial bar holds whatever preceded the first downbeat.

        Whole bars are added until the shift reaches `min_shift`, so a caller
        that moved the notes earlier (by `onset_delay`) still gets non-negative
        ticks without having to squash the notes near the start.
        """
        step = self.bar_seconds
        if step is None:
            # No meter to align to, so the only reason to shift is the headroom;
            # do it in whole beats, which are all this grid has.
            step, offset = 60.0 / self.bpm, 0.0
        else:
            offset = (step - self.first_downbeat % step) % step
        if offset < min_shift:
            offset += step * math.ceil((min_shift - offset) / step)
        return offset

    def with_onset_delay(self, onsets: Onsets) -> "BeatGrid":
        """This grid with `onsets`' lag against it measured and filled in.

        Subtracting that lag from the note times moves them onto the beats. We
        trust the beats tracked by beat_this over the transcription - they're a bit
        more accurate based on our measurements. 0.0 when there is nothing to
        measure from; see estimate_onset_delay() for details.

        Measuring once and carrying the answer around is what keeps the MIDI file
        and the UI (which is told the same number) shifting by the same amount.
        """
        measured = None
        if self.beats is not None:
            measured = estimate_onset_delay(onsets, self)
        if measured is None:
            return dataclasses.replace(self, onset_delay=0.0, beat_subdivision=None)
        logger.info(
            "onsets sit %+.1f ms off a 1/%d-beat grid (|R| = %.2f over %d "
            "onsets); moving the notes back by that much",
            1000 * measured.seconds,
            measured.subdivision,
            measured.concentration,
            measured.n_onsets,
        )
        return dataclasses.replace(
            self, onset_delay=measured.seconds, beat_subdivision=measured.subdivision
        )


@dataclasses.dataclass(frozen=True)
class OnsetDelay:
    """How late a set of note onsets sits against the beat subdivision it is on."""

    # Signed seconds, positive when the onsets are late.
    seconds: float
    # Resultant length in [0, 1]: how tightly the onsets sit on the grid.
    concentration: float
    # Subdivisions per beat of the grid the delay is measured against.
    subdivision: int
    # Distinct onset times that went into it.
    n_onsets: int


def get_onsets_phase(onsets: Onsets, beats: np.ndarray) -> np.ndarray:
    """Onset times as positions in continuous beats.

    Onsets outside the tracked span drop out, since there is no beat to place
    them in. One entry per distinct onset time (to the millisecond) rather than
    per note, so a six-note chord does not outvote six single notes elsewhere.
    """
    times = np.unique(np.round(np.asarray(onsets, dtype=float), 3))
    inside = (times >= beats[0]) & (times <= beats[-1])
    return np.interp(times[inside], beats, np.arange(len(beats)))


def get_phase_with_subdivision(
    phase_beats: np.ndarray, subdivision: int
) -> tuple[float, float]:
    """Compute the average of unit vectors (phase_beats) on a circle.

    The angle tells us the mean offset, and the magnitude tells us how well they align
    (the concentration).
    """
    angles = 2 * np.pi * np.mod(phase_beats * subdivision, 1.0)
    mean = np.exp(1j * angles).mean()
    turns = 1 / (2 * np.pi * subdivision)  # radians on the fine grid → beats
    return float(np.abs(mean)), float(np.angle(mean)) * turns


def estimate_onset_delay(onsets: Onsets, grid: "BeatGrid") -> OnsetDelay | None:
    """How late `onsets` sit against the beat subdivision they are on.

    This is to correct for the fact that we observed that the detected onsets don't
    always align well with the beats.

    Algorithm: For each onset, plot it as a unit vector on a circle where the angle is
    its relative position in the beat. Then take the average of these vectors: the
    angle tells you the average offset and the magnitude says how well they align.
    In practice, the onsets are on a subdivision of the beat (e.g. 8th notes) so
    also repeat this for different subdivisions and pick the one where the notes align
    the best to read the alignment from.
    """
    if grid.beats is None or len(grid.beats) < 2:
        return None
    period_s = 60.0 / grid.bpm

    # The phases relative to full beats
    phase_beats = get_onsets_phase(onsets, grid.beats)
    if len(phase_beats) < MIN_ONSETS:
        logger.info(
            "not correcting the downbeat: %d onset time(s) on the tracked span, "
            "need %d",
            len(phase_beats),
            MIN_ONSETS,
        )
        return None

    candidates = [
        s for s in ONSET_SUBDIVISIONS if period_s / (2 * s) >= MAX_ONSET_DELAY_S
    ]
    if not candidates:
        # Should not happen with reasonable tempos but guard it anyway
        logger.info("not correcting the downbeat: no grid coarse enough to fit")
        return None

    scored = {s: get_phase_with_subdivision(phase_beats, s) for s in candidates}

    # Keep the subdivision with the highest concentration
    subdivision = max(scored, key=lambda s: scored[s][0])
    concentration, delay_beats = scored[subdivision]

    if concentration < MIN_ONSET_CONCENTRATION:
        logger.info(
            "not correcting the downbeat: onsets sit on no subdivision of the beat "
            "(best |R| = %.2f, need %.2f)",
            concentration,
            MIN_ONSET_CONCENTRATION,
        )
        return None

    delay_s = delay_beats * period_s
    if abs(delay_s) > MAX_ONSET_DELAY_S:
        logger.warning(
            "not correcting the downbeat: onsets measured %+.0f ms off a "
            "1/%d-beat grid, further than the %+.0f ms this can plausibly be",
            1000 * delay_s,
            subdivision,
            1000 * MAX_ONSET_DELAY_S,
        )
        return None

    return OnsetDelay(
        seconds=delay_s,
        concentration=concentration,
        subdivision=subdivision,
        n_onsets=len(phase_beats),
    )


def fit_tempo(beats: np.ndarray) -> tuple[float, float]:
    """Least-squares tempo over the beat sequence.

    Returns (bpm, residual RMS in seconds). Fitting a line through beat index
    against time beats taking the median inter-beat interval: trackers quantise
    beats to a frame grid (50 Hz for beat_this), which alone limits median-IBI
    tempo resolution to a few BPM.
    """
    index = np.arange(len(beats))
    slope, intercept = np.polyfit(index, beats, 1)
    residual = beats - (intercept + slope * index)
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
    return BeatGrid(
        bpm=bpm,
        beats_per_bar=beats_per_bar,
        first_downbeat=first_downbeat,
        beats=beats,
    )
