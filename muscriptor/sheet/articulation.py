"""Re-attacked or held? Gluing split sustained notes, checked against the audio.

MuScriptor often ends a sustained string note and starts the same pitch again
at the same instant: on a pop-ballad strings stem, 75% of same-pitch
successors start exactly where the previous note ends. Some of those are
real re-bowings (a rhythmic ostinato), some are one long note the model cut
up. The notes alone cannot tell them apart, so :func:`pitch_rise` looks at
the stem: a new bow stroke makes the energy *at that pitch* (fundamental +
octave, constant-Q) rise at the boundary; a held note does not.

Measured on that song (log-CQT rise across the boundary): new-pitch attacks
median 1.18, the middle of held notes 0.49, same-pitch boundaries 0.66 —
a mix of both. Below 0.3, 88% of real attacks are excluded, so that is the
default threshold.

Modes (``cleanup.strings_articulation``):

``audio``  merge a same-pitch boundary when the stem shows no new attack
           there, or when either side is too short to be a bow stroke
``legato`` merge every back-to-back same-pitch pair
``off``    keep the transcription's articulation
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from muscriptor.sheet.ir import PerformanceNote

logger = logging.getLogger(__name__)

_SR = 22050
_HOP = 256
_LOWEST_MIDI = 24  # C1


def pitch_rise(audio: Path, queries: list[tuple[int, float]]) -> np.ndarray:
    """For each (pitch, time): how much the energy at that pitch rises at `time`.

    log-CQT summed over the fundamental and the octave above; max over the
    70 ms after `time` minus min over the ~180 ms before it. Raises
    ImportError without librosa (the 'sheet' extra).
    """
    import librosa

    y, sr = librosa.load(str(audio), sr=_SR, mono=True)
    cqt = np.abs(
        librosa.cqt(
            y,
            sr=sr,
            hop_length=_HOP,
            fmin=librosa.midi_to_hz(_LOWEST_MIDI),
            n_bins=96,
            bins_per_octave=12,
        )
    )
    log = np.log1p(100 * cqt / (cqt.max() + 1e-12))
    fps = sr / _HOP
    out = np.zeros(len(queries))
    for k, (pitch, t) in enumerate(queries):
        rows = [
            r
            for r in (pitch - _LOWEST_MIDI, pitch + 12 - _LOWEST_MIDI)
            if 0 <= r < log.shape[0]
        ]
        if not rows:
            continue
        e = log[rows].sum(0)
        i = int(t * fps)
        before = e[max(0, i - 8) : max(1, i - 1)]
        after = e[i : i + 6]
        if len(before) and len(after):
            out[k] = float(after.max() - before.min())
    return out


def _chains(notes: list[PerformanceNote], max_gap: float) -> list[tuple[int, int]]:
    """(previous, next) index pairs of back-to-back same-pitch notes."""
    by_pitch: dict[int, list[int]] = defaultdict(list)
    for i, n in enumerate(notes):
        by_pitch[n.pitch].append(i)
    pairs = []
    for idx in by_pitch.values():
        idx.sort(key=lambda i: notes[i].onset)
        for a, b in itertools.pairwise(idx):
            if -max_gap <= notes[b].onset - notes[a].offset <= max_gap:
                pairs.append((a, b))
    return pairs


def merge_repeated_notes(
    notes: list[PerformanceNote],
    mode: str = "audio",
    audio: Path | None = None,
    threshold: float = 0.3,
    max_gap: float = 0.03,
    min_stroke: float = 0.12,
) -> list[PerformanceNote]:
    """`notes` (one part) with split sustained notes glued back together."""
    if mode == "off" or len(notes) < 2:
        return notes
    notes = sorted(notes, key=lambda n: (n.onset, n.pitch))
    pairs = _chains(notes, max_gap)
    if not pairs:
        return notes
    merge = np.ones(len(pairs), dtype=bool)
    if mode == "audio":
        short = np.array(
            [min(notes[a].duration, notes[b].duration) < min_stroke for a, b in pairs]
        )
        rise = None
        if audio is not None:
            try:
                rise = pitch_rise(
                    audio, [(notes[b].pitch, notes[b].onset) for _, b in pairs]
                )
            except ImportError:
                logger.warning(
                    "articulation: librosa not installed ('sheet' extra): "
                    "only fragments shorter than %.2fs are merged",
                    min_stroke,
                )
        merge = short | (rise < threshold) if rise is not None else short
    elif mode != "legato":
        raise ValueError(
            f"strings articulation must be audio, legato or off, not {mode!r}"
        )

    # Union the merged pairs into chains; each chain becomes its first note
    # lasting to the end of its last.
    nxt = {a: b for (a, b), m in zip(pairs, merge) if m}
    has_prev = set(nxt.values())
    out = []
    for i, n in enumerate(notes):
        if i in has_prev:
            continue
        end = n.offset
        j = i
        while j in nxt:
            j = nxt[j]
            end = max(end, notes[j].offset)
        out.append(dataclasses.replace(n, offset=end))
    logger.info(
        "articulation: %d of %d same-pitch repeats merged (%s), %d → %d notes",
        int(merge.sum()),
        len(pairs),
        mode,
        len(notes),
        len(out),
    )
    return out


__all__ = ["merge_repeated_notes", "pitch_rise"]
