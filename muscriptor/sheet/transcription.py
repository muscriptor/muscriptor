"""Transcription stage: MuScriptor on the mix and on selected stems.

The full mix is transcribed unconstrained — MuScriptor is trained on mixes
and its instrument labels are the instrument detection of this pipeline.
Stems listed in the config are then transcribed again with MuScriptor's
instrument constraint (`instruments=`), so the vocals stem can only yield a
voice part, the bass stem only bass, and the "other" stem only strings,
winds, brass and synths (never piano or guitar). Their notes replace the
mix's notes for that instrument family: an isolated stem gives a cleaner
line than the same instrument picked out of the full mix, and the label is
no longer the mix model's guess. See :func:`transcribe_stem` for why stems
are decoded twice.
"""

from __future__ import annotations

import dataclasses
import logging
from collections import Counter
from pathlib import Path

from muscriptor.events import NoteEndEvent, NoteStartEvent
from muscriptor.sheet.instruments import STEM_GROUPS
from muscriptor.sheet.ir import PerformanceNote

logger = logging.getLogger(__name__)


def transcribe_audio(
    model,
    audio: Path,
    source: str = "mix",
    instruments: list[str] | None = None,
    beam_size: int = 1,
    condition: bool = True,
) -> list[PerformanceNote]:
    """Run MuScriptor on one file and pair its start/end events into notes.

    `instruments` is a hard constraint; with `condition=False` it is not also
    given to the model as conditioning (see :func:`transcribe_stem`).
    """
    open_notes: dict[int, NoteStartEvent] = {}
    notes: list[PerformanceNote] = []
    kwargs = {} if condition else {"condition_on_instruments": False}
    events = model.transcribe(
        audio, instruments=instruments, beam_size=beam_size, **kwargs
    )
    for ev in events:
        if isinstance(ev, NoteStartEvent):
            open_notes[ev.index] = ev
        elif isinstance(ev, NoteEndEvent):
            start = open_notes.pop(ev.start_event_index, None)
            if start is None:
                continue
            end = max(ev.end_time, start.start_time)
            notes.append(
                PerformanceNote(
                    pitch=start.pitch,
                    onset=float(start.start_time),
                    offset=float(end),
                    instrument=start.instrument,
                    source=source,
                )
            )
    notes.sort(key=lambda n: (n.onset, n.pitch))
    logger.info(
        "transcription[%s]: %d notes: %s",
        source,
        len(notes),
        ", ".join(
            f"{k} {v}" for k, v in Counter(n.instrument for n in notes).most_common()
        ),
    )
    return notes


def detect_instruments(
    notes: list[PerformanceNote],
    min_share: float = 0.1,
    min_notes: int = 20,
    max_groups: int = 4,
) -> list[str]:
    """The groups that make up a real share of a transcription, most notes first."""
    counts = Counter(n.instrument for n in notes)
    total = sum(counts.values())
    return [
        g
        for g, k in counts.most_common()
        if k >= min_notes and k / max(total, 1) >= min_share
    ][:max_groups]


def transcribe_stem(
    model,
    audio: Path,
    source: str,
    family: list[str],
    beam_size: int = 1,
) -> tuple[list[PerformanceNote], list[str]]:
    """Transcribe a separated stem in two passes; (notes, detected groups).

    MuScriptor was trained with the conditioning naming the instruments that
    actually play. Conditioning on a whole family ("any of these 20") is out
    of distribution and degrades badly (on a real strings stem it produced
    no notes at all), while conditioning on a 3-group family the stem only
    partly covers splits one piano into acoustic + electric halves.

    1. *Recognise*: decode with only the hard constraint (piano, guitar, …
       forbidden on the "other" stem) and no conditioning; the model picks
       among the allowed instruments by itself.
    2. *Transcribe*: decode again constrained to and conditioned on the
       groups that make up a real share of pass 1.

    A single-group family needs one pass.
    """
    if len(family) <= 1:
        return transcribe_audio(model, audio, source, family, beam_size), family
    first = transcribe_audio(
        model, audio, f"{source}:detect", family, beam_size, condition=False
    )
    detected = detect_instruments(first)
    if not detected:
        return [dataclasses.replace(n, source=source) for n in first], detected
    logger.info("transcription[%s]: detected %s", source, ", ".join(detected))
    notes = transcribe_audio(model, audio, source, detected, beam_size)
    return notes, detected


def gate_notes(
    notes: list[PerformanceNote], activity, gate_db: float, source: str = ""
) -> list[PerformanceNote]:
    """Drop notes sounding where the stem is near-silent relative to the mix.

    A separated stem is never perfectly clean: a loud piano chord leaves a
    faint trace in the strings stem, and the constrained transcription of
    that trace is a phantom string note. `activity` is the stem's
    :class:`~muscriptor.sheet.separation.Activity` curve.
    """
    if activity is None or not notes:
        return notes
    kept = [n for n in notes if activity.level(n.onset, n.offset) >= gate_db]
    if len(kept) < len(notes):
        logger.info(
            "transcription[%s]: gated %d of %d notes in inactive stretches (< %.0f dB)",
            source,
            len(notes) - len(kept),
            len(notes),
            gate_db,
        )
    return kept


def merge_stem_notes(
    mix_notes: list[PerformanceNote],
    stem_notes: dict[str, list[PerformanceNote]],
    min_stem_notes: int,
    absent: set[str] | frozenset[str] = frozenset(),
) -> list[PerformanceNote]:
    """The mix transcription with each family replaced by its stem's notes.

    Families of `absent` stems (separated, but silent) are removed from the
    mix transcription without replacement.
    """
    replace: set[str] = set()
    for stem in absent:
        groups = set(STEM_GROUPS.get(stem, ()))
        dropped = sum(1 for n in mix_notes if n.instrument in groups)
        if dropped:
            logger.info(
                "transcription: dropped %d mix notes labelled %s (the %s stem is silent)",
                dropped,
                "/".join(sorted(groups)),
                stem,
            )
        replace |= groups
    added: list[PerformanceNote] = []
    for stem, notes in stem_notes.items():
        groups = set(STEM_GROUPS.get(stem, ()))
        if len(notes) < min_stem_notes:
            logger.info(
                "transcription: %s stem gave %d notes, keeping the mix's",
                stem,
                len(notes),
            )
            continue
        replace |= groups
        added.extend(n for n in notes if n.instrument in groups)
    kept = [n for n in mix_notes if n.instrument not in replace]
    out = kept + added
    out.sort(key=lambda n: (n.onset, n.pitch))
    return out
