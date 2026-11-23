"""Transcription stage: MuScriptor on the mix and on selected stems.

The full mix is transcribed unconstrained — MuScriptor is trained on mixes
and its instrument labels are the instrument detection of this pipeline.
Stems listed in the config are then transcribed again with MuScriptor's
instrument constraint (`instruments=`), so the vocals stem can only yield a
voice part and the bass stem only bass. Their notes replace the mix's notes
for that instrument family: an isolated stem gives a cleaner line than the
same instrument picked out of the full mix.
"""

from __future__ import annotations

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
) -> list[PerformanceNote]:
    """Run MuScriptor on one file and pair its start/end events into notes."""
    open_notes: dict[int, NoteStartEvent] = {}
    notes: list[PerformanceNote] = []
    for ev in model.transcribe(audio, instruments=instruments, beam_size=beam_size):
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


def merge_stem_notes(
    mix_notes: list[PerformanceNote],
    stem_notes: dict[str, list[PerformanceNote]],
    min_stem_notes: int,
) -> list[PerformanceNote]:
    """The mix transcription with each family replaced by its stem's notes."""
    replace: set[str] = set()
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
