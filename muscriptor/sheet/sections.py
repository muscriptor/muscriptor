"""Split a string section into violin, viola and cello parts, cleanly.

MuScriptor mostly labels a bowed ensemble ``string_ensemble``: one part for
the whole section, written on a grand staff like a piano. Arrangers write it
as separate parts. This stage takes the section's notes (after cleanup,
still in performance time) and

1. glues sustained notes the model split back together, where the stem
   shows no new bow stroke (:mod:`muscriptor.sheet.articulation`);
2. gives each note to a section — cello, viola, violin (or just cello and
   violin);
3. tidies each part so it notates cleanly (see :func:`tidy_part`).

Assignment: each group of notes starting together is sorted by pitch and cut
into contiguous runs, low to high, one per section, at the cut points with
the lowest cost:

* range       outside what the instrument can play costs a lot (violin
              below G3, viola below C3, cello below C2), outside where it
              plays comfortably a little
* continuity  each part stays near where it just played (a smoothed centre
              per part), so a line does not wander between the parts
* sustain     notes still sounding count against a part's divisi limit, and
              a new note crossing a held note of another part costs
* divisi      more notes at once in a part than `divisi` costs

Greedy left to right, like the hand split of keyboard parts
(:func:`muscriptor.sheet.voices.split_hands`). A part left with almost
nothing is dissolved and the split redone without it: a melody in thirds is
one violin part, not a violin part with a few stray viola notes.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from muscriptor.sheet.articulation import merge_repeated_notes
from muscriptor.sheet.ir import PerformanceNote

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Section:
    name: str  # MuScriptor group the part is written as
    lowest: int  # lowest playable (sounding MIDI pitch)
    comfort_low: int
    comfort_high: int
    highest: int
    center: float  # where the part starts
    # Cost of giving this part anything: on a tie the violins carry the line.
    use_cost: float = 0.0


CELLO = Section("cello", 36, 36, 67, 76, 50.0, 0.2)  # C2, up to G4 / E5
VIOLA = Section("viola", 48, 53, 74, 81, 62.0, 0.4)  # C3
VIOLIN = Section("violin", 55, 62, 96, 103, 74.0)  # G3


@dataclass(frozen=True)
class SectionSplitOptions:
    # Groups treated as one string section.
    groups: tuple[str, ...] = ("string_ensemble", "violin", "viola", "cello")
    # Low to high.
    sections: tuple[Section, ...] = (CELLO, VIOLA, VIOLIN)
    chord_window: float = 0.04
    divisi: int = 2
    # A part with less than this share of the notes is dissolved.
    min_share: float = 0.08
    # Notes still sounding this long after a new onset count as held.
    hold_margin: float = 0.05
    # --- repeated notes (articulation.merge_repeated_notes): audio | legato | off
    articulation: str = "audio"  # audio | legato | off
    # --- tidy_part ---
    # Chord notes ending within this of the chord's median end, end together.
    align_window: float = 0.15
    # Gaps up to this between consecutive chords are closed (legato).
    legato_gap: float = 0.12
    # A note overhanging the next onset by up to this is cut there.
    overhang: float = 0.1


# ---------------------------------------------------------------------------
# Assignment


def _clusters(notes: list[PerformanceNote], window: float) -> list[list[int]]:
    """Indices of `notes` (sorted by onset) grouped into chords."""
    out: list[list[int]] = []
    for i, n in enumerate(notes):
        if out and n.onset - notes[out[-1][0]].onset <= window:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def _section_cost(
    s: int,
    pitches: list[int],
    held: list[list[int]],
    centers: list[float],
    secs: tuple[Section, ...],
    divisi: int,
) -> float:
    sec = secs[s]
    cost = 0.0
    for p in pitches:
        cost += 5.0 * (p < sec.lowest) + 2.0 * max(0, p - sec.highest)
        cost += 0.3 * max(0, sec.comfort_low - p) + 0.3 * max(0, p - sec.comfort_high)
    cost += 1.5 * max(0, len(pitches) + len(held[s]) - divisi)
    if pitches:
        cost += sec.use_cost
        cost += 0.1 * abs(sum(pitches) / len(pitches) - centers[s])
        lo, hi = min(pitches), max(pitches)
        for other, notes in enumerate(held):
            if other < s:  # lower part holding a note above ours
                cost += sum(1 for q in notes if q > lo)
            elif other > s:
                cost += sum(1 for q in notes if q < hi)
    return cost


def _assign(
    notes: list[PerformanceNote], secs: tuple[Section, ...], o: SectionSplitOptions
) -> list[int]:
    """Section index per note (notes sorted by onset)."""
    labels = [0] * len(notes)
    centers = [s.center for s in secs]
    sounding: list[int] = []
    for cluster in _clusters(notes, o.chord_window):
        t = notes[cluster[0]].onset
        sounding = [i for i in sounding if notes[i].offset > t + o.hold_margin]
        held = [
            [notes[i].pitch for i in sounding if labels[i] == s]
            for s in range(len(secs))
        ]
        cluster.sort(key=lambda i: notes[i].pitch)
        pitches = [notes[i].pitch for i in cluster]
        n = len(pitches)
        best, best_cost = None, float("inf")
        for cuts in itertools.combinations_with_replacement(
            range(n + 1), len(secs) - 1
        ):
            bounds = (0, *cuts, n)
            cost = sum(
                _section_cost(
                    s, pitches[bounds[s] : bounds[s + 1]], held, centers, secs, o.divisi
                )
                for s in range(len(secs))
            )
            if cost < best_cost:
                best, best_cost = bounds, cost
        for s in range(len(secs)):
            part = pitches[best[s] : best[s + 1]]
            for j in range(best[s], best[s + 1]):
                labels[cluster[j]] = s
            if part:
                centers[s] = 0.7 * centers[s] + 0.3 * sum(part) / len(part)
        sounding.extend(cluster)
    return labels


def assign_sections(
    notes: list[PerformanceNote], o: SectionSplitOptions = SectionSplitOptions()
) -> list[PerformanceNote]:
    """`notes` relabelled with section names (dissolving near-empty parts)."""
    notes = sorted(notes, key=lambda n: (n.onset, n.pitch))
    secs = o.sections
    while True:
        labels = _assign(notes, secs, o)
        counts = Counter(labels)
        if len(secs) == 1:
            break
        weakest = min(range(len(secs)), key=lambda s: counts[s])
        if counts[weakest] / max(len(notes), 1) >= o.min_share:
            break
        logger.info(
            "sections: %s would get %d notes, dissolved",
            secs[weakest].name,
            counts[weakest],
        )
        secs = secs[:weakest] + secs[weakest + 1 :]
    if len(secs) == 1 and len(o.sections) > 1:
        # One part: the section whose register fits it best.
        median = sorted(n.pitch for n in notes)[len(notes) // 2]
        secs = (min(o.sections, key=lambda s: abs(s.center - median)),)
        labels = [0] * len(notes)
    return [
        dataclasses.replace(n, instrument=secs[s].name) for n, s in zip(notes, labels)
    ]


# ---------------------------------------------------------------------------
# Tidying one part


def _fold_into_range(
    notes: list[PerformanceNote], sec: Section, window: float
) -> tuple[list[PerformanceNote], int, int]:
    """Notes below `sec.lowest` moved up by octaves; dropped if that doubles."""
    present = {
        (n.pitch, round(n.onset / window)) for n in notes if n.pitch >= sec.lowest
    }
    out, folded, dropped = [], 0, 0
    for n in notes:
        if n.pitch >= sec.lowest:
            out.append(n)
            continue
        p = n.pitch
        while p < sec.lowest:
            p += 12
        key = (p, round(n.onset / window))
        if key in present:
            dropped += 1
            continue
        present.add(key)
        out.append(dataclasses.replace(n, pitch=p))
        folded += 1
    return out, folded, dropped


def _cut_held_over_divisi(notes: list[PerformanceNote], o: SectionSplitOptions) -> int:
    """Cut held notes where a new chord leaves no room for them (in place).

    A note still sounding when the next chord starts is only kept if the
    part has a free desk for it (held + new ≤ `divisi`) and it lies wholly
    under or over the new chord — a pedal note, written as a second voice.
    Anything else ends where the new chord starts. Returns how many were cut.
    """
    cut = 0
    onsets = sorted({n.onset for n in notes})
    by_onset: dict[float, list[PerformanceNote]] = {}
    for n in notes:
        by_onset.setdefault(n.onset, []).append(n)
    for t in onsets:
        chord = by_onset[t]
        held = sorted(
            (n for n in notes if n.onset < t and n.offset > t + 1e-6),
            key=lambda n: n.pitch,
        )
        if not held:
            continue
        lo = min(n.pitch for n in chord)
        hi = max(n.pitch for n in chord)
        room = max(0, o.divisi - len(chord))
        # Outer held notes first: the lowest pedal, then the highest.
        candidates = [n for n in held if n.pitch < lo][:1] + [
            n for n in held if n.pitch > hi
        ][-1:]
        keep = {id(n) for n in candidates[:room]}
        for n in held:
            if id(n) not in keep:
                n.offset = t
                cut += 1
    return cut


def tidy_part(
    notes: list[PerformanceNote],
    sec: Section,
    o: SectionSplitOptions = SectionSplitOptions(),
) -> list[PerformanceNote]:
    """Lengths and overlaps of one string part, made writable.

    * notes under the instrument's lowest string fold up an octave (or go,
      when that doubles a note already played)
    * at most `divisi` notes per chord: the outer ones are kept
    * the notes of a chord end together when they nearly do
    * a note overhanging the next chord a little is cut at it; a short gap
      before the next chord is closed (bowed strings play legato)
    * a note never overlaps the next note of the same pitch

    After this a part is a sequence of chords plus, at most, a held note
    under or over them — exactly what two voices on one staff can write.
    """
    if not notes:
        return notes
    notes, folded, dropped = _fold_into_range(notes, sec, o.chord_window)
    notes = sorted(notes, key=lambda n: (n.onset, n.pitch))
    clusters = _clusters(notes, o.chord_window)
    capped = 0
    chords: list[list[PerformanceNote]] = []
    for c in clusters:
        chord = sorted((notes[i] for i in c), key=lambda n: n.pitch)
        # Same pitch twice in one chord: one note.
        uniq = {n.pitch: n for n in chord}
        chord = [uniq[p] for p in sorted(uniq)]
        if len(chord) > o.divisi:
            keep = [chord[0], chord[-1]] if o.divisi >= 2 else [chord[-1]]
            capped += len(chord) - len(keep)
            chord = keep
        onset = min(n.onset for n in chord)
        ends = sorted(n.offset for n in chord)
        median_end = ends[len(ends) // 2]
        chord = [
            dataclasses.replace(
                n,
                onset=onset,
                offset=median_end
                if abs(n.offset - median_end) <= o.align_window
                else n.offset,
            )
            for n in chord
        ]
        chords.append(chord)

    out: list[PerformanceNote] = []
    for k, chord in enumerate(chords):
        nxt = chords[k + 1][0].onset if k + 1 < len(chords) else None
        for n in chord:
            end = n.offset
            if nxt is not None and (
                0 < nxt - end <= o.legato_gap or 0 < end - nxt <= o.overhang
            ):
                end = nxt
            out.append(dataclasses.replace(n, offset=max(end, n.onset + 0.02)))
    cut = _cut_held_over_divisi(out, o)
    # Same pitch: a note ends where the next one starts.
    by_pitch: dict[int, list[PerformanceNote]] = {}
    for n in sorted(out, key=lambda n: n.onset):
        prev = by_pitch.get(n.pitch)
        if prev is not None and prev.offset > n.onset:
            prev.offset = n.onset
        by_pitch[n.pitch] = n
    out = [n for n in out if n.offset - n.onset >= 0.02]
    if folded or dropped or capped or cut:
        logger.info(
            "sections: %s: %d notes folded into range, %d doublings dropped, "
            "%d inner chord notes over divisi %d dropped, %d held notes cut "
            "at the next chord",
            sec.name,
            folded,
            dropped,
            capped,
            o.divisi,
            cut,
        )
    return sorted(out, key=lambda n: (n.onset, n.pitch))


# ---------------------------------------------------------------------------


def split_string_sections(
    notes: list[PerformanceNote],
    o: SectionSplitOptions = SectionSplitOptions(),
    audio: dict[str, Path] | None = None,
) -> list[PerformanceNote]:
    """All notes, the string section re-articulated, split and tidied.

    `audio` maps a note's `source` ("other", "mix", ...) to the audio it was
    transcribed from, for the repeated-note check.
    """
    section = [n for n in notes if n.instrument in o.groups]
    if not section:
        return notes
    rest = [n for n in notes if n.instrument not in o.groups]
    # 1. Re-articulation, per source (each checked against its own audio)
    #    and per original label (a violin line is not glued to the ensemble).
    merged: list[PerformanceNote] = []
    keys = sorted({(n.source, n.instrument) for n in section})
    for source, inst in keys:
        part = [n for n in section if n.source == source and n.instrument == inst]
        merged += merge_repeated_notes(part, o.articulation, (audio or {}).get(source))
    # 2. Sections, 3. tidy.
    assigned = assign_sections(merged, o)
    by_name = {s.name: s for s in o.sections}
    out_parts: list[PerformanceNote] = []
    for name in sorted({n.instrument for n in assigned}):
        out_parts += tidy_part(
            [n for n in assigned if n.instrument == name], by_name[name], o
        )
    counts = Counter(n.instrument for n in out_parts)
    logger.info(
        "sections: %d string-section notes → %d after re-articulation → %s",
        len(section),
        len(merged),
        ", ".join(f"{k} {v}" for k, v in counts.most_common()),
    )
    out = rest + out_parts
    out.sort(key=lambda n: (n.onset, n.pitch))
    return out


__all__ = [
    "CELLO",
    "VIOLA",
    "VIOLIN",
    "Section",
    "SectionSplitOptions",
    "assign_sections",
    "split_string_sections",
    "tidy_part",
]
