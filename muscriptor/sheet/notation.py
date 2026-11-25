"""Build the Notation IR: measures, readable durations, ties, rests, spelling.

Input: per part, notes with exact beat positions and staff/voice assigned
(see voices.py). Output: :class:`~muscriptor.sheet.ir.Score`.

The readability rules are the conventional engraving ones:

* nothing crosses a bar line (tie instead);
* a note starting on a beat may span beats, but not across the middle of a
  4/4 (or 6/4) bar unless it starts on the downbeat;
* a note starting off the beat never crosses the next beat (tie instead), so
  the beat stays visible in syncopations;
* only plain or single-dotted values are used;
* a beat whose notes fall on thirds/sixths is written as a triplet
  (one bracket per beat); everything else is binary;
* an empty measure in a voice is one whole-measure rest; an empty second
  voice is simply not written.

Pitch spelling follows the key: each pitch class takes the spelling closest
to the key's centre on the line of fifths, so B-flat major gets E-flat and
A-flat rather than D-sharp and G-sharp. Accidentals are then derived per
measure and staff from the key signature and earlier notes in the bar.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction

from muscriptor.sheet.instruments import InstrumentSpec, drum_spec
from muscriptor.sheet.ir import (
    ChordSymbol,
    DrumHead,
    Event,
    Measure,
    Part,
    Pitch,
    QuantizedNote,
    Score,
    Tuplet,
    Voice,
)
from muscriptor.sheet.voices import StaffLayout

logger = logging.getLogger(__name__)

F = Fraction

# ---------------------------------------------------------------------------
# Pitch spelling

_STEPS = ["F", "C", "G", "D", "A", "E", "B"]  # line of fifths, F = -1
_PC_OF_STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _lof_spellings(pc: int) -> list[tuple[int, str, int]]:
    """(line-of-fifths position, step, alter) for every spelling of `pc` with |alter| ≤ 2."""
    out = []
    for lof in range(-15, 20):
        step = _STEPS[(lof + 1) % 7]
        alter = (lof + 1) // 7
        if abs(alter) <= 2 and (_PC_OF_STEP[step] + alter) % 12 == pc:
            out.append((lof, step, alter))
    return out


def spell(midi: int, key_fifths: int, mode: str = "major") -> Pitch:
    """Spell a MIDI pitch in a key (fifths = number of sharps, negative flats)."""
    # The diatonic notes of a major key span fifths-1 .. fifths+5 on the line
    # of fifths; its centre is fifths+2. Minor keys share the relative
    # major's signature, but their raised 6th/7th (leading tone) sit at the
    # sharp end, so the centre moves up by one.
    center = key_fifths + 2 + (1 if mode == "minor" else 0)
    pc = midi % 12
    lof, step, alter = min(
        _lof_spellings(pc), key=lambda s: (abs(s[0] - center), abs(s[2]))
    )
    octave = (midi - alter) // 12 - 1
    return Pitch(step=step, alter=alter, octave=octave, midi=midi)


def key_alterations(fifths: int) -> dict[str, int]:
    """The alteration the key signature gives each step."""
    alters = {s: 0 for s in "CDEFGAB"}
    if fifths > 0:
        for s in ["F", "C", "G", "D", "A", "E", "B"][:fifths]:
            alters[s] = 1
    elif fifths < 0:
        for s in ["B", "E", "A", "D", "G", "C", "F"][:-fifths]:
            alters[s] = -1
    return alters


_ACCIDENTAL_NAMES = {
    -2: "flat-flat",
    -1: "flat",
    0: "natural",
    1: "sharp",
    2: "double-sharp",
}


def mark_accidentals(measure: Measure, key_fifths: int) -> None:
    """Set Pitch.accidental for the notes of one measure, per staff."""
    by_staff: dict[int, list[Event]] = defaultdict(list)
    for v in measure.voices:
        by_staff[v.staff].extend(v.events)
    for events in by_staff.values():
        state = {}  # (step, octave) → current alteration
        base = key_alterations(key_fifths)
        for ev in sorted(events, key=lambda e: e.offset):
            for p in ev.pitches:
                key = (p.step, p.octave)
                current = state.get(key, base[p.step])
                if ev.tie_stop and ev.offset == 0:
                    # A note tied over the bar line repeats no accidental and
                    # leaves the bar's state alone.
                    continue
                if p.alter != current:
                    p.accidental = _ACCIDENTAL_NAMES[p.alter]
                    state[key] = p.alter


# ---------------------------------------------------------------------------
# Durations


def strong_boundaries(beats_per_bar: int) -> list[int]:
    """Beat positions a note starting later on a beat may not cross."""
    if beats_per_bar == 4:
        return [2]
    if beats_per_bar == 6:
        return [3]
    if beats_per_bar == 5:
        return [3]
    return []


# Plain and single-dotted note values (quarter = 1), longest first.
_BINARY_VALUES = [
    F(4),
    F(3),
    F(2),
    F(3, 2),
    F(1),
    F(3, 4),
    F(1, 2),
    F(3, 8),
    F(1, 4),
    F(3, 16),
    F(1, 8),
]
# Values inside a triplet beat, as sounding lengths (written value × 2/3).
_TRIPLET_VALUES = [F(2, 3), F(1, 2), F(1, 3), F(1, 4), F(1, 6), F(1, 12)]


def _undotted(value: Fraction) -> Fraction:
    """The base value of a (possibly dotted) duration."""
    if value.numerator == 3:
        return value * F(2, 3)
    return value


def _is_binary(x: Fraction) -> bool:
    d = x.denominator
    return d & (d - 1) == 0


def next_piece(
    t: Fraction,
    end: Fraction,
    bar_len: Fraction,
    beats_per_bar: int,
    triplet_beats: set[int],
) -> tuple[Fraction, bool]:
    """Length of the next notatable piece of a span [t, end) in a measure.

    Returns (length, is_triplet). Positions are relative to the measure.
    """
    beat = math.floor(t)
    if beat in triplet_beats:
        limit = min(end, F(beat + 1))
        avail = limit - t
        pos = t - beat
        for v in _TRIPLET_VALUES:
            base = _undotted(v * F(3, 2)) * F(2, 3)  # sounding length of the base value
            if v <= avail and (pos / base).denominator == 1:
                return v, True
        return avail, True

    # Never run into a triplet beat.
    limit = end
    for b in sorted(triplet_beats):
        if b >= t:
            limit = min(limit, F(b))
            break
    if t.denominator == 1:  # on a beat
        if t != 0:
            for s in strong_boundaries(beats_per_bar):
                if t < s:
                    limit = min(limit, F(s))
                    break
        avail = limit - t
        for v in _BINARY_VALUES:
            if v > avail:
                continue
            base = _undotted(v)
            if v >= 1 and base >= 1 and (t / base).denominator != 1 and t != 0:
                # A half note must start on beat 1 or 3 (4/4), etc. Dotted
                # values only start on the beat they belong to.
                if not (beats_per_bar == 3 and base == 2):
                    continue
            if v == F(3) and t != 0:
                continue
            return v, False
        return avail, False

    # The syncopation every pop chart writes as one note: a quarter (or a
    # dotted quarter) starting on the "and", as long as it neither leaves the
    # half bar nor ends off the eighth grid ("eighth – quarter – eighth").
    pos = t - beat
    if pos == F(1, 2) and beats_per_bar in (2, 4):
        half_end = F(2) if t < 2 else F(beats_per_bar)
        for v in (F(3, 2), F(1)):
            if t + v <= min(limit, half_end) and t + v <= end:
                return v, False

    # Off the beat: stay inside the beat.
    limit = min(limit, F(beat + 1))
    avail = limit - t
    pos = t - beat
    for v in _BINARY_VALUES:
        if v >= 1 or v > avail:
            continue
        base = _undotted(v)
        if v == base:
            ok = (pos / base).denominator == 1
        else:
            # A dotted value off the beat only where it lines up with its
            # own beat division, or where it ends exactly on the next beat
            # (sixteenth + dotted eighth).
            ok = (pos / (2 * base)).denominator == 1 or pos + v == 1
        if ok:
            return v, False
    return avail, False


def split_span(
    start: Fraction,
    end: Fraction,
    bar_len: Fraction,
    beats_per_bar: int,
    triplet_beats: set[int],
) -> list[tuple[Fraction, Fraction, bool]]:
    """[start, end) within one measure as notatable (offset, length, triplet) pieces."""
    out = []
    t = start
    while t < end:
        length, trip = next_piece(t, end, bar_len, beats_per_bar, triplet_beats)
        if length <= 0:
            break
        out.append((t, length, trip))
        t += length
    return out


# ---------------------------------------------------------------------------
# Measure building


@dataclass
class _Span:
    start: Fraction  # within the measure
    end: Fraction
    pitches: list[int]
    tie_in: bool = False
    tie_out: bool = False
    drums: bool = False


def _defensive_snap(x: Fraction) -> Fraction:
    """Positions are always multiples of 1/8 or 1/6 of a beat; enforce it."""
    if (x * 24).denominator == 1:
        return x
    return F(round(x * 24), 24)


def _voice_spans(
    notes: list[QuantizedNote], bar_len: Fraction, n_measures: int
) -> dict[int, list[_Span]]:
    """Group a voice's notes into chords and split them at bar lines, per measure."""
    chords: dict[tuple[Fraction, Fraction], list[int]] = defaultdict(list)
    for n in notes:
        chords[(_defensive_snap(n.onset), _defensive_snap(n.offset))].append(n.pitch)
    spans: dict[int, list[_Span]] = defaultdict(list)
    last_end = F(-1)
    for (on, off), pitches in sorted(chords.items()):
        if on < last_end:  # defensive: never overlap inside a voice
            on = last_end
        if off <= on:
            continue
        last_end = off
        m = int(on // bar_len)
        t = on
        while t < off and m < n_measures:
            m_start = m * bar_len
            seg_end = min(off, m_start + bar_len)
            spans[m].append(
                _Span(
                    t - m_start,
                    seg_end - m_start,
                    sorted(set(pitches)),
                    tie_in=t > on,
                    tie_out=seg_end < off,
                )
            )
            t = seg_end
            m += 1
    return spans


def _triplet_beats(spans: list[_Span]) -> set[int]:
    beats = set()
    for s in spans:
        for x in (s.start, s.end):
            if not _is_binary(x - math.floor(x)) and x.denominator != 1:
                beats.add(math.floor(x))
    return beats


def _events_for_voice(
    spans: list[_Span],
    bar_len: Fraction,
    beats_per_bar: int,
    pitched: bool,
    spell_fn,
    is_drum: bool,
    fill_measure: bool,
) -> list[Event]:
    """Notes and rests of one voice in one measure."""
    if not spans:
        if not fill_measure:
            return []
        return [Event(offset=F(0), duration=bar_len, measure_rest=True)]

    triplets = _triplet_beats(spans)
    # Rests for the gaps, then the notes, as spans in time order.
    timeline: list[_Span | tuple[Fraction, Fraction]] = []
    t = F(0)
    for s in sorted(spans, key=lambda s: s.start):
        if s.start > t:
            timeline.append((t, s.start))
        timeline.append(s)
        t = max(t, s.end)
    if t < bar_len:
        timeline.append((t, bar_len))

    events: list[Event] = []
    for item in timeline:
        if isinstance(item, tuple):
            a, b = item
            for off, length, trip in split_span(a, b, bar_len, beats_per_bar, triplets):
                events.append(
                    Event(
                        offset=off,
                        duration=length,
                        tuplet=Tuplet(3, 2) if trip else None,
                    )
                )
            continue
        pieces = split_span(item.start, item.end, bar_len, beats_per_bar, triplets)
        for i, (off, length, trip) in enumerate(pieces):
            ev = Event(
                offset=off, duration=length, tuplet=Tuplet(3, 2) if trip else None
            )
            if is_drum:
                if i > 0:
                    continue  # drum hits are never tied: drop the continuation
                for p in item.pitches:
                    key, spec = drum_spec(p)
                    ev.drums.append(
                        DrumHead(key, spec.step, spec.octave, spec.notehead)
                    )
            else:
                ev.pitches = [spell_fn(p) for p in item.pitches]
                ev.tie_start = i < len(pieces) - 1 or item.tie_out
                ev.tie_stop = i > 0 or item.tie_in
            events.append(ev)
        if is_drum and len(pieces) > 1:
            # Replace the dropped continuation with rests.
            for off, length, trip in pieces[1:]:
                events.append(
                    Event(
                        offset=off,
                        duration=length,
                        tuplet=Tuplet(3, 2) if trip else None,
                    )
                )
    events.sort(key=lambda e: e.offset)
    _mark_tuplet_brackets(events)
    return events


def _mark_tuplet_brackets(events: list[Event]) -> None:
    """One bracket per triplet beat: start on its first event, stop on its last."""
    by_beat: dict[int, list[Event]] = defaultdict(list)
    for e in events:
        if e.tuplet is not None:
            by_beat[math.floor(e.offset)].append(e)
    for group in by_beat.values():
        group[0].tuplet.start = True
        group[-1].tuplet.stop = True


# ---------------------------------------------------------------------------


@dataclass
class PartInput:
    spec: InstrumentSpec
    layout: StaffLayout
    notes: list[QuantizedNote]  # staff/voice assigned
    part_id: str = ""


@dataclass
class ScoreContext:
    """Everything outside the parts that the measures carry."""

    beats_per_bar: int
    n_measures: int
    title: str
    # measure index → (fifths, mode); measure 0 must be present.
    keys: dict[int, tuple[int, str]] = field(default_factory=lambda: {0: (0, "major")})
    # measure index → BPM, where a metronome mark goes.
    tempos: dict[int, float] = field(default_factory=dict)
    # measure index → symbols (offset within measure set).
    chords: dict[int, list[ChordSymbol]] = field(default_factory=dict)
    # measure index → rehearsal label
    sections: dict[int, str] = field(default_factory=dict)
    # Parts that carry the chord symbols.
    chord_parts: set[str] = field(default_factory=set)
    words: dict[int, str] = field(default_factory=dict)
    measure_seconds: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def voice_number(staff: int, voice: int) -> int:
    """MusicXML voice number: 1-4 on staff 1, 5-8 on staff 2 (MuseScore's convention)."""
    return (staff - 1) * 4 + voice


def build_part(pi: PartInput, ctx: ScoreContext) -> Part:
    spec, layout = pi.spec, pi.layout
    bar_len = F(ctx.beats_per_bar)
    is_drum = spec.role == "drums"
    shift = 12 * layout.written_octave_shift

    part = Part(
        id=pi.part_id,
        name=spec.name,
        abbreviation=spec.abbreviation,
        instrument=spec.group,
        role=spec.role,
        clefs=layout.clefs,
        gm_program=spec.gm_program,
        is_drum=is_drum,
        written_octave_shift=layout.written_octave_shift,
    )
    if is_drum:
        part.drum_keys = sorted({drum_spec(n.pitch)[0] for n in pi.notes})

    by_sv: dict[tuple[int, int], list[QuantizedNote]] = defaultdict(list)
    for n in pi.notes:
        by_sv[(n.staff, n.voice)].append(n)
    spans_by_sv = {
        sv: _voice_spans(ns, bar_len, ctx.n_measures) for sv, ns in by_sv.items()
    }

    key = ctx.keys.get(0, (0, "major"))
    for m in range(ctx.n_measures):
        if m in ctx.keys:
            key = ctx.keys[m]
        fifths, mode = key
        measure = Measure(number=m + 1, length=bar_len)
        if m == 0:
            measure.time_signature = (ctx.beats_per_bar, 4)
        if m in ctx.keys and (
            m == 0 or ctx.keys[m] != ctx.keys.get(_prev_key_measure(ctx.keys, m))
        ):
            measure.key_fifths, measure.key_mode = fifths, mode
        measure.tempo = ctx.tempos.get(m)
        measure.rehearsal = ctx.sections.get(m)
        measure.words = ctx.words.get(m)
        measure.double_bar = (m + 1) in ctx.sections
        if m < len(ctx.measure_seconds):
            measure.start_seconds = ctx.measure_seconds[m]
        if pi.part_id in ctx.chord_parts:
            measure.chords = list(ctx.chords.get(m, []))

        def spell_fn(midi: int, _f=fifths, _mode=mode):
            return spell(midi + shift, _f, _mode)

        for staff in range(1, len(layout.clefs) + 1):
            for v in (1, 2):
                spans = spans_by_sv.get((staff, v), {}).get(m, [])
                events = _events_for_voice(
                    spans,
                    bar_len,
                    ctx.beats_per_bar,
                    not is_drum,
                    spell_fn,
                    is_drum,
                    fill_measure=(v == 1),
                )
                if events:
                    measure.voices.append(
                        Voice(number=voice_number(staff, v), staff=staff, events=events)
                    )
        if not is_drum:
            mark_accidentals(measure, fifths)
        part.measures.append(measure)
    return part


def _prev_key_measure(keys: dict[int, tuple[int, str]], m: int) -> int:
    earlier = [k for k in keys if k < m]
    return max(earlier) if earlier else 0


def build_score(parts: list[PartInput], ctx: ScoreContext) -> Score:
    built = [build_part(p, ctx) for p in parts]
    return Score(title=ctx.title, parts=built, metadata=dict(ctx.metadata))
