"""Notation IR → MusicXML 4.0 (partwise).

A deterministic, dependency-free writer: the same Score always produces the
same bytes (no timestamps), which keeps the output diffable and testable.
All notation decisions were made upstream; this module only encodes them.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

from muscriptor.sheet.instruments import DRUM_MAP
from muscriptor.sheet.ir import ChordSymbol, Event, Measure, Part, Score

_TYPES = {
    Fraction(4): "whole",
    Fraction(2): "half",
    Fraction(1): "quarter",
    Fraction(1, 2): "eighth",
    Fraction(1, 4): "16th",
    Fraction(1, 8): "32nd",
    Fraction(1, 16): "64th",
}

DOCTYPE = (
    '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
    '"http://www.musicxml.org/dtds/partwise.dtd">'
)


class MusicXMLError(ValueError):
    """The IR holds something MusicXML cannot express (a bug upstream)."""


def note_type(written: Fraction) -> tuple[str, int]:
    """(MusicXML type, number of dots) of a written duration in quarter notes."""
    if written in _TYPES:
        return _TYPES[written], 0
    if written.numerator == 3 and written * Fraction(2, 3) in _TYPES:
        return _TYPES[written * Fraction(2, 3)], 1
    if written.numerator == 7 and written * Fraction(4, 7) in _TYPES:
        return _TYPES[written * Fraction(4, 7)], 2
    raise MusicXMLError(f"no note value for a duration of {written} quarter notes")


def divisions_for(score: Score) -> int:
    """Smallest <divisions> (per quarter) that makes every duration an integer."""
    lcm = 1
    for part in score.parts:
        for m in part.measures:
            for v in m.voices:
                for e in v.events:
                    lcm = math.lcm(lcm, e.duration.denominator, e.offset.denominator)
            for c in m.chords:
                lcm = math.lcm(lcm, c.offset.denominator)
    return lcm


def _sub(parent: ET.Element, tag: str, _text=None, **attrib) -> ET.Element:
    el = ET.SubElement(
        parent, tag, {k.replace("_", "-"): str(v) for k, v in attrib.items()}
    )
    if _text is not None:
        el.text = str(_text)
    return el


def _dur(x: Fraction, div: int) -> int:
    v = x * div
    if v.denominator != 1:
        raise MusicXMLError(f"duration {x} is not a multiple of 1/{div}")
    return int(v)


# ---------------------------------------------------------------------------


def _part_list(root: ET.Element, score: Score) -> None:
    pl = _sub(root, "part-list")
    channel = 0
    for part in score.parts:
        sp = _sub(pl, "score-part", id=part.id)
        _sub(sp, "part-name", part.name)
        _sub(sp, "part-abbreviation", part.abbreviation)
        if part.is_drum:
            for key in part.drum_keys:
                si = _sub(sp, "score-instrument", id=f"{part.id}-I{key}")
                _sub(si, "instrument-name", DRUM_MAP[key].name)
            for key in part.drum_keys:
                mi = _sub(sp, "midi-instrument", id=f"{part.id}-I{key}")
                _sub(mi, "midi-channel", 10)
                _sub(mi, "midi-unpitched", key + 1)
        else:
            channel += 1
            if channel == 10:
                channel += 1
            si = _sub(sp, "score-instrument", id=f"{part.id}-I1")
            _sub(si, "instrument-name", part.name)
            mi = _sub(sp, "midi-instrument", id=f"{part.id}-I1")
            _sub(mi, "midi-channel", min(channel, 16))
            _sub(mi, "midi-program", part.gm_program + 1)


def _attributes(
    meas_el: ET.Element, part: Part, m: Measure, div: int, first: bool
) -> None:
    if not (first or m.key_fifths is not None or m.time_signature is not None):
        return
    at = _sub(meas_el, "attributes")
    if first:
        _sub(at, "divisions", div)
    if m.key_fifths is not None:
        key = _sub(at, "key")
        _sub(key, "fifths", m.key_fifths)
        _sub(key, "mode", m.key_mode)
    if m.time_signature is not None:
        t = _sub(at, "time")
        _sub(t, "beats", m.time_signature[0])
        _sub(t, "beat-type", m.time_signature[1])
    if first:
        if part.staves > 1:
            _sub(at, "staves", part.staves)
        for i, clef in enumerate(part.clefs, start=1):
            c = _sub(at, "clef", number=i) if part.staves > 1 else _sub(at, "clef")
            _sub(c, "sign", clef.sign)
            _sub(c, "line", clef.line)
            if clef.octave_change:
                _sub(c, "clef-octave-change", clef.octave_change)
        if part.written_octave_shift:
            tr = _sub(at, "transpose")
            _sub(tr, "diatonic", 0)
            _sub(tr, "chromatic", 0)
            _sub(tr, "octave-change", -part.written_octave_shift)


def _directions(meas_el: ET.Element, m: Measure) -> None:
    if m.rehearsal:
        d = _sub(meas_el, "direction", placement="above")
        dt = _sub(d, "direction-type")
        _sub(dt, "rehearsal", m.rehearsal)
    if m.tempo is not None:
        bpm = int(round(m.tempo))
        d = _sub(meas_el, "direction", placement="above")
        dt = _sub(d, "direction-type")
        met = _sub(dt, "metronome", parentheses="no")
        _sub(met, "beat-unit", "quarter")
        _sub(met, "per-minute", bpm)
        _sub(d, "sound", tempo=bpm)
    if m.words:
        d = _sub(meas_el, "direction", placement="above")
        dt = _sub(d, "direction-type")
        _sub(dt, "words", m.words)


def _harmony(meas_el: ET.Element, chord: ChordSymbol) -> None:
    h = _sub(meas_el, "harmony", print_frame="no")
    root = _sub(h, "root")
    _sub(root, "root-step", chord.root_step)
    if chord.root_alter:
        _sub(root, "root-alter", chord.root_alter)
    _sub(h, "kind", chord.kind, text=chord.text)
    if chord.bass_step:
        bass = _sub(h, "bass")
        _sub(bass, "bass-step", chord.bass_step)
        if chord.bass_alter:
            _sub(bass, "bass-alter", chord.bass_alter)
    for value, alter, kind in chord.degrees:
        deg = _sub(h, "degree")
        _sub(deg, "degree-value", value)
        _sub(deg, "degree-alter", alter)
        _sub(deg, "degree-type", kind)


def _write_harmonies(meas_el: ET.Element, m: Measure, div: int) -> None:
    """Chord symbols at their offsets, then back to the start of the measure."""
    pos = Fraction(0)
    for chord in sorted(m.chords, key=lambda c: c.offset):
        if chord.offset > pos:
            fw = _sub(meas_el, "forward")
            _sub(fw, "duration", _dur(chord.offset - pos, div))
            pos = chord.offset
        _harmony(meas_el, chord)
    if pos > 0:
        bk = _sub(meas_el, "backup")
        _sub(bk, "duration", _dur(pos, div))


def _note(
    meas_el: ET.Element,
    part: Part,
    ev: Event,
    voice: int,
    staff: int,
    div: int,
    stem: str | None,
) -> None:
    written = ev.duration * Fraction(3, 2) if ev.tuplet else ev.duration
    heads: list = ev.pitches or ev.drums or [None]
    for i, head in enumerate(heads):
        n = _sub(meas_el, "note")
        if i > 0:
            _sub(n, "chord")
        if head is None:
            if ev.measure_rest:
                _sub(n, "rest", measure="yes")
            else:
                _sub(n, "rest")
        elif ev.drums:
            up = _sub(n, "unpitched")
            _sub(up, "display-step", head.step)
            _sub(up, "display-octave", head.octave)
        else:
            p = _sub(n, "pitch")
            _sub(p, "step", head.step)
            if head.alter:
                _sub(p, "alter", head.alter)
            _sub(p, "octave", head.octave)
        _sub(n, "duration", _dur(ev.duration, div))
        if ev.tie_stop and head is not None and not ev.drums:
            _sub(n, "tie", type="stop")
        if ev.tie_start and head is not None and not ev.drums:
            _sub(n, "tie", type="start")
        if ev.drums and head is not None:
            _sub(n, "instrument", id=f"{part.id}-I{head.midi}")
        _sub(n, "voice", voice)
        if not ev.measure_rest:
            t, dots = note_type(written)
            _sub(n, "type", t)
            for _ in range(dots):
                _sub(n, "dot")
        if head is not None and not ev.drums and head.accidental:
            _sub(n, "accidental", head.accidental)
        if ev.tuplet:
            tm = _sub(n, "time-modification")
            _sub(tm, "actual-notes", ev.tuplet.actual)
            _sub(tm, "normal-notes", ev.tuplet.normal)
        if head is not None and stem:
            _sub(n, "stem", stem)
        if ev.drums and head is not None and head.notehead != "normal":
            _sub(n, "notehead", head.notehead)
        if part.staves > 1:
            _sub(n, "staff", staff)
        notations = []
        if head is not None and not ev.drums and (ev.tie_start or ev.tie_stop):
            notations.append(("tied", ev))
        if ev.tuplet and i == 0 and (ev.tuplet.start or ev.tuplet.stop):
            notations.append(("tuplet", ev))
        if notations:
            nt = _sub(n, "notations")
            if ev.tie_stop and head is not None and not ev.drums:
                _sub(nt, "tied", type="stop")
            if ev.tie_start and head is not None and not ev.drums:
                _sub(nt, "tied", type="start")
            if ev.tuplet and i == 0:
                if ev.tuplet.start:
                    _sub(nt, "tuplet", type="start", bracket="yes")
                if ev.tuplet.stop:
                    _sub(nt, "tuplet", type="stop")


def _measure(
    part_el: ET.Element,
    part: Part,
    m: Measure,
    div: int,
    first: bool,
    last: bool,
    system_text: bool,
    chords: bool,
) -> None:
    me = _sub(part_el, "measure", number=m.number)
    _attributes(me, part, m, div, first)
    if system_text:
        _directions(me, m)
    if chords and m.chords:
        _write_harmonies(me, m, div)

    staves_with_two = {
        s
        for s in {v.staff for v in m.voices}
        if sum(1 for v in m.voices if v.staff == s) > 1
    }
    voices = sorted(m.voices, key=lambda v: (v.staff, v.number))
    for i, v in enumerate(voices):
        if part.is_drum or v.staff in staves_with_two:
            stem = "up" if v.number % 2 == 1 else "down"
        else:
            stem = None
        total = Fraction(0)
        for ev in v.events:
            _note(me, part, ev, v.number, v.staff, div, stem)
            total += ev.duration
        if total != m.length:
            raise MusicXMLError(
                f"{part.name} measure {m.number} voice {v.number}: durations sum to "
                f"{total}, not {m.length}"
            )
        if i < len(voices) - 1:
            bk = _sub(me, "backup")
            _sub(bk, "duration", _dur(m.length, div))

    if last:
        bl = _sub(me, "barline", location="right")
        _sub(bl, "bar-style", "light-heavy")
    elif m.double_bar:
        bl = _sub(me, "barline", location="right")
        _sub(bl, "bar-style", "light-light")


def score_to_musicxml(score: Score, chord_part_ids: set[str] | None = None) -> bytes:
    """Serialize a Score as MusicXML 4.0 bytes.

    Rehearsal marks, tempo and other system text go on the top part only
    (notation software shows them once per system from there).
    `chord_part_ids` limits chord symbols to those parts; None writes every
    part's own symbols.
    """
    div = divisions_for(score)
    root = ET.Element("score-partwise", version="4.0")
    work = _sub(root, "work")
    _sub(work, "work-title", score.title)
    ident = _sub(root, "identification")
    if score.composer:
        _sub(ident, "creator", score.composer, type="composer")
    enc = _sub(ident, "encoding")
    _sub(enc, "software", "muscriptor audio-to-sheet")
    if score.source:
        _sub(ident, "source", score.source)

    _part_list(root, score)
    for index, part in enumerate(score.parts):
        pe = _sub(root, "part", id=part.id)
        n = len(part.measures)
        chords = chord_part_ids is None or part.id in chord_part_ids
        for i, m in enumerate(part.measures):
            _measure(
                pe,
                part,
                m,
                div,
                first=i == 0,
                last=i == n - 1,
                system_text=index == 0,
                chords=chords,
            )

    ET.indent(root, space=" ")
    body = ET.tostring(root, encoding="unicode")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n' + DOCTYPE + "\n" + body + "\n"
    ).encode("utf-8")


def write_musicxml(
    score: Score, path: Path, chord_part_ids: set[str] | None = None
) -> Path:
    path.write_bytes(score_to_musicxml(score, chord_part_ids))
    return path


def single_part(score: Score, part_id: str) -> Score:
    """A copy of `score` with only one part (for per-instrument files)."""
    return Score(
        title=score.title,
        parts=[p for p in score.parts if p.id == part_id],
        composer=score.composer,
        source=score.source,
        metadata=score.metadata,
    )


__all__ = ["score_to_musicxml", "write_musicxml", "single_part", "note_type"]
