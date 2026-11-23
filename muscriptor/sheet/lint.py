"""Readability metrics for a Notation IR score.

Note-level F1 says nothing about whether a score can be read. These counts
flag what makes a transcription unreadable — fragmented rests, very short
values, tie chains, tuplet clutter, accidentals, crowded bars — so changes to
the reconstruction stages can be compared on the same song.
"""

from __future__ import annotations

from fractions import Fraction

from muscriptor.sheet.ir import Score


def lint_score(score: Score) -> dict:
    report: dict = {"parts": {}}
    totals = dict(
        events=0,
        notes=0,
        rests=0,
        short_rests=0,
        thirtyseconds=0,
        ties=0,
        tuplet_notes=0,
        accidentals=0,
    )
    for part in score.parts:
        p = {k: 0 for k in totals}
        busiest = 0
        empty = 0
        two_voice_measures = 0
        chords = 0
        for m in part.measures:
            chords += len(m.chords)
            n_events = 0
            staves = {}
            for v in m.voices:
                staves[v.staff] = staves.get(v.staff, 0) + 1
                for e in v.events:
                    n_events += 1
                    if e.is_rest:
                        if not e.hidden:
                            p["rests"] += 1
                            if e.duration < Fraction(1, 2) and not e.measure_rest:
                                p["short_rests"] += 1
                        continue
                    p["notes"] += 1
                    if e.duration < Fraction(1, 4) and not e.tuplet:
                        p["thirtyseconds"] += 1
                    if e.tie_start:
                        p["ties"] += 1
                    if e.tuplet:
                        p["tuplet_notes"] += 1
                    p["accidentals"] += sum(1 for x in e.pitches if x.accidental)
            p["events"] += n_events
            busiest = max(busiest, n_events)
            if all(e.is_rest for v in m.voices for e in v.events):
                empty += 1
            if any(c > 1 for c in staves.values()):
                two_voice_measures += 1
        notes = max(p["notes"], 1)
        p.update(
            measures=len(part.measures),
            empty_measures=empty,
            busiest_measure_events=busiest,
            two_voice_measures=two_voice_measures,
            chord_symbols=chords,
            tie_ratio=round(p["ties"] / notes, 3),
            short_rest_ratio=round(p["short_rests"] / max(p["rests"], 1), 3),
            events_per_measure=round(
                p["events"] / max(len(part.measures) - empty, 1), 1
            ),
        )
        report["parts"][f"{part.id} {part.name}"] = p
        for k in totals:
            totals[k] += p[k]
    report["totals"] = totals
    return report
