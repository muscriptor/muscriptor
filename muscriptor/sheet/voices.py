"""Staves and voices: lay quantized notes out the way a part is written.

Per instrument role ("part compilers" in the research document):

* keyboards, string sections, synth pads → grand staff. Each onset group is
  split between the hands by minimizing a cost over split points (each hand
  stays within reach, near the middle-C boundary, and close to where it just
  was) rather than a fixed split at middle C.
* single-staff instruments → one staff, clef chosen from the instrument and,
  for voice, from the actual range sung.
* drums → one percussion staff, hands (cymbals, snare, toms; stems up) in
  voice 1 and feet (kick, pedal hi-hat; stems down) in voice 2.

Within each staff at most two voices are used. Voice 1 is a chord stream:
notes starting together form a chord, held until the next onset. A note held
well past the next onset (a sustained bass under moving chords, a melody note
over an accompaniment) goes to voice 2 when it is free, instead of being cut
short or written as a pile of ties.

The output keeps :class:`QuantizedNote` but with `staff`/`voice` set, and the
guarantee the notation stage relies on: inside one (staff, voice), notes
either start and end together (a chord) or do not overlap.
"""

from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction

from muscriptor.sheet.instruments import InstrumentSpec, drum_spec
from muscriptor.sheet.ir import Clef, QuantizedNote

logger = logging.getLogger(__name__)

CLEFS = {
    "treble": [Clef("G", 2)],
    "treble_8vb": [Clef("G", 2, -1)],
    "bass": [Clef("F", 4)],
    "bass_8vb": [Clef("F", 4)],
    "alto": [Clef("C", 3)],
    "grand": [Clef("G", 2), Clef("F", 4)],
    "percussion": [Clef("percussion", 2)],
}

# Octaves between written and sounding pitch: guitars and basses are written
# an octave above where they sound (MusicXML <transpose> octave-change -1).
WRITTEN_OCTAVE_SHIFT = {"treble_8vb": 1, "bass_8vb": 1}


@dataclass
class StaffLayout:
    layout: str
    clefs: list[Clef]
    written_octave_shift: int


def choose_layout(spec: InstrumentSpec, notes: list[QuantizedNote]) -> StaffLayout:
    """The staff layout for a part, adjusted to the notes actually played."""
    layout = spec.layout
    if notes and spec.role == "voice":
        pitches = sorted(n.pitch for n in notes)
        median = pitches[len(pitches) // 2]
        if median < 57:  # a low (male) voice reads from the tenor clef
            layout = "treble_8vb"
    if (
        notes
        and layout == "grand"
        and spec.role in ("string_section", "synth", "other")
    ):
        # A "grand staff" part that never goes below middle C (or never above)
        # only needs one staff.
        pitches = [n.pitch for n in notes]
        if min(pitches) >= 57:
            layout = "treble"
        elif max(pitches) <= 64:
            layout = "bass"
    return StaffLayout(layout, CLEFS[layout], WRITTEN_OCTAVE_SHIFT.get(layout, 0))


# ---------------------------------------------------------------------------
# Hand / staff split for grand staves


def _onset_groups(notes: list[QuantizedNote]) -> list[list[QuantizedNote]]:
    groups: dict[Fraction, list[QuantizedNote]] = defaultdict(list)
    for n in notes:
        groups[n.onset].append(n)
    return [sorted(groups[t], key=lambda n: n.pitch) for t in sorted(groups)]


def split_hands(notes: list[QuantizedNote]) -> list[QuantizedNote]:
    """Assign each note to staff 1 (right hand) or 2 (left hand)."""
    rh_center, lh_center = 67.0, 50.0
    out = []
    for group in _onset_groups(notes):
        pitches = [n.pitch for n in group]
        best_k, best_cost = 0, float("inf")
        for k in range(len(pitches) + 1):
            lh, rh = pitches[:k], pitches[k:]
            cost = 0.0
            cost += sum(0.6 * max(0, p - 64) + (0.4 if p >= 60 else 0) for p in lh)
            cost += sum(0.6 * max(0, 55 - p) + (0.4 if p < 60 else 0) for p in rh)
            if lh:
                cost += 3.0 * max(0, lh[-1] - lh[0] - 14)
                cost += 0.08 * abs(sum(lh) / len(lh) - lh_center)
            if rh:
                cost += 3.0 * max(0, rh[-1] - rh[0] - 14)
                cost += 0.08 * abs(sum(rh) / len(rh) - rh_center)
            if cost < best_cost:
                best_k, best_cost = k, cost
        for i, n in enumerate(group):
            out.append(dataclasses.replace(n, staff=2 if i < best_k else 1))
        lh, rh = pitches[:best_k], pitches[best_k:]
        if lh:
            lh_center = 0.7 * lh_center + 0.3 * (sum(lh) / len(lh))
        if rh:
            rh_center = 0.7 * rh_center + 0.3 * (sum(rh) / len(rh))
    return out


# ---------------------------------------------------------------------------
# Voices within a staff


def assign_voices(
    notes: list[QuantizedNote], max_voices: int = 2, long_hold: Fraction = Fraction(1)
) -> list[QuantizedNote]:
    """Two-voice layout of one staff's notes (see module docstring)."""
    groups = _onset_groups(notes)
    out: list[QuantizedNote] = []
    v2_end = Fraction(-1)
    for i, group in enumerate(groups):
        onset = group[0].onset
        next_onset = groups[i + 1][0].onset if i + 1 < len(groups) else None

        held, rest = [], []
        for n in group:
            if (
                max_voices >= 2
                and next_onset is not None
                and n.offset - next_onset >= long_hold
                and v2_end <= onset
            ):
                held.append(n)
            else:
                rest.append(n)
        # Voice 2 only for the outer notes of the group: a held bass note
        # under the chord, or a held top note over it. A held inner note is
        # cut instead, which reads far better than crossing voices.
        if held and rest:
            lowest, highest = (
                min(rest, key=lambda n: n.pitch),
                max(rest, key=lambda n: n.pitch),
            )
            ok = [n for n in held if n.pitch < lowest.pitch] or [
                n for n in held if n.pitch > highest.pitch
            ]
            rest += [n for n in held if n not in ok]
            held = ok
        if held:
            end = min(n.offset for n in held)
            for n in held:
                out.append(dataclasses.replace(n, offset=end, voice=2))
            v2_end = end

        if rest:
            # One duration for the chord: the most common offset, never past
            # the next onset in this voice.
            offsets = sorted(n.offset for n in rest)
            end = offsets[len(offsets) // 2]
            if next_onset is not None:
                end = min(end, next_onset)
            if end <= onset:
                end = next_onset if next_onset is not None else onset + Fraction(1, 4)
            for n in rest:
                out.append(dataclasses.replace(n, offset=end, voice=1))
    return out


def layout_drums(notes: list[QuantizedNote]) -> list[QuantizedNote]:
    """Drum hits into hands (voice 1) and feet (voice 2), with readable lengths.

    A hit has no length of its own; it is written lasting until the next hit
    in its voice, but never across a beat (so drum parts need no ties) and
    never longer than a beat.
    """
    by_voice: dict[int, list[QuantizedNote]] = defaultdict(list)
    for n in notes:
        key, spec = drum_spec(n.pitch)
        by_voice[spec.voice].append(
            dataclasses.replace(n, pitch=key, voice=spec.voice, staff=1)
        )
    out = []
    for voice, group in by_voice.items():
        onsets = sorted({n.onset for n in group})
        nxt = {a: b for a, b in zip(onsets, onsets[1:])}
        for n in group:
            beat_end = Fraction(int(n.onset) + 1)
            end = min(nxt.get(n.onset, beat_end), beat_end)
            if end <= n.onset:
                end = n.onset + Fraction(1, 4)
            out.append(dataclasses.replace(n, offset=end))
    # Same drum twice at one onset: keep one.
    seen = set()
    unique = []
    for n in sorted(out, key=lambda n: (n.onset, n.voice, n.pitch)):
        key = (n.onset, n.pitch)
        if key not in seen:
            seen.add(key)
            unique.append(n)
    return unique


def layout_part(
    spec: InstrumentSpec, notes: list[QuantizedNote]
) -> tuple[StaffLayout, list[QuantizedNote]]:
    """Staff layout and staff/voice-assigned notes for one part."""
    layout = choose_layout(spec, notes)
    if spec.role == "drums":
        return layout, layout_drums(notes)
    if len(layout.clefs) == 2:
        notes = split_hands(notes)
    else:
        notes = [dataclasses.replace(n, staff=1) for n in notes]
    max_voices = 1 if spec.monophonic else 2
    out = []
    for staff in sorted({n.staff for n in notes}):
        out.extend(assign_voices([n for n in notes if n.staff == staff], max_voices))
    return layout, sorted(out, key=lambda n: (n.onset, n.staff, n.voice, n.pitch))
