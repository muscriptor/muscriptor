"""
instruments
===========

Instrument-level transcription-quality analysis for multitrack AMT MIDI
files (e.g. the output of Google's MT3 or any similar automatic-music-
transcription system that emits one MIDI file containing many, possibly
wrong, instruments).

The public function :func:`analyze_midi_instruments`

1. parses the MIDI file (path, bytes or ``mido.MidiFile``),
2. groups every note into one of the instrument buckets defined in
   ``MT3_FULL_PLUS_GROUP_NAMES`` (General-MIDI program -> bucket; anything
   played on MIDI channel 10 -> ``drums``),
3. runs a battery of musical / statistical sanity checks on every bucket,
4. returns ONE flat list - one ``(instrument, quality_score, note_count)``
   tuple per bucket found, sorted best-first, each with a continuous
   quality score in ``[0.0, 1.0]``.

Quality signals used (see ``_score_bucket`` for the full model):

* **Evidence sufficiency** - a bucket with a handful of notes can never be
  verified statistically, so its score is capped (``_EVIDENCE_CAPS``).
* **Duration hygiene** - spurious micro-notes (classic AMT jitter) and
  stuck notes (note_on never closed).
* **Pitch-range plausibility** - e.g. contrabass notes above C5 are almost
  certainly leakage from another instrument.
* **Polyphony plausibility** - a flute "playing" 5-note chords is a
  transcription artifact; ensembles/pianos are allowed much higher
  polyphony.
* **Monophonic-line sanity** - machine-gun re-articulations and
  same-pitch retriggers that are impossible on wind/mono instruments.
* **Same-pitch overlaps** - retriggered note_on without note_off.
* **Temporal density** - sparse "one note every 10 s" noise vs. real
  material, and absurd hyper-density.
* **Cross-bucket leakage** - notes that are exact twins (same pitch, same
  instant) of another bucket's notes.  MT3 frequently splits one real
  instrument into two programs; the smaller "echo" bucket is penalized,
  and near-identical mutual duplicates are penalized on both sides.
* **Note pairing integrity** - unmatched note_offs / truncated notes.
* **Onset-gap (dropout) detection** - long silent holes inside an
  instrument's active window (the classic MT3 dropout, where the model
  suddenly stops emitting an instrument for several seconds) are measured
  against the instrument's own median onset spacing and penalised in
  proportion to the share of the active window they waste.
* **Duration-outlier detection** - notes whose duration wildly exceeds
  the bucket's median duration (stuck / inflated notes that escape the
  per-instrument ceiling) are flagged statistically.
* **Chord sanity** - groups of simultaneous onsets ("chords") are checked
  for chromatic clusters (runs of >= 3 consecutive semitones), unison
  double-triggers of the same pitch, and physically impossible chord
  sizes (e.g. a 7-note "chord" on a flute).

Only dependency: ``mido`` (plus the Python standard library).

Notes & caveats
---------------
* Only buckets that actually contain notes are reported.  Buckets absent
  from the file are listed in ``details["absent_buckets"]``.
* Velocity is deliberately *not* scored: MT3-style output carries almost
  no meaningful velocity information.
* The function returns ONE undivided list (best instrument first) - apply
  your own cutoff.  ``good_threshold`` (default 0.70) is only used to
  annotate ``details["classification"]`` when ``return_details=True``.
  Small buckets are capped by design (an unverifiable instrument must not
  be trusted); tune ``_EVIDENCE_CAPS`` if you disagree.

Examples
--------
>>> analyze_midi_instruments("song.mid")
[('drums', 0.99, 812), ('acoustic_piano', 0.87, 431), ('voice', 0.04, 57)]

>>> analyze_midi_instruments("song.mid", use_bucket_numbers=True)
[(36, 0.99, 812), (0, 0.87, 431), (17, 0.04, 57)]

>>> results, details = analyze_midi_instruments("song.mid", return_details=True)
>>> details["instruments"]["flutes"]["flags"]
['bad_chords:4/4 simultaneous-onset chords flagged (chromatic_cluster:2, oversize:1, unison_double:1)']

License
-------
Apache 2.0

Attribution
-----------
Coded exclusively for MuScriptor project by
Project Los Angeles
Tegridy Code 2026
"""

from __future__ import annotations

import bisect
import io
import math
import os
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import mido

__all__ = ["analyze_midi_instruments", "MT3_FULL_PLUS_GROUP_NAMES"]

# ---------------------------------------------------------------------------
# 1. Target instrument taxonomy (verbatim) + General MIDI program mapping
# ---------------------------------------------------------------------------

MT3_FULL_PLUS_GROUP_NAMES: Dict[str, int] = {
    "acoustic_piano": 0,
    "electric_piano": 1,
    "chromatic_percussion": 2,
    "organ": 3,
    "acoustic_guitar": 4,
    "clean_electric_guitar": 5,
    "distorted_electric_guitar": 6,
    "acoustic_bass": 7,
    "electric_bass": 8,
    "violin": 9,
    "viola": 10,
    "cello": 11,
    "contrabass": 12,
    "orchestral_harp": 13,
    "timpani": 14,
    "string_ensemble": 15,
    "synth_strings": 16,
    "voice": 17,
    "orchestra_hit": 18,
    "trumpet": 19,
    "trombone": 20,
    "tuba": 21,
    "french_horn": 22,
    "brass_section": 23,
    "soprano_and_alto_sax": 24,
    "tenor_sax": 25,
    "baritone_sax": 26,
    "oboe": 27,
    "english_horn": 28,
    "bassoon": 29,
    "clarinet": 30,
    "flutes": 31,
    "synth_lead": 32,
    "synth_pad": 33,
    "drums": 36,
}

# GM program (0-127, 0-based) -> bucket name.  The first block mirrors the
# classic Slakh/MT3 program grouping; the remainder is a deliberate
# catch-all so that *any* well-formed MIDI file can be analyzed (unusual
# programs are folded into their closest musical family and reported in
# ``details["parsing"]["fallback_programs"]``).


def _build_program_map() -> Dict[int, str]:
    spec: Dict[str, Tuple[int, ...]] = {
        "acoustic_piano": (0, 1, 3, 6, 7),
        "electric_piano": (2, 4, 5),
        "chromatic_percussion": tuple(range(8, 16)) + tuple(range(112, 120)),
        "organ": tuple(range(16, 24)),
        "acoustic_guitar": (24, 25),
        "clean_electric_guitar": (26, 27, 31),
        "distorted_electric_guitar": (28, 29, 30),
        "acoustic_bass": (32,),
        "electric_bass": tuple(range(33, 40)),
        "violin": (40,),
        "viola": (41,),
        "cello": (42,),
        "contrabass": (43,),
        "string_ensemble": (44, 45, 48, 49),
        "orchestral_harp": (46,),
        "timpani": (47,),
        "synth_strings": (50, 51),
        "voice": (52, 53, 54),
        "orchestra_hit": (55,),
        "trumpet": (56, 59),
        "trombone": (57,),
        "tuba": (58,),
        "french_horn": (60,),
        "brass_section": (61, 62, 63),
        "soprano_and_alto_sax": (64, 65),
        "tenor_sax": (66,),
        "baritone_sax": (67,),
        "oboe": (68,),
        "english_horn": (69,),
        "bassoon": (70,),
        "clarinet": (71,),
        "flutes": tuple(range(72, 80)),
        "synth_lead": tuple(range(80, 88)) + tuple(range(104, 112)),
        "synth_pad": tuple(range(88, 96)) + tuple(range(96, 104)) + tuple(range(120, 128)),
    }
    mapping: Dict[int, str] = {}
    for bucket, programs in spec.items():
        for program in programs:
            mapping[program] = bucket
    return mapping


_GM_PROGRAM_TO_BUCKET: Dict[int, str] = _build_program_map()


# ---------------------------------------------------------------------------
# 2. Per-instrument physical / musical profiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _InstrumentProfile:
    soft_poly: int                 # simultaneous notes considered normal up to here
    hard_poly: int                 # beyond this = physically impossible
    pitch_lo: Optional[int]        # inclusive playable MIDI range (None = full)
    pitch_hi: Optional[int]
    min_density: float             # notes per active second below which the
                                   # bucket looks like sparse noise
    max_note_dur: Optional[float]  # seconds; longer notes are "stuck" (None = n/a)


_INSTRUMENT_PROFILES: Dict[str, _InstrumentProfile] = {
    "acoustic_piano": _InstrumentProfile(10, 14, 21, 108, 0.5, 20.0),
    "electric_piano": _InstrumentProfile(10, 14, 21, 108, 0.4, 20.0),
    "chromatic_percussion": _InstrumentProfile(6, 10, 33, 100, 0.15, 30.0),
    "organ": _InstrumentProfile(10, 16, 21, 108, 0.5, None),
    "acoustic_guitar": _InstrumentProfile(6, 8, 40, 90, 0.8, 12.0),
    "clean_electric_guitar": _InstrumentProfile(6, 10, 40, 98, 0.8, 15.0),
    "distorted_electric_guitar": _InstrumentProfile(6, 10, 26, 98, 0.8, 15.0),
    "acoustic_bass": _InstrumentProfile(2, 4, 27, 72, 0.6, 12.0),
    "electric_bass": _InstrumentProfile(2, 4, 23, 75, 0.6, 12.0),
    "violin": _InstrumentProfile(2, 4, 55, 107, 1.0, 30.0),
    "viola": _InstrumentProfile(2, 4, 48, 102, 1.0, 30.0),
    "cello": _InstrumentProfile(2, 4, 36, 90, 0.9, 30.0),
    "contrabass": _InstrumentProfile(2, 3, 27, 76, 0.7, 30.0),
    "orchestral_harp": _InstrumentProfile(8, 14, 23, 105, 1.2, 12.0),
    "timpani": _InstrumentProfile(4, 6, 35, 77, 0.5, 20.0),
    "string_ensemble": _InstrumentProfile(12, 20, 28, 108, 0.8, None),
    "synth_strings": _InstrumentProfile(12, 20, 24, 108, 0.5, None),
    "voice": _InstrumentProfile(6, 12, 36, 88, 0.5, 25.0),
    "orchestra_hit": _InstrumentProfile(8, 16, 24, 108, 0.3, 8.0),
    "trumpet": _InstrumentProfile(2, 3, 51, 97, 0.8, 20.0),
    "trombone": _InstrumentProfile(2, 3, 39, 86, 0.7, 20.0),
    "tuba": _InstrumentProfile(1, 2, 25, 68, 0.5, 20.0),
    "french_horn": _InstrumentProfile(2, 3, 40, 90, 0.6, 25.0),
    "brass_section": _InstrumentProfile(6, 12, 36, 98, 0.7, 25.0),
    "soprano_and_alto_sax": _InstrumentProfile(1, 2, 50, 97, 0.8, 20.0),
    "tenor_sax": _InstrumentProfile(1, 2, 43, 92, 0.8, 20.0),
    "baritone_sax": _InstrumentProfile(1, 2, 36, 85, 0.7, 20.0),
    "oboe": _InstrumentProfile(1, 2, 57, 97, 0.8, 20.0),
    "english_horn": _InstrumentProfile(1, 2, 49, 92, 0.7, 20.0),
    "bassoon": _InstrumentProfile(1, 2, 33, 90, 0.7, 20.0),
    "clarinet": _InstrumentProfile(1, 2, 48, 99, 0.8, 20.0),
    "flutes": _InstrumentProfile(1, 2, 54, 109, 0.8, 25.0),
    "synth_lead": _InstrumentProfile(6, 12, 24, 108, 0.8, None),
    "synth_pad": _InstrumentProfile(10, 20, 24, 108, 0.25, None),
    "drums": _InstrumentProfile(6, 12, 27, 88, 1.0, 4.0),
}

_DEFAULT_PROFILE = _InstrumentProfile(8, 14, 21, 108, 0.5, 30.0)

# Physically plausible maximum number of simultaneous onsets (chord size).
# Defaults to the profile's soft polyphony (>= 2); bowed strings get an
# explicit allowance for triple stops.
_MAX_CHORD_OVERRIDES = {"violin": 3, "viola": 3, "cello": 3, "contrabass": 3}


def _max_chord_for(bucket: str, prof: _InstrumentProfile) -> int:
    if bucket == "drums":
        return 16  # unused - drums skip the chord-size check entirely
    return _MAX_CHORD_OVERRIDES.get(bucket, max(2, prof.soft_poly))


# ---------------------------------------------------------------------------
# 3. Tuning constants
# ---------------------------------------------------------------------------

_SHORT_HARD_S = 0.040   # note shorter than this (40 ms) = spurious micro-note
_SHORT_SOFT_S = 0.100   # 40-100 ms = suspicious on sustained instruments
_FAST_IOI_S = 0.055     # re-articulation gap smaller than this = machine-gun
_TWIN_EPS_S = 0.030     # |dt| tolerance for cross-bucket twin notes
_SAME_PITCH_EPS_S = 1e-4
_MIN_UNION_S = 0.75     # less total sounding time than this => low evidence
_MIN_UNION_CAP = 0.50

# Onset-gap ("dropout") detection.
_GAP_ABS_MIN_S = 2.0      # silences shorter than this are never "excessive"
_GAP_IOI_RATIO = 8.0      # a hole must also exceed 8x the median onset interval
_GAP_FACTOR_WEIGHT = 10.0  # scaling of hole-fraction inside f_gap

# Duration-outlier detection (statistical, long side).
_DUR_OUTLIER_X = 12.0     # duration > 12x the median duration = outlier...
_DUR_OUTLIER_MIN_S = 1.0  # ...and at least 1 s long (absolute floor)
_DUR_CV_FLAG = 2.5        # duration CV above this = wildly inconsistent

# Chord sanity (simultaneous-onset groups).
_CHORD_TOL_S = 0.1      # onsets within 45 ms land in the same chord (AMT
                          # output is frame-quantised)
_CHORD_CLUSTER_RUN = 3    # >= 3 consecutive semitones = chromatic cluster

# (note_count_below -> max score) - statistical confidence in tiny buckets.
_EVIDENCE_CAPS: Tuple[Tuple[int, float], ...] = (
    (4, 0.45), (8, 0.58), (16, 0.72), (32, 0.82), (64, 0.90),
)


# ---------------------------------------------------------------------------
# 4. MIDI parsing (mido) - tempo map, note pairing, bucket assignment
# ---------------------------------------------------------------------------

# GM programs that map 1:1 onto a canonical Slakh/MT3 instrument class.
# Anything else reaching a bucket is reported as a fallback assignment.
_CORE_SLAKH_PROGRAMS = frozenset(
    list(range(0, 2)) + [4, 5] + [8, 9, 11, 12, 13, 14]
    + list(range(16, 24)) + [24, 25, 26, 27, 28, 29, 30]
    + [32, 33, 34, 35] + list(range(40, 44)) + [46, 47]
    + [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 64, 66, 67, 68, 69, 70, 71]
    + list(range(73, 80)) + list(range(80, 88)) + list(range(88, 96)) + [128]
)


def _parse_midi(source: Union[str, "os.PathLike[str]", bytes, bytearray, mido.MidiFile]) -> mido.MidiFile:
    """Accept a path, raw bytes, or an already-parsed mido.MidiFile."""
    if isinstance(source, mido.MidiFile):
        return source
    try:
        if isinstance(source, (bytes, bytearray, memoryview)):
            return mido.MidiFile(file=io.BytesIO(bytes(source)))
        if isinstance(source, (str, os.PathLike)):
            return mido.MidiFile(filename=os.fspath(source))
    except Exception as exc:  # mido raises OSError / EOFError / ValueError / KeyError ...
        raise ValueError(f"analyze_midi_instruments: could not parse MIDI input ({exc})") from exc
    raise TypeError(
        f"analyze_midi_instruments: unsupported MIDI source type "
        f"{type(source).__name__} (expected path, bytes or mido.MidiFile)"
    )


def _make_tick_to_sec_fn(mf: mido.MidiFile):
    """Build a tick->seconds converter honoring the file's tempo map.

    Returns (fn, info_dict).  Falls back to SMPTE interpretation when
    ticks_per_beat is negative, and to 120 bpm when no tempo is given.
    """
    tpb = mf.ticks_per_beat or 480
    if tpb < 0:  # SMPTE time
        fps = -(tpb >> 8) or 30
        subframes = (tpb & 0xFF) or 4
        sec_per_tick = 1.0 / (float(fps) * subframes)
        info = {"ticks_per_beat": tpb, "smpte": True, "tempo_events": 0}

        def t2s_smpte(q: int) -> float:
            return q * sec_per_tick

        return t2s_smpte, info

    tempo_at_tick: Dict[int, int] = {}
    for track in mf.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo" and msg.tempo > 0:
                tempo_at_tick[tick] = msg.tempo

    seg_ticks = [0]
    seg_tempos = [500000]  # default 120 bpm
    for t in sorted(tempo_at_tick):
        if t == seg_ticks[-1]:
            seg_tempos[-1] = tempo_at_tick[t]
        else:
            seg_ticks.append(t)
            seg_tempos.append(tempo_at_tick[t])

    prefix = [0.0]
    for i in range(len(seg_ticks)):
        nxt = seg_ticks[i + 1] if i + 1 < len(seg_ticks) else seg_ticks[i]
        prefix.append(prefix[-1] + (nxt - seg_ticks[i]) * seg_tempos[i] / (1_000_000.0 * tpb))

    info = {"ticks_per_beat": tpb, "smpte": False, "tempo_events": len(seg_ticks) - 1}

    def t2s(q: int) -> float:
        i = bisect.bisect_right(seg_ticks, q) - 1
        return prefix[i] + (q - seg_ticks[i]) * seg_tempos[i] / (1_000_000.0 * tpb)

    return t2s, info


def _extract_notes(mf: mido.MidiFile):
    """Collect (start, end, pitch, velocity) notes, grouped by bucket.

    Returns (buckets, piece_duration_s, parse_diagnostics).
    """
    t2s, tick_info = _make_tick_to_sec_fn(mf)
    program_of_channel: Dict[int, int] = defaultdict(int)
    pending: Dict[Tuple[int, int], deque] = defaultdict(deque)  # (ch, pitch) -> open note_ons
    buckets: Dict[str, List[Tuple[float, float, int, int]]] = defaultdict(list)
    programs_by_bucket: Dict[str, set] = defaultdict(set)
    truncated_by_bucket: Dict[str, int] = defaultdict(int)
    unmatched_note_offs = 0
    fallback_programs = set()
    max_end = 0.0

    for track in mf.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            mtype = msg.type
            if mtype == "note_on" and msg.velocity > 0:
                ch, pitch = msg.channel, msg.note
                if ch == 9:
                    bucket = "drums"
                    programs_by_bucket[bucket].add(128)
                else:
                    prog = program_of_channel[ch] & 0x7F
                    bucket = _GM_PROGRAM_TO_BUCKET[prog]
                    if prog not in _CORE_SLAKH_PROGRAMS:
                        fallback_programs.add(prog)
                    programs_by_bucket[bucket].add(prog)
                pending[(ch, pitch)].append((tick, msg.velocity, bucket, pitch))
            elif mtype == "note_off" or (mtype == "note_on" and msg.velocity == 0):
                q = pending.get((msg.channel, msg.note))
                if q:
                    start_tick, vel, bucket, pitch = q.popleft()
                    start, end = t2s(start_tick), t2s(tick)
                    if end < start:
                        end = start
                    buckets[bucket].append((start, end, pitch, vel))
                    if end > max_end:
                        max_end = end
                else:
                    unmatched_note_offs += 1
            elif mtype == "program_change":
                if msg.channel != 9:
                    program_of_channel[msg.channel] = msg.program
            # every other message type is irrelevant here

    # Close notes that never received a note_off at the last observed note end.
    for q in list(pending.values()):
        while q:
            start_tick, vel, bucket, pitch = q.popleft()
            start = t2s(start_tick)
            end = max(start, max_end)
            buckets[bucket].append((start, end, pitch, vel))
            truncated_by_bucket[bucket] += 1

    diag = {
        "unmatched_note_offs": unmatched_note_offs,
        "truncated_by_bucket": dict(truncated_by_bucket),
        "programs_by_bucket": {k: set(v) for k, v in programs_by_bucket.items()},
        "fallback_programs": sorted(fallback_programs),
        "tick_info": tick_info,
        "midi_type": getattr(mf, "type", None),
    }
    return dict(buckets), max_end, diag


# ---------------------------------------------------------------------------
# 5. Low-level note metrics
# ---------------------------------------------------------------------------

# Notes are (start_s, end_s, pitch, velocity) tuples throughout.


def _union_duration(notes: Sequence[tuple]) -> float:
    """Total sounding time (union of note intervals) in seconds."""
    if not notes:
        return 0.0
    ivs = sorted((s, e) for s, e, _p, _v in notes)
    total = 0.0
    cur_s, cur_e = ivs[0]
    for s, e in ivs[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        elif e > cur_e:
            cur_e = e
    return total + (cur_e - cur_s)


def _polyphony_stats(notes: Sequence[tuple], soft: int, hard: int):
    """Sweep-line over note events.

    Returns (soft_excess_rate, hard_violation_rate, n_onsets):
    fraction of note onsets that, counting all notes already sounding,
    land above ``soft`` (musically unusual) or above ``hard`` (physically
    impossible) polyphony.
    """
    events: List[Tuple[float, int]] = []
    for s, e, _p, _v in notes:
        if e - s <= 1e-9:  # zero-length notes distort the sweep; they are
            continue       # already penalised as micro-notes
        events.append((s, 1))
        events.append((e, -1))
    if not events:
        return 0.0, 0.0, 0
    events.sort(key=lambda ev: (ev[0], ev[1]))  # note_offs before note_ons
    sounding = 0
    onsets = soft_excess = hard_violation = 0
    for _t, delta in events:
        if delta == 1:
            onsets += 1
            sounding += 1
            if sounding > hard:
                hard_violation += 1
            elif sounding > soft:
                soft_excess += 1
        elif sounding > 0:
            sounding -= 1
    denom = max(1, onsets)
    return soft_excess / denom, hard_violation / denom, onsets


def _same_pitch_overlap_rate(notes: Sequence[tuple]) -> float:
    """Fraction of notes starting while an earlier same-pitch note still sounds."""
    by_pitch: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    for s, e, p, _v in notes:
        by_pitch[p].append((s, e))
    total = violations = 0
    for ivs in by_pitch.values():
        ivs.sort()
        prev_end = -1.0
        for s, e in ivs:
            total += 1
            if s < prev_end - _SAME_PITCH_EPS_S:
                violations += 1
            if e > prev_end:
                prev_end = e
    return violations / total if total else 0.0


def _fast_ioi_rate(notes: Sequence[tuple]) -> float:
    """Fraction of positive gaps between consecutive notes below _FAST_IOI_S.

    Only real gaps (previous note ended before the new onset) count;
    overlaps are measured by the polyphony / same-pitch metrics instead.
    """
    if len(notes) < 2:
        return 0.0
    ordered = sorted(notes, key=lambda n: n[0])
    gaps = fast = 0
    prev_end = ordered[0][1]
    for s, e, _p, _v in ordered[1:]:
        gap = s - prev_end
        if gap > _SAME_PITCH_EPS_S:
            gaps += 1
            if gap < _FAST_IOI_S:
                fast += 1
        if e > prev_end:
            prev_end = e
    return fast / gaps if gaps else 0.0


def _evidence_cap(count: int, union_s: float) -> float:
    """Statistical confidence ceiling for buckets with little material."""
    cap = 1.0
    for threshold, value in _EVIDENCE_CAPS:
        if count < threshold:
            cap = value
            break
    if union_s < _MIN_UNION_S:
        cap = min(cap, _MIN_UNION_CAP)
    return cap




def _onset_gap_stats(notes: Sequence[tuple]) -> Tuple[float, float, int, float]:
    """Detect silent holes inside an instrument's active window.

    A *hole* is a stretch of silence (no note of this bucket sounding)
    that lasts at least ``_GAP_ABS_MIN_S`` seconds AND at least
    ``_GAP_IOI_RATIO`` times the bucket's median onset-to-onset interval,
    lying between the bucket's first and last onset.  Instrument entrances
    and exits are natural and are never counted.

    Returns ``(hole_fraction, longest_hole_s, n_holes, median_ioi_s)``
    where ``hole_fraction`` is the share of the active window wasted on
    excessive holes (0.0 = no suspicious gap at all).
    """
    onsets = sorted(s for s, _e, _p, _v in notes)
    if len(onsets) < 2:
        return 0.0, 0.0, 0, 0.0
    iois = [b - a for a, b in zip(onsets, onsets[1:]) if b > a]
    med_ioi = statistics.median(iois) if iois else 0.0
    span = onsets[-1] - onsets[0]
    if span <= 1e-9:
        return 0.0, 0.0, 0, med_ioi
    thr = max(_GAP_ABS_MIN_S, _GAP_IOI_RATIO * med_ioi)

    ivs = sorted((s, e) for s, e, _p, _v in notes)
    holes: List[float] = []
    cur_e = ivs[0][1]
    for s, e in ivs[1:]:
        if s > cur_e:
            hole = s - cur_e
            if hole >= thr:
                holes.append(hole)
        if e > cur_e:
            cur_e = e
    if not holes:
        return 0.0, 0.0, 0, med_ioi
    return sum(holes) / span, max(holes), len(holes), med_ioi


def _duration_outlier_stats(durs: Sequence[float]) -> Tuple[float, float]:
    """Statistical outliers on the long side of the duration distribution.

    Returns ``(outlier_rate, cv)``: the fraction of notes whose duration
    exceeds both ``_DUR_OUTLIER_X`` times the median duration and
    ``_DUR_OUTLIER_MIN_S`` seconds (stuck / inflated notes that escape the
    per-instrument hard ceiling), plus the coefficient of variation of all
    durations (wildly inconsistent note lengths).
    """
    if not durs:
        return 0.0, 0.0
    med = statistics.median(durs)
    if med <= 1e-6:
        return 0.0, 0.0  # degenerate; the micro-note factor handles this
    n_out = sum(1 for d in durs if d > max(_DUR_OUTLIER_MIN_S, _DUR_OUTLIER_X * med))
    mean = sum(durs) / len(durs)
    cv = (statistics.pstdev(durs) / mean) if mean > 1e-9 else 0.0
    return n_out / len(durs), cv


def _chord_stats(notes: Sequence[tuple], max_chord: int, is_drums: bool):
    """Group simultaneous onsets into chords and sanity-check them.

    AMT output is frame-quantised, so genuinely simultaneous notes land on
    (almost) the same instant: notes whose onsets lie within
    ``_CHORD_TOL_S`` of the first member are grouped into one chord.
    Groups of >= 2 notes are checked for

    * ``unison_double`` - the same pitch twice at the same instant
      (double-triggered note);
    * ``chromatic_cluster`` - a run of >= ``_CHORD_CLUSTER_RUN``
      consecutive semitones (e.g. 60-61-62), a classic mistranscription
      artifact;
    * ``oversize`` - more than ``max_chord`` simultaneous onsets, which is
      physically implausible for the instrument (skipped for drums).

    Returns ``(n_chords, n_flagged, flagged_rate, reason_counts)``.
    """
    if len(notes) < 2:
        return 0, 0, 0.0, {}
    ordered = sorted(notes, key=lambda nt: nt[0])
    groups: List[List[tuple]] = []
    current: List[tuple] = [ordered[0]]
    for nt in ordered[1:]:
        if nt[0] - current[0][0] <= _CHORD_TOL_S:
            current.append(nt)
        else:
            if len(current) >= 2:
                groups.append(current)
            current = [nt]
    if len(current) >= 2:
        groups.append(current)

    n_flagged = 0
    reasons: Dict[str, int] = defaultdict(int)
    for group in groups:
        pitches = sorted(p for _s, _e, p, _v in group)
        reason = None
        if len(pitches) != len(set(pitches)):
            reason = "unison_double"
        elif not is_drums:
            run = best = 1
            for a, b in zip(pitches, pitches[1:]):
                run = run + 1 if b - a <= 1 else 1
                best = max(best, run)
            if best >= _CHORD_CLUSTER_RUN:
                reason = "chromatic_cluster"
            elif len(pitches) > max_chord:
                reason = "oversize"
        if reason is not None:
            n_flagged += 1
            reasons[reason] += 1
    n_chords = len(groups)
    rate = n_flagged / n_chords if n_chords else 0.0
    return n_chords, n_flagged, rate, dict(reasons)

# ---------------------------------------------------------------------------
# 6. Cross-bucket leakage / split-instrument detection
# ---------------------------------------------------------------------------


def _twin_fraction(bucket_notes: Sequence[tuple], other_onsets_by_pitch: Dict[int, List[float]]) -> float:
    """Fraction of ``bucket_notes`` that are exact twins of notes in another
    bucket (same pitch, onset within ``_TWIN_EPS_S``).  AMT models emit
    frame-aligned events, so a note duplicated by the model lands on the
    very same instant - this is the signature of MT3 "split" leakage."""
    total = twins = 0
    for s, _e, p, _v in bucket_notes:
        total += 1
        onsets = other_onsets_by_pitch.get(p)
        if not onsets:
            continue
        i = bisect.bisect_left(onsets, s - _TWIN_EPS_S)
        if i < len(onsets) and onsets[i] <= s + _TWIN_EPS_S:
            twins += 1
    return twins / total if total else 0.0


def _onsets_by_pitch(notes: Sequence[tuple]) -> Dict[int, List[float]]:
    d: Dict[int, List[float]] = defaultdict(list)
    for s, _e, p, _v in notes:
        d[p].append(s)
    for onsets in d.values():
        onsets.sort()
    return dict(d)


def _leakage_analysis(buckets: Dict[str, List[tuple]]):
    """Compare every bucket pair; penalise buckets that duplicate another.

    Returns (leak_factor per bucket in (0,1], leak_flags per bucket).

    * asymmetric leak  - the smaller bucket largely duplicates the bigger
      one  =>  the small bucket is an "echo" and heavily penalised;
    * mutual duplicate - both buckets are near-identical (MT3 could not
      decide which program an instrument belongs to) => both moderately
      penalised: the material is probably real but the instrument
      identity is unreliable.
    """
    names = [n for n, ns in buckets.items() if ns]
    counts = {n: len(buckets[n]) for n in names}
    onsets = {n: _onsets_by_pitch(buckets[n]) for n in names}
    leak_factor = {n: 1.0 for n in names}
    flags: Dict[str, List[str]] = {n: [] for n in names}

    def penalize(bucket: str, value: float, flag: str) -> None:
        if value < leak_factor[bucket]:
            leak_factor[bucket] = value
        if flag not in flags[bucket]:
            flags[bucket].append(flag)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            small, large = (a, b) if counts[a] <= counts[b] else (b, a)
            c_small, c_large = counts[small], counts[large]
            ratio = c_small / c_large if c_large else 1.0

            tf_small = _twin_fraction(buckets[small], onsets[large])
            tf_large = _twin_fraction(buckets[large], onsets[small]) if tf_small >= 0.5 else 0.0
            if c_small < 4:
                continue  # too little evidence; the evidence cap handles it

            if tf_small >= 0.60 and ratio <= 0.50 and small != "drums":
                penalize(small, max(0.05, 1.0 - 1.4 * tf_small),
                         f"leak_from:{large} ({tf_small:.0%} of notes duplicated)")
            elif tf_small >= 0.85 and ratio <= 0.90 and tf_large < 0.80:
                penalize(small, max(0.10, 1.0 - 1.2 * tf_small),
                         f"mostly_duplicated_by:{large} ({tf_small:.0%})")
            if small == "drums" and c_small >= 8 and tf_small >= 0.85 and tf_large < 0.80:
                penalize(small, max(0.10, 1.0 - 1.2 * tf_small),
                         f"mostly_duplicated_by:{large} ({tf_small:.0%})")
            if tf_small >= 0.80 and tf_large >= 0.80:
                penalize(small, 0.40, f"likely_split_duplicate_of:{large}")
                penalize(large, 0.40, f"likely_split_duplicate_of:{small}")

    return leak_factor, flags


# ---------------------------------------------------------------------------
# 7. Per-bucket scoring model
# ---------------------------------------------------------------------------


def _score_bucket(
    bucket: str,
    notes: List[Tuple[float, float, int, int]],
    piece_duration_s: float,
    leak_factor: float,
    leak_flags: Sequence[str],
    truncated: int,
) -> Tuple[float, Dict[str, Any]]:
    """Score one instrument bucket in [0, 1] and produce diagnostics.

    Model: a weighted geometric mean of "soft" consistency metrics
    (temporal shape of the material), multiplied by hard veto factors for
    unambiguous failure modes (micro-notes, impossible polyphony, wrong
    pitch range, broken monophonic lines, leakage from another bucket),
    and finally capped by the statistical evidence available.
    """
    is_drums = bucket == "drums"
    prof = _INSTRUMENT_PROFILES.get(bucket, _DEFAULT_PROFILE)
    n = len(notes)
    durs = [e - s for s, e, _p, _v in notes]
    med_dur = statistics.median(durs) if durs else 0.0
    flags: List[str] = []

    # --- hard factor 1: micro-notes (AMT jitter) --------------------------
    r_micro = sum(1 for d in durs if d < _SHORT_HARD_S) / n
    r_short = sum(1 for d in durs if _SHORT_HARD_S <= d < _SHORT_SOFT_S) / n
    if is_drums:
        f_short = 1.0 - min(1.0, r_micro * 0.5)  # percussive notes are short by nature
    else:
        soft_slope = 0.5 if med_dur > 0.12 else 0.25
        f_short = 1.0 - min(1.0, r_micro * 1.8 + r_short * soft_slope)
    if r_micro >= 0.05:
        flags.append(f"micro_notes:{r_micro:.0%} shorter than {_SHORT_HARD_S * 1000:.0f} ms")

    # --- hard factor 2: pitch-range plausibility ---------------------------
    if prof.pitch_lo is not None:
        r_range = sum(1 for _s, _e, p, _v in notes
                      if p < prof.pitch_lo or p > prof.pitch_hi) / n
        f_range = 1.0 - min(1.0, r_range * (1.0 if is_drums else 2.0))
        if r_range >= 0.03:
            flags.append(f"out_of_range:{r_range:.0%} outside MIDI {prof.pitch_lo}-{prof.pitch_hi}")
    else:
        r_range, f_range = 0.0, 1.0

    # --- hard factor 3: polyphony plausibility -----------------------------
    r_poly_soft, r_poly_hard, n_onsets = _polyphony_stats(notes, prof.soft_poly, prof.hard_poly)
    soft_slope_poly = 0.8 if prof.soft_poly <= 2 else 0.35
    f_poly = 1.0 - min(1.0, r_poly_hard * 1.8 + r_poly_soft * soft_slope_poly)
    if r_poly_hard > 0.30:
        f_poly *= 0.7
    if r_poly_soft >= 0.05 or r_poly_hard >= 0.02:
        flags.append(
            f"polyphony_violations: soft {r_poly_soft:.0%} / hard {r_poly_hard:.0%} "
            f"(limits {prof.soft_poly}/{prof.hard_poly})"
        )

    # --- hard factor 4: monophonic-line sanity -----------------------------
    r_overlap = _same_pitch_overlap_rate(notes)
    if prof.hard_poly <= 4 and not is_drums:
        r_fast = _fast_ioi_rate(notes)
        f_line = 1.0 - min(1.0, r_fast * 1.5 + r_overlap * 1.2)
        if r_fast >= 0.10 or r_overlap >= 0.05:
            flags.append(
                f"line_jitter: {r_fast:.0%} machine-gun re-articulations, "
                f"{r_overlap:.0%} same-pitch retriggers"
            )
    else:
        r_fast, f_line = 0.0, 1.0

    # --- hard factor 5: onset gaps / dropouts -------------------------------
    if n >= 4:
        hole_fraction, longest_hole, n_holes, med_ioi = _onset_gap_stats(notes)
    else:
        hole_fraction, longest_hole, n_holes, med_ioi = 0.0, 0.0, 0, 0.0
    f_gap = 1.0 - min(1.0, _GAP_FACTOR_WEIGHT * hole_fraction)
    if n_holes:
        flags.append(
            f"onset_gaps:{n_holes} silent hole(s) >= "
            f"{max(_GAP_ABS_MIN_S, _GAP_IOI_RATIO * med_ioi):.1f} s inside the "
            f"active window (longest {longest_hole:.1f} s, "
            f"{hole_fraction:.0%} of the span)"
        )

    # --- soft components (weighted geometric mean) -------------------------
    components: List[Tuple[float, float]] = []  # (score, weight)

    s_overlap = 1.0 - min(1.0, r_overlap * (0.4 if is_drums else 1.2))
    components.append((s_overlap, 0.3 if is_drums else 0.7))
    if r_overlap >= 0.10 and not (prof.hard_poly <= 4 and not is_drums):
        flags.append(f"same_pitch_overlaps:{r_overlap:.0%}")

    if prof.max_note_dur is not None:
        r_stuck = sum(1 for d in durs if d > prof.max_note_dur) / n
        s_stuck = 1.0 - min(1.0, r_stuck * 1.5)
        components.append((s_stuck, 1.0 if is_drums else 0.7))
        if r_stuck >= 0.03:
            flags.append(f"stuck_notes:{r_stuck:.0%} longer than {prof.max_note_dur:.0f} s")

    union_s = _union_duration(notes)
    density = n / union_s if union_s > 1e-9 else 0.0
    if union_s > 1e-9 and prof.min_density > 0:
        s_density = min(1.0, (density / prof.min_density) ** 0.6)
    else:
        s_density = 0.3  # notes exist but never sound
    if n < 3:
        s_density = min(s_density, 0.5)
    components.append((s_density, 0.8))
    if density < prof.min_density * 0.7:
        flags.append(f"sparse:{density:.2f} notes/s active (expected >= {prof.min_density})")
    if density > 30.0 and not is_drums:
        components.append((max(0.0, 1.0 - (density - 30.0) / 30.0), 0.4))
        flags.append(f"hyperdense:{density:.1f} notes/s active")

    start = min(s for s, _e, _p, _v in notes)
    end = max(e for _s, e, _p, _v in notes)
    span_s = max(0.0, end - start)
    s_span = 1.0
    if piece_duration_s > 1e-9:
        span_ratio = span_s / piece_duration_s
        if n < 24 and span_ratio < 0.02:
            s_span = 0.5
            flags.append(f"fragmentary:{n} notes over {span_s:.1f} s ({span_ratio:.1%} of piece)")
        elif n < 8 and span_ratio < 0.05:
            s_span = 0.7
        elif n >= 8:
            last_onset = max(s for s, _e, _p, _v in notes)
            if last_onset < 0.5 * piece_duration_s:
                flags.append(
                    f"early_exit: last onset at {last_onset:.1f} s of a "
                    f"{piece_duration_s:.1f} s piece (possible dropout)"
                )
    components.append((s_span, 0.7))

    r_truncated = truncated / n
    s_pairing = 1.0 - min(1.0, r_truncated * 2.0)
    components.append((s_pairing, 0.5))
    if r_truncated >= 0.05:
        flags.append(f"truncated_notes:{r_truncated:.0%} (missing note_off)")


    # --- duration outliers (statistical, long side) -------------------------
    d_out_rate, d_cv = _duration_outlier_stats(durs)
    s_dur = 1.0 - min(1.0, d_out_rate * 2.2)
    components.append((s_dur, 0.6))
    if d_out_rate >= 0.02:
        flags.append(
            f"duration_outliers:{d_out_rate:.0%} of notes longer than "
            f"{_DUR_OUTLIER_X:.0f}x the median duration"
        )
    if n >= 12 and d_cv > _DUR_CV_FLAG:
        components.append((max(0.0, 1.0 - (d_cv - _DUR_CV_FLAG) / 6.0), 0.25))
        flags.append(f"duration_variance: CV {d_cv:.1f} (wildly inconsistent note lengths)")

    # --- chord sanity ---------------------------------------------------------
    n_chords, n_bad_chords, chord_rate, chord_reasons = _chord_stats(
        notes, _max_chord_for(bucket, prof), is_drums)
    s_chord = (1.0 - min(1.0, chord_rate * 1.3)) if n_chords else 1.0
    components.append((s_chord, 0.4 if is_drums else 0.7))
    f_chord = 1.0
    if n_chords >= 3 and chord_rate >= 0.35:
        f_chord = 0.75  # systematic chord corruption
    if n_chords and n_bad_chords:
        reasons_txt = ", ".join(f"{k}:{v}" for k, v in sorted(chord_reasons.items()))
        flags.append(
            f"bad_chords:{n_bad_chords}/{n_chords} simultaneous-onset chords "
            f"flagged ({reasons_txt})"
        )
        if f_chord < 1.0:
            flags.append("chord_structure_corrupt: most chords are implausible")


    w_sum = sum(w for _s, w in components)
    base = math.exp(sum(w * math.log(max(s, 1e-3)) for s, w in components) / w_sum) if w_sum else 1.0

    # --- evidence ceiling ---------------------------------------------------
    cap = _evidence_cap(n, union_s)
    if n < 32:
        flags.append(f"low_evidence:{n} notes (score capped at {cap:.2f})")

    score = (base * f_short * f_range * f_poly * f_line * f_gap * f_chord
             * max(0.0, min(1.0, leak_factor)))
    score = max(0.0, min(1.0, min(score, cap)))

    flags.extend(leak_flags)

    diag: Dict[str, Any] = {
        "note_count": n,
        "active_span_s": round(span_s, 3),
        "union_active_s": round(union_s, 3),
        "notes_per_active_s": round(density, 3),
        "median_note_dur_s": round(med_dur, 4),
        "metrics": {
            "micro_note_rate": round(r_micro, 4),
            "short_note_rate": round(r_short, 4),
            "same_pitch_overlap_rate": round(r_overlap, 4),
            "soft_polyphony_violation_rate": round(r_poly_soft, 4),
            "hard_polyphony_violation_rate": round(r_poly_hard, 4),
            "fast_rearticulation_rate": round(r_fast, 4),
            "out_of_range_rate": round(r_range, 4),
            "truncated_rate": round(r_truncated, 4),
            "leak_factor": round(leak_factor, 4),
            "onset_hole_fraction": round(hole_fraction, 4),
            "longest_onset_hole_s": round(longest_hole, 3),
            "n_onset_holes": n_holes,
            "median_onset_ioi_s": round(med_ioi, 4),
            "duration_outlier_rate": round(d_out_rate, 4),
            "duration_cv": round(d_cv, 4),
            "chord_count": n_chords,
            "flagged_chord_rate": round(chord_rate, 4),
        },
        "factors": {
            "base": round(base, 4),
            "f_short": round(f_short, 4),
            "f_range": round(f_range, 4),
            "f_poly": round(f_poly, 4),
            "f_line": round(f_line, 4),
            "f_gap": round(f_gap, 4),
            "f_chord": round(f_chord, 4),
            "f_leak": round(leak_factor, 4),
            "evidence_cap": cap,
        },
        "chord_reasons": chord_reasons,
        "flags": flags,
    }
    return score, diag


# ---------------------------------------------------------------------------
# 8. Public API
# ---------------------------------------------------------------------------


def analyze_midi_instruments(
    midi: Union[str, "os.PathLike[str]", bytes, bytearray, mido.MidiFile],
    *,
    use_bucket_numbers: bool = False,
    good_threshold: float = 0.70,
    return_details: bool = False,
    bucket_names: Optional[Dict[str, int]] = None,
):
    """Analyze per-instrument transcription quality of an AMT (MT3) MIDI file.

    Parameters
    ----------
    midi:
        Path to a ``.mid`` file, raw bytes, or a ``mido.MidiFile``.
    use_bucket_numbers:
        If True, the tuples contain the bucket numbers from
        ``MT3_FULL_PLUS_GROUP_NAMES`` (e.g. 36 for drums) instead of the
        bucket names.
    good_threshold:
        Only used to annotate ``details["classification"]`` when
        ``return_details=True``; the returned list itself is undivided.
    return_details:
        If True, return ``(results, details)`` where ``details`` holds
        per-instrument metrics, factors, flags, parsing info and the
        buckets absent from the file.
    bucket_names:
        Optional replacement for the module-level
        ``MT3_FULL_PLUS_GROUP_NAMES`` mapping (name -> bucket number).

    Returns
    -------
    A single list of ``(instrument, quality_score, note_count)`` tuples,
    sorted best-first (quality score descending).  ``instrument`` is the
    bucket name, or the bucket number when ``use_bucket_numbers=True``;
    ``note_count`` is the number of notes assigned to that bucket.  Only
    instruments actually present in the file are reported.  With
    ``return_details=True`` the return value is ``(results, details)``.
    """
    name_to_number = dict(MT3_FULL_PLUS_GROUP_NAMES if bucket_names is None else bucket_names)

    mf = _parse_midi(midi)
    buckets, piece_duration_s, parse_diag = _extract_notes(mf)
    buckets = {name: notes for name, notes in buckets.items() if notes}

    if not buckets or piece_duration_s <= 0.0:
        details = {
            "piece_duration_s": round(piece_duration_s, 3),
            "good_threshold": good_threshold,
            "total_notes": 0,
            "classification": {},
            "instruments": {},
            "parsing": parse_diag,
            "absent_buckets": sorted(name_to_number),
            "note": "No note events found in the MIDI file.",
        }
        return ([], details) if return_details else []

    leak_factor, leak_flags = _leakage_analysis(buckets)

    instruments: Dict[str, Dict[str, Any]] = {}
    scored: List[Tuple[str, float]] = []
    for name in sorted(buckets):
        notes = buckets[name]
        score, diag = _score_bucket(
            name, notes, piece_duration_s,
            leak_factor.get(name, 1.0),
            leak_flags.get(name, []),
            parse_diag["truncated_by_bucket"].get(name, 0),
        )
        diag["bucket_number"] = name_to_number.get(name)
        diag["programs"] = sorted(parse_diag["programs_by_bucket"].get(name, set()))
        diag["score"] = round(score, 4)
        diag["classification"] = "good" if score >= good_threshold else "bad"
        instruments[name] = diag
        scored.append((name, score))

    results = [
        (name_to_number.get(name, name) if use_bucket_numbers else name,
         round(score, 3), len(buckets[name]))
        for name, score in scored
    ]
    results.sort(key=lambda t: (-t[1], str(t[0])))

    if return_details:
        details = {
            "piece_duration_s": round(piece_duration_s, 3),
            "good_threshold": good_threshold,
            "total_notes": sum(len(ns) for ns in buckets.values()),
            "use_bucket_numbers": use_bucket_numbers,
            "classification": {name: diag["classification"] for name, diag in instruments.items()},
            "instruments": instruments,
            "parsing": {
                **parse_diag,
                "programs_by_bucket": {
                    k: sorted(v) for k, v in parse_diag["programs_by_bucket"].items()
                },
            },
            "absent_buckets": sorted(set(name_to_number) - set(buckets)),
        }
        return results, details
    return results


# ---------------------------------------------------------------------------
# 9. CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import sys

    argv = sys.argv[1:]
    args = [a for a in argv if not a.startswith("--")]
    flag_set = {a for a in argv if a.startswith("--")}
    if not args:
        print("usage: python analyze_midi_instruments.py <file.mid> [--numbers] [--details]")
        raise SystemExit(0)

    res = analyze_midi_instruments(
        args[0],
        use_bucket_numbers="--numbers" in flag_set,
        return_details="--details" in flag_set,
    )
    results, details = res if isinstance(res, tuple) else (res, None)
    print(f"Instruments ({len(results)}), best first:  (bucket, score, note_count)")
    for key, score, count in results:
        print(f"  {score:5.2f}  {count:6d} notes  {key}")
    if details is not None:
        import json
        print(json.dumps(details, indent=2, default=str))
