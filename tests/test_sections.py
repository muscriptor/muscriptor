"""String section: re-articulation, violin/viola/cello split, tidy parts."""

from collections import Counter

import numpy as np
import pytest

from muscriptor.sheet.articulation import merge_repeated_notes
from muscriptor.sheet.ir import PerformanceNote
from muscriptor.sheet.sections import (
    CELLO,
    VIOLIN,
    SectionSplitOptions,
    assign_sections,
    split_string_sections,
    tidy_part,
)

TWO = SectionSplitOptions(sections=(CELLO, VIOLIN), articulation="off")
THREE = SectionSplitOptions(articulation="off")


def _n(pitch, onset, dur=0.9, inst="string_ensemble", source="other"):
    return PerformanceNote(pitch, onset, onset + dur, inst, source=source)


def _by(notes):
    out = {}
    for n in notes:
        out.setdefault(n.instrument, []).append(n.pitch)
    return out


def _block_chords():
    chords = []
    for i, root in enumerate([36, 41, 43, 36] * 4):
        for p in (root, root + 7, root + 16, root + 31, root + 36, root + 40):
            chords.append(_n(p, float(i)))
    return chords


# --- assignment -------------------------------------------------------------


def test_two_parts_split_block_chords():
    out = _by(assign_sections(_block_chords(), TWO))
    assert set(out) == {"cello", "violin"}
    assert max(out["cello"]) < min(out["violin"])
    assert min(out["violin"]) >= 55


def test_three_parts_take_two_notes_each_in_register_order():
    out = assign_sections(_block_chords(), THREE)
    assert set(_by(out)) == {"cello", "viola", "violin"}
    rank = {"cello": 0, "viola": 1, "violin": 2}
    for onset in {n.onset for n in out}:
        chord = sorted((n for n in out if n.onset == onset), key=lambda n: n.pitch)
        assert [rank[n.instrument] for n in chord] == sorted(
            rank[n.instrument] for n in chord
        )
    per_chord = Counter((n.onset, n.instrument) for n in out)
    assert set(per_chord.values()) == {2}


def test_ranges_respected():
    notes = [_n(50, i * 0.5, 0.4) for i in range(20)] + [
        _n(88, i * 0.5, 0.4) for i in range(20)
    ]
    out = _by(assign_sections(notes, TWO))
    assert set(out["cello"]) == {50}
    assert set(out["violin"]) == {88}


def test_moving_lines_stay_in_their_part():
    low = [
        _n(p, i * 0.5, 0.45) for i, p in enumerate([50, 52, 53, 55, 57, 55, 53, 52] * 2)
    ]
    high = [
        _n(p, i * 0.5 + 0.25, 0.45)
        for i, p in enumerate([69, 67, 65, 64, 62, 64, 65, 67] * 2)
    ]
    out = _by(assign_sections(low + high, TWO))
    assert sorted(out["cello"]) == sorted(n.pitch for n in low)
    assert sorted(out["violin"]) == sorted(n.pitch for n in high)


def test_single_high_line_is_one_violin_part():
    melody = [_n(p, i * 0.5, 0.45) for i, p in enumerate([67, 69, 71, 72, 74, 72] * 4)]
    melody.append(_n(59, 20.0))
    assert set(_by(assign_sections(melody, THREE))) == {"violin"}


# --- tidy -------------------------------------------------------------------


def _overlaps(notes):
    """(same-pitch overlaps, chords whose notes end apart)."""
    same = 0
    by_pitch = {}
    for n in sorted(notes, key=lambda n: n.onset):
        p = by_pitch.get(n.pitch)
        if p is not None and p.offset > n.onset + 1e-9:
            same += 1
        by_pitch[n.pitch] = n
    ends = {}
    for n in notes:
        ends.setdefault(n.onset, set()).add(round(n.offset, 6))
    ragged = sum(1 for e in ends.values() if len(e) > 1)
    return same, ragged


def test_tidy_aligns_chord_ends_closes_gaps_and_caps_divisi():
    notes = [
        _n(60, 0.0, 0.95),
        _n(64, 0.0, 1.02),
        _n(67, 0.0, 1.05),  # 3 notes: divisi 2 keeps 60 and 67
        _n(62, 1.08, 0.9),  # 0.08 s gap → closed
        _n(65, 1.08, 1.6),  # rings past the next chord: overlap kept for voice 2
        _n(62, 2.0, 0.5),  # same pitch as the 62 at 1.08 (which ends 1.98)
    ]
    out = tidy_part(notes, VIOLIN)
    first = [n for n in out if n.onset == 0.0]
    assert sorted(n.pitch for n in first) == [60, 67]
    assert {n.offset for n in first} == {1.08}
    assert _overlaps(out)[0] == 0


def test_tidy_cello_folds_below_c2():
    notes = [_n(31, 0.0), _n(43, 0.0), _n(28, 1.0)]
    out = tidy_part(notes, CELLO)
    # G1 doubles G2: dropped; E1 → E2.
    assert sorted(n.pitch for n in out) == [40, 43]


def test_full_stage_parts_are_clean():
    rng = np.random.default_rng(0)
    notes = []
    t = 0.0
    for _ in range(40):
        root = int(rng.choice([36, 41, 43, 45]))
        dur = float(rng.uniform(0.2, 1.0))
        for p in (root, root + 7, root + 16, root + 24, root + 28, root + 31):
            notes.append(
                _n(
                    p,
                    t + float(rng.uniform(0, 0.02)),
                    dur + float(rng.uniform(-0.1, 0.1)),
                )
            )
        t += dur + float(rng.uniform(-0.05, 0.1))
    notes.append(_n(60, 0.0, inst="acoustic_piano", source="piano"))
    out = split_string_sections(notes, THREE)
    assert [n.instrument for n in out if n.source == "piano"] == ["acoustic_piano"]
    for part in ("cello", "viola", "violin"):
        pn = [n for n in out if n.instrument == part]
        assert pn, part
        assert _overlaps(pn)[0] == 0
        assert max(Counter(n.onset for n in pn).values()) <= 2


# --- re-articulation ---------------------------------------------------------


def test_merge_all_and_off():
    notes = [
        _n(60, 0.0, 0.25),
        _n(60, 0.25, 0.25),
        _n(60, 0.5, 0.25),
        _n(62, 0.75, 0.25),
    ]
    assert len(merge_repeated_notes(notes, "off")) == 4
    merged = merge_repeated_notes(notes, "legato")
    assert [(n.pitch, n.onset, n.offset) for n in merged] == [
        (60, 0.0, 0.75),
        (62, 0.75, 1.0),
    ]


def test_merge_audio_keeps_real_rebowing(tmp_path):
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("librosa")
    sr = 22050
    t = np.arange(int(4 * sr)) / sr
    f = 440.0
    tone = np.sin(2 * np.pi * f * t) + 0.5 * np.sin(4 * np.pi * f * t)
    # 0-2 s: one held note. 2-4 s: four strokes, each starting from silence.
    env = np.ones_like(t)
    for k in range(4):
        a = int((2 + 0.5 * k) * sr)
        env[a : a + int(0.12 * sr)] = np.linspace(0, 1, int(0.12 * sr))
        env[a - int(0.05 * sr) : a] = 0.0
    path = tmp_path / "strings.wav"
    sf.write(str(path), (0.3 * tone * env).astype(np.float32), sr)
    held = [_n(69, 0.5 * k, 0.5) for k in range(4)]  # model split the held note
    strokes = [_n(69, 2 + 0.5 * k, 0.5) for k in range(4)]
    out = merge_repeated_notes(held + strokes, "audio", path)
    onsets = sorted(n.onset for n in out)
    assert onsets[0] == 0.0 and 0.5 not in onsets and 1.0 not in onsets
    assert {2.5, 3.0, 3.5} <= set(onsets)


def test_merge_audio_without_audio_merges_only_fragments():
    notes = [_n(60, 0.0, 0.5), _n(60, 0.5, 0.05), _n(60, 0.55, 0.5)]
    out = merge_repeated_notes(notes, "audio", None)
    assert len(out) == 1


def test_tidy_cuts_held_notes_without_a_free_desk():
    # A long D4 under a two-note chord at 1.0: no room (divisi 2) → cut there.
    # A long A3 pedal under a single moving note: room → kept as a second voice.
    notes = [
        _n(62, 0.0, 3.0),
        _n(67, 1.0, 0.5),
        _n(71, 1.0, 0.5),
        _n(57, 4.0, 3.0),
        _n(69, 5.0, 0.5),
    ]
    out = {(n.pitch, n.onset): n for n in tidy_part(notes, VIOLIN)}
    assert out[(62, 0.0)].offset == 1.0
    assert out[(57, 4.0)].offset == 7.0
