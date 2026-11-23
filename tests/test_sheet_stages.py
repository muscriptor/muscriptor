"""Unit tests for the audio-to-sheet reconstruction stages (no models needed)."""

import xml.etree.ElementTree as ET
from fractions import Fraction as F

import numpy as np
import pytest

from muscriptor.sheet import cleanup, harmony, notation, rhythm, voices
from muscriptor.sheet.instruments import spec_for
from muscriptor.sheet.ir import PerformanceNote, QuantizedNote
from muscriptor.sheet.lint import lint_score
from muscriptor.sheet.musicxml import MusicXMLError, note_type, score_to_musicxml


def pn(pitch, onset, offset, inst="acoustic_piano"):
    return PerformanceNote(pitch=pitch, onset=onset, offset=offset, instrument=inst)


def qn(pitch, onset, offset, inst="acoustic_piano"):
    return QuantizedNote(pitch=pitch, onset=F(onset), offset=F(offset), instrument=inst)


# ---------------------------------------------------------------------------
# cleanup


def test_duplicates_are_merged():
    notes = [pn(60, 1.0, 1.5), pn(60, 1.01, 1.7)]
    out = cleanup.remove_duplicates(notes, 0.035)
    assert len(out) == 1 and out[0].offset == 1.7


def test_split_fragment_merged_but_repeated_notes_kept():
    # A sustained note with a 40 ms fragment split off is one note; two full
    # notes back to back (a re-strummed chord, a repeated syllable) are two.
    notes = [pn(60, 0.0, 1.0), pn(60, 1.01, 1.05), pn(60, 2.0, 2.5), pn(60, 2.5, 3.0)]
    out = cleanup.merge_rearticulations(notes, 0.05)
    assert [(n.onset, n.offset) for n in out] == [(0.0, 1.05), (2.0, 2.5), (2.5, 3.0)]


def test_bass_labels_unified():
    notes = [pn(40 + i % 3, i, i + 0.5, "electric_bass") for i in range(50)]
    notes += [pn(40, i + 0.5, i + 0.9, "acoustic_bass") for i in range(45)]
    out = cleanup.consolidate_instruments(notes, cleanup.CleanupOptions())
    assert {n.instrument for n in out} == {"electric_bass"}


def test_harmonic_ghost_removed():
    notes = [pn(48, 1.0, 2.0), pn(60, 1.01, 1.08), pn(64, 1.0, 1.9)]
    out = cleanup.remove_harmonic_ghosts(notes, 0.15, 0.04)
    assert sorted(n.pitch for n in out) == [48, 64]


def test_bass_monophony_keeps_lowest_and_cuts_overlaps():
    notes = [
        pn(40, 0.0, 1.5, "electric_bass"),
        pn(52, 0.01, 1.0, "electric_bass"),
        pn(43, 1.0, 2.0, "electric_bass"),
    ]
    out = cleanup.enforce_monophony(notes, "low", 0.04)
    assert [(n.pitch, n.offset) for n in out] == [(40, 1.0), (43, 2.0)]


def test_monophonic_part_drops_chord_loops():
    # A vocal stem where the model looped on a 7-note chord: not singable.
    chord = [
        pn(p, 1.0 + 0.2 * k, 1.2 + 0.2 * k, "voice")
        for k in range(10)
        for p in (50, 57, 62, 66, 69, 74, 78)
    ]
    melody = [pn(67, 0.0, 0.5, "voice"), pn(69, 0.5, 1.0, "voice")]
    out = cleanup.enforce_monophony(melody + chord, "high", 0.04)
    assert [n.pitch for n in out] == [67, 69]


def test_octave_errors_folded_bleed_dropped():
    notes = [
        pn(18, 0, 1, "electric_bass"),
        pn(80, 1, 2, "electric_bass"),
        pn(40, 2, 3, "electric_bass"),
    ]
    out = cleanup.fold_into_range(notes, 23, 67)
    assert [n.pitch for n in out] == [30, 40]


def test_octave_jump_in_line_is_fixed():
    pitches = [60, 62, 64, 77, 65, 64, 62]
    notes = [pn(p, i, i + 1, "voice") for i, p in enumerate(pitches)]
    out = cleanup.fix_octave_jumps(notes)
    assert out[3].pitch == 65


def test_sparse_instrument_merged_into_same_role():
    notes = [pn(60 + i % 5, i, i + 0.5, "acoustic_guitar") for i in range(60)]
    notes += [pn(64, i, i + 0.5, "clean_electric_guitar") for i in range(5)]
    notes += [pn(80, i, i + 0.5, "flutes") for i in range(5)]
    out = cleanup.consolidate_instruments(notes, cleanup.CleanupOptions())
    assert {n.instrument for n in out} == {"acoustic_guitar"}
    assert len(out) == 65


def test_polyphony_limit_keeps_outer_voices():
    notes = [
        pn(p, 0.0, 1.0 + i * 0.1, "acoustic_guitar")
        for i, p in enumerate([40, 45, 50, 55, 59, 64, 69, 76])
    ]
    out = cleanup.limit_polyphony(notes, 6, 0.04)
    pitches = sorted(n.pitch for n in out)
    assert len(pitches) == 6 and pitches[0] == 40 and pitches[-1] == 76


# ---------------------------------------------------------------------------
# rhythm


def test_grid_choice_binary_vs_triplet():
    choices = rhythm.grid_choices(swing=False)
    sixteenths = np.array([0.0, 0.25, 0.5, 0.75])
    triplets = np.array([1.0, 1 + 1 / 3, 1 + 2 / 3])
    grids = rhythm.choose_grids(
        np.concatenate([sixteenths, triplets + 3]), 6, choices, rhythm.QuantizeOptions()
    )
    assert grids[0].subdivision == 4
    assert grids[4].family == "triplet"


def test_isolated_jitter_does_not_become_a_triplet():
    # Straight eighths with one slightly late off-beat.
    pos = np.array([0, 0.5, 1, 1.5, 2, 2.58, 3, 3.5])
    grids = rhythm.choose_grids(
        pos, 4, rhythm.grid_choices(False), rhythm.QuantizeOptions()
    )
    assert all(g.family != "triplet" for g in grids)


def test_swing_detected_and_written_straight():
    pos = np.concatenate([[b, b + 0.66] for b in range(60)])
    sw = rhythm.detect_swing(pos)
    assert sw.swing
    grids = rhythm.choose_grids(
        pos, 60, rhythm.grid_choices(True), rhythm.QuantizeOptions()
    )
    q, choice = rhythm.quantize_position(10.66, grids, 0.12)
    assert q == F(21, 2) and choice.name == "swing8"


def test_straight_eighths_are_not_swing():
    pos = np.concatenate([[b, b + 0.5] for b in range(60)])
    assert not rhythm.detect_swing(pos).swing


def test_quantize_part_on_placeholder_grid():
    meter = rhythm.build_meter_map(
        None, [0.0], beats_per_bar=4
    )  # 120 BPM: a beat is 0.5 s
    notes = [pn(60, 0.01, 0.49), pn(62, 0.26, 0.5), pn(64, 0.5, 1.48)]
    q = rhythm.quantize_part(notes, meter, 8, swing=False)
    assert [(n.onset, n.offset) for n in q] == [
        (F(0), F(1)),
        (F(1, 2), F(1)),
        (F(1), F(3)),
    ]


def test_meter_phase_from_chord_changes():
    beats = np.arange(64) * 0.5
    # Chord changes every 4 beats starting at beat 1 (index 1), kicks on 1 and 3.
    events = [(beats[i], 1.0) for i in range(1, 64, 4)]
    events += [(beats[i], 0.25) for i in range(1, 64, 2)]
    guess = rhythm.infer_meter_phase(beats, events)
    assert (guess.beats_per_bar, guess.phase) == (4, 1)


def test_waltz_meter_detected():
    beats = np.arange(60) * 0.5
    events = [(beats[i], 1.0) for i in range(0, 60, 3)]
    guess = rhythm.infer_meter_phase(beats, events)
    assert (guess.beats_per_bar, guess.phase) == (3, 0)


def test_fill_gaps_holds_notes_to_next_onset():
    q = [qn(60, 0, F(1, 2)), qn(62, 1, 2)]
    out = rhythm.fill_gaps(q, F(1, 2))
    assert out[0].offset == 1


# ---------------------------------------------------------------------------
# voices


def test_hands_split_around_middle_c():
    notes = [qn(p, 0, 1) for p in (36, 43, 48, 64, 67, 72)]
    out = voices.split_hands(notes)
    assert {n.pitch for n in out if n.staff == 2} == {36, 43, 48}


def test_voice_invariant_no_overlap_within_voice():
    notes = [qn(48, 0, 4), qn(64, 0, 1), qn(67, 1, 2), qn(65, 2, 3), qn(64, 3, 4)]
    out = voices.assign_voices(notes)
    by_voice = {}
    for n in out:
        by_voice.setdefault(n.voice, []).append(n)
    assert {n.pitch for n in by_voice[2]} == {48}  # held bass goes to voice 2
    for vs in by_voice.values():
        spans = sorted({(n.onset, n.offset) for n in vs})
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            assert a1 <= b0


def test_drums_hands_and_feet():
    notes = [
        qn(36, 0, F(1, 4), "drums"),
        qn(42, 0, F(1, 4), "drums"),
        qn(38, 1, F(5, 4), "drums"),
    ]
    out = voices.layout_drums(notes)
    assert {n.pitch: n.voice for n in out} == {36: 2, 42: 1, 38: 1}
    assert all(n.offset <= int(n.onset) + 1 for n in out)


# ---------------------------------------------------------------------------
# notation


@pytest.mark.parametrize(
    "start,end,expected",
    [
        (0, 4, [(0, 4)]),
        (1, 3, [(1, 1), (2, 1)]),  # half note on beat 2 → tied across the middle
        (2, 4, [(2, 2)]),
        (F(1, 2), F(3, 2), [(F(1, 2), F(1))]),  # eighth-quarter-eighth syncopation
        (
            F(1, 2),
            F(5, 2),
            [(F(1, 2), F(3, 2)), (2, F(1, 2))],
        ),  # ... but tied across the half bar
        (
            F(1, 4),
            F(5, 4),
            [(F(1, 4), F(3, 4)), (1, F(1, 4))],
        ),  # off-sixteenth: tied at the beat
        (0, F(3, 2), [(0, F(3, 2))]),  # dotted quarter
        (F(1, 4), 1, [(F(1, 4), F(3, 4))]),  # sixteenth + dotted eighth
        (0, 3, [(0, 3)]),
    ],
)
def test_split_span(start, end, expected):
    got = [(o, d) for o, d, _ in notation.split_span(F(start), F(end), F(4), 4, set())]
    assert got == [(F(a), F(b)) for a, b in expected]


def test_split_span_triplet_beat():
    got = notation.split_span(F(0), F(2, 3), F(4), 4, {0})
    assert got == [(F(0), F(2, 3), True)]


@pytest.mark.parametrize(
    "midi,fifths,mode,expected",
    [
        (70, -2, "major", ("B", -1)),  # Bb in Bb major
        (66, 2, "minor", ("F", 1)),  # F# in B minor
        (70, 2, "minor", ("A", 1)),  # A# leading tone in B minor
        (63, 0, "major", ("E", -1)),  # Eb chromatic in C
        (61, 0, "major", ("C", 1)),
    ],
)
def test_spelling(midi, fifths, mode, expected):
    p = notation.spell(midi, fifths, mode)
    assert (p.step, p.alter) == expected


def _score(parts_notes, bpb=4, n_measures=2, chords=None):
    parts = []
    for i, (inst, notes) in enumerate(parts_notes, start=1):
        spec = spec_for(inst)
        layout, laid = voices.layout_part(spec, notes)
        parts.append(notation.PartInput(spec, layout, laid, f"P{i}"))
    ctx = notation.ScoreContext(
        beats_per_bar=bpb,
        n_measures=n_measures,
        title="t",
        keys={0: (2, "minor")},
        tempos={0: 90},
        chords=chords or {},
        chord_parts={"P1"},
    )
    return notation.build_score(parts, ctx)


def test_musicxml_measures_sum_and_parse():
    score = _score(
        [
            (
                "voice",
                [
                    qn(66, 0, F(3, 2)),
                    qn(69, F(3, 2), 3),
                    qn(71, F(10, 3), F(11, 3)),
                    qn(73, F(11, 3), 6),
                ],
            ),
            (
                "acoustic_piano",
                [qn(47, 0, 4), qn(62, 0, 2), qn(66, 0, 2), qn(64, 2, 4), qn(50, 4, 8)],
            ),
            (
                "drums",
                [
                    qn(36, 0, F(1, 2), "drums"),
                    qn(42, 0, F(1, 2), "drums"),
                    qn(38, 1, 2, "drums"),
                ],
            ),
        ],
        chords={
            0: [
                harmony.chord_symbol(
                    harmony.ChordLabel(11, "min"), harmony.Key(11, "minor"), F(0)
                )
            ]
        },
    )
    xml = score_to_musicxml(score)
    root = ET.fromstring(xml.split(b"\n", 2)[2])
    assert root.tag == "score-partwise"
    assert len(root.findall("part")) == 3
    piano = root.findall("part")[1]
    assert piano.find("measure/attributes/staves").text == "2"
    assert root.find("part/measure/harmony/kind").text == "minor"
    assert root.find("part/measure/attributes/key/fifths").text == "2"
    # Triplet on beat 4 of the voice part's first measure.
    assert list(root.findall("part")[0].findall("measure")[0].iter("time-modification"))
    # Deterministic.
    assert score_to_musicxml(score) == xml
    lint = lint_score(score)
    assert lint["totals"]["notes"] > 0


def test_tie_across_barline():
    score = _score([("voice", [qn(67, 3, 5)])])
    m1, m2 = score.parts[0].measures
    last = [e for e in m1.voices[0].events if not e.is_rest][-1]
    first = [e for e in m2.voices[0].events if not e.is_rest][0]
    assert last.tie_start and first.tie_stop and first.offset == 0


def test_accidental_once_per_measure():
    score = _score(
        [("voice", [qn(65, 0, 1), qn(65, 1, 2), qn(66, 2, 3)])], n_measures=1
    )
    pitches = [
        p for e in score.parts[0].measures[0].voices[0].events for p in e.pitches
    ]
    # Key of B minor (F#): F natural needs a natural sign once, F# then a sharp.
    assert [p.accidental for p in pitches] == ["natural", None, "sharp"]


def test_note_type_rejects_unwritable_values():
    assert note_type(F(3, 2)) == ("quarter", 1)
    with pytest.raises(MusicXMLError):
        note_type(F(5, 4))


# ---------------------------------------------------------------------------
# harmony


def test_parse_harte():
    c = harmony.parse_harte("D:maj/5")
    assert (c.root, c.quality, c.bass) == (2, "maj", 9)
    assert harmony.parse_harte("A:sus4(b7)").quality == "7sus4"
    assert harmony.parse_harte("N").is_none


def _frames_notes(chord_pitches, bass, beats=4, inst="acoustic_piano"):
    notes = []
    for b in range(beats):
        # An arpeggio: one chord tone per beat, never all at once.
        p = chord_pitches[b % len(chord_pitches)]
        notes.append(pn(p, b * 0.5, b * 0.5 + 0.5, inst))
    notes.append(pn(bass, 0.0, beats * 0.5, "electric_bass"))
    return notes


def test_arpeggio_recognized_as_chord_with_inversion():
    notes = _frames_notes([60, 64, 67, 72], bass=40)  # C major arpeggio over E
    segs = harmony.recognize_chords(notes, np.arange(5) * 0.5, 4, None)
    assert len(segs) == 1
    assert str(segs[0].chord) == "C/E"


def test_passing_bass_note_makes_no_slash_chord():
    notes = [pn(p, 0, 2.0) for p in (57, 60, 64)]  # A minor
    notes += [
        pn(45, 0, 1.6, "electric_bass"),
        pn(47, 1.6, 1.8, "electric_bass"),
        pn(48, 1.8, 2.0, "electric_bass"),
    ]
    segs = harmony.recognize_chords(notes, np.arange(5) * 0.5, 4, None)
    assert str(segs[0].chord) == "Am"


def test_seventh_needs_evidence():
    triad = [pn(p, 0, 2.0) for p in (55, 59, 62)]
    seventh = triad + [pn(65, 0, 2.0)]
    t = harmony.recognize_chords(triad, np.arange(5) * 0.5, 4, None)
    s = harmony.recognize_chords(seventh, np.arange(5) * 0.5, 4, None)
    assert str(t[0].chord) == "G" and str(s[0].chord) == "G7"


def test_audio_evidence_breaks_ambiguity():
    # Notes E G B D are Em7 or G6: the audio model says G6 → it wins.
    notes = [pn(p, 0, 2.0) for p in (52, 55, 59, 62)]
    audio = [(0.0, 2.0, "G:maj6")]
    segs = harmony.recognize_chords(notes, np.arange(5) * 0.5, 4, audio)
    assert str(segs[0].chord) == "G6"


def test_chord_spelling_in_key():
    d_major = harmony.Key(2, "major")
    assert (
        harmony.chord_symbol(harmony.ChordLabel(10, "maj"), d_major, F(0)).label == "Bb"
    )
    b_minor = harmony.Key(11, "minor")
    assert (
        harmony.chord_symbol(harmony.ChordLabel(6, "maj", 10), b_minor, F(0)).label
        == "F#/A#"
    )
    assert (
        harmony.chord_symbol(harmony.ChordLabel(9, "7sus4"), b_minor, F(0)).label
        == "A7sus4"
    )


def test_key_estimation_relative_minor_by_chords():
    scale = [71, 73, 74, 76, 78, 79, 81]  # B natural minor / D major collection
    notes = [pn(p, i * 0.5, i * 0.5 + 0.5) for i, p in enumerate(scale * 4)]
    chords = [harmony.ChordLabel(11, "min")] * 6 + [harmony.ChordLabel(2, "maj")] * 2
    key = harmony.estimate_key(notes, chords)
    assert (key.tonic, key.mode, key.fifths) == (11, "minor", 2)
