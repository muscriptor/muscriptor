"""Harmony: key estimation and chord symbols from audio + notes + bass line.

Chord recognition fuses three sources, per beat, and decodes the sequence
jointly with a Viterbi pass:

1. **Audio** — lv-chordia (Jiang et al., ISMIR 2019, MIT), a large-vocabulary
   recognizer that decomposes chords into root / quality / bass. It runs on
   the original mix, which keeps the full harmonic context (separated stems
   lose some of it). Optional: without it the symbolic evidence decides alone.
2. **Notes** — the cleaned transcription's pitch classes, weighted by how long
   they sound in the beat, whether they start on the beat, and the role of the
   instrument (accompaniment counts fully, a sung melody — full of passing and
   neighbour tones — counts half, drums not at all). Because the decoder keeps
   a chord across beats unless the evidence changes, an arpeggio spread over a
   bar accumulates into its chord instead of being read one note at a time.
3. **Bass** — the bass part (or the lowest notes when there is none) decides
   inversions and slash chords after the chord is chosen. Following the
   Chordotomy rule, only a bass note that dominates the chord's span — long or
   on strong beats — can make a slash chord; passing bass notes cannot.

Chord changes cost more off the bar line than on it, so symbols land where a
musician expects them (bar, half bar) and ornamental notes do not flip the
chord. Extensions (7, 9, 6), sus and diminished/augmented qualities need
evidence beyond the triad to be chosen: each carries a small complexity cost.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from muscriptor.sheet.instruments import spec_for
from muscriptor.sheet.ir import ChordSymbol, PerformanceNote

logger = logging.getLogger(__name__)

PC_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


# ---------------------------------------------------------------------------
# Chord vocabulary


@dataclass(frozen=True)
class Quality:
    name: str  # internal id
    intervals: tuple[int, ...]
    suffix: str  # printed after the root
    kind: str  # MusicXML <kind>
    complexity: float  # evidence needed beyond the triad
    degrees: tuple[tuple[int, int, str], ...] = ()


QUALITIES: list[Quality] = [
    Quality("maj", (0, 4, 7), "", "major", 0.0),
    Quality("min", (0, 3, 7), "m", "minor", 0.0),
    Quality("7", (0, 4, 7, 10), "7", "dominant", 0.04),
    Quality("maj7", (0, 4, 7, 11), "maj7", "major-seventh", 0.05),
    Quality("min7", (0, 3, 7, 10), "m7", "minor-seventh", 0.04),
    Quality("dim", (0, 3, 6), "dim", "diminished", 0.07),
    Quality("dim7", (0, 3, 6, 9), "dim7", "diminished-seventh", 0.08),
    Quality("hdim7", (0, 3, 6, 10), "m7b5", "half-diminished", 0.07),
    Quality("aug", (0, 4, 8), "+", "augmented", 0.08),
    Quality("sus4", (0, 5, 7), "sus4", "suspended-fourth", 0.05),
    Quality("sus2", (0, 2, 7), "sus2", "suspended-second", 0.06),
    Quality(
        "7sus4", (0, 5, 7, 10), "7sus4", "suspended-fourth", 0.06, ((7, 0, "add"),)
    ),
    Quality("maj6", (0, 4, 7, 9), "6", "major-sixth", 0.06),
    Quality("min6", (0, 3, 7, 9), "m6", "minor-sixth", 0.06),
    Quality("add9", (0, 2, 4, 7), "add9", "major", 0.07, ((9, 0, "add"),)),
    Quality("9", (0, 2, 4, 7, 10), "9", "dominant-ninth", 0.08),
    Quality("maj9", (0, 2, 4, 7, 11), "maj9", "major-ninth", 0.09),
    Quality("min9", (0, 2, 3, 7, 10), "m9", "minor-ninth", 0.08),
]
QUALITY_BY_NAME = {q.name: q for q in QUALITIES}

# Harte-syntax qualities (lv-chordia output) → our vocabulary.
_HARTE = {
    "maj": "maj",
    "min": "min",
    "7": "7",
    "maj7": "maj7",
    "min7": "min7",
    "dim": "dim",
    "dim7": "dim7",
    "hdim7": "hdim7",
    "aug": "aug",
    "sus4": "sus4",
    "sus2": "sus2",
    "sus4(b7)": "7sus4",
    "7sus4": "7sus4",
    "maj6": "maj6",
    "min6": "min6",
    "maj(9)": "add9",
    "add9": "add9",
    "9": "9",
    "maj9": "maj9",
    "min9": "min9",
    "minmaj7": "min",
    "11": "7",
    "min11": "min7",
    "13": "7",
    "maj13": "maj7",
    "min13": "min7",
    "5": "maj",
    "1": "maj",
    "7(b9)": "7",
    "7(#9)": "7",
    "aug7": "7",
    "min(9)": "min",
}

_DEGREE_SEMITONES = {
    "1": 0,
    "b2": 1,
    "2": 2,
    "b3": 3,
    "3": 4,
    "4": 5,
    "#4": 6,
    "b5": 6,
    "5": 7,
    "#5": 8,
    "b6": 8,
    "6": 9,
    "bb7": 9,
    "b7": 10,
    "7": 11,
    "9": 2,
    "b9": 1,
}

_NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _pc(name: str) -> int:
    pc = _NOTE_PC[name[0].upper()]
    for ch in name[1:]:
        pc += 1 if ch == "#" else -1 if ch == "b" else 0
    return pc % 12


@dataclass(frozen=True)
class ChordLabel:
    root: int  # pitch class, -1 for no chord
    quality: str
    bass: int | None = None  # pitch class of a bass other than the root

    @property
    def is_none(self) -> bool:
        return self.root < 0

    def pcs(self) -> frozenset[int]:
        if self.is_none:
            return frozenset()
        return frozenset(
            (self.root + i) % 12 for i in QUALITY_BY_NAME[self.quality].intervals
        )

    def __str__(self) -> str:
        if self.is_none:
            return "N"
        s = PC_NAMES[self.root] + QUALITY_BY_NAME[self.quality].suffix
        if self.bass is not None:
            s += "/" + PC_NAMES[self.bass]
        return s


NO_CHORD = ChordLabel(-1, "maj")


def parse_harte(label: str) -> ChordLabel:
    """Parse a Harte-syntax label ("D:maj/5", "A:sus4(b7)", "N")."""
    if label in ("N", "X", ""):
        return NO_CHORD
    m = re.match(r"^([A-G][#b]*)(?::([^/]+))?(?:/(.+))?$", label)
    if not m:
        return NO_CHORD
    root = _pc(m.group(1))
    quality = _HARTE.get(m.group(2) or "maj")
    if quality is None:
        # Unknown quality: fall back on its third.
        quality = "min" if (m.group(2) or "").startswith("min") else "maj"
    bass = None
    if m.group(3):
        semis = _DEGREE_SEMITONES.get(m.group(3))
        if semis:
            bass = (root + semis) % 12
    return ChordLabel(root, quality, bass)


ALL_CHORDS: list[ChordLabel] = [
    ChordLabel(r, q.name) for r in range(12) for q in QUALITIES
]


# ---------------------------------------------------------------------------
# Key estimation (Krumhansl–Kessler profiles)

_KK_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_KK_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

# Major-key tonic pitch class → signature (fifths); pick the conventional
# spelling (F# major with 6 sharps rather than Gb with 6 flats is a toss-up;
# we take the side with fewer accidentals, then flats).
_MAJOR_FIFTHS = {
    0: 0,
    7: 1,
    2: 2,
    9: 3,
    4: 4,
    11: 5,
    6: 6,
    1: -5,
    8: -4,
    3: -3,
    10: -2,
    5: -1,
}


@dataclass(frozen=True)
class Key:
    tonic: int  # pitch class
    mode: str  # "major" | "minor"
    score: float = 0.0

    @property
    def fifths(self) -> int:
        major_tonic = self.tonic if self.mode == "major" else (self.tonic + 3) % 12
        return _MAJOR_FIFTHS[major_tonic]

    def __str__(self) -> str:
        names_sharp = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        name = (names_sharp if self.fifths >= 0 else PC_NAMES)[self.tonic]
        return f"{name} {self.mode}"


def _key_scores(pcp: np.ndarray) -> np.ndarray:
    """Correlation of a 12-d profile with all 24 keys: [major 0..11, minor 0..11]."""
    out = np.zeros(24)
    if pcp.sum() <= 0:
        return out
    for t in range(12):
        out[t] = np.corrcoef(pcp, np.roll(_KK_MAJOR, t))[0, 1]
        out[12 + t] = np.corrcoef(pcp, np.roll(_KK_MINOR, t))[0, 1]
    return np.nan_to_num(out)


def pitch_class_profile(
    notes: list[PerformanceNote], start: float = -math.inf, end: float = math.inf
) -> np.ndarray:
    pcp = np.zeros(12)
    for n in notes:
        if n.is_drum or n.offset <= start or n.onset >= end:
            continue
        dur = min(n.offset, end) - max(n.onset, start)
        weight = 1.5 if spec_for(n.instrument).role == "bass" else 1.0
        pcp[n.pitch % 12] += weight * min(dur, 2.0)
    return pcp


def estimate_key(
    notes: list[PerformanceNote], chords: list[ChordLabel] | None = None
) -> Key:
    """Global key from the notes' pitch-class profile (plus the chords, if any).

    Relative major/minor share a profile shape; the chord sequence breaks the
    tie, since the tonic chord is the one a song keeps returning to and ends on.
    """
    pcp = pitch_class_profile(notes)
    scores = _key_scores(pcp)
    if chords:
        named = [c for c in chords if not c.is_none]
        if named:
            dominant_q = ("maj", "7", "7sus4", "sus4", "9")
            for i in range(24):
                tonic, minor = i % 12, i >= 12
                tonic_q = (
                    ("min", "min7", "min9", "min6")
                    if minor
                    else ("maj", "maj7", "maj6", "add9", "maj9")
                )

                def is_tonic(c: ChordLabel) -> bool:
                    return c.root == tonic and c.quality in tonic_q

                hits = sum(1 for c in named if is_tonic(c))
                scores[i] += 0.5 * hits / len(named)
                scores[i] += 0.08 * is_tonic(named[0]) + 0.08 * is_tonic(named[-1])
                # Authentic cadences V(7) → I: the strongest tonal signal there is.
                cadences = sum(
                    1
                    for a, b in zip(named, named[1:])
                    if a.root == (tonic + 7) % 12
                    and a.quality in dominant_q
                    and is_tonic(b)
                )
                scores[i] += 0.4 * cadences / max(len(named) - 1, 1)
    best = int(np.argmax(scores))
    return Key(best % 12, "minor" if best >= 12 else "major", float(scores[best]))


def estimate_local_keys(
    bar_pcps: np.ndarray,
    global_key: Key,
    switch_penalty: float = 2.0,
    min_bars: int = 8,
) -> list[Key]:
    """One key per bar: the global key unless another clearly takes over.

    Viterbi over the 24 keys on a smoothed per-bar profile, with a high cost
    for modulating, then any excursion shorter than `min_bars` is folded back.
    """
    n = len(bar_pcps)
    if n == 0:
        return []
    window = 4
    emissions = np.zeros((n, 24))
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        emissions[i] = _key_scores(bar_pcps[lo:hi].sum(axis=0))
    g = global_key.tonic + (12 if global_key.mode == "minor" else 0)
    # Treat relative major/minor as the same key signature for modulation:
    # only signature changes are written.
    emissions[:, g] += 0.1
    cost = -emissions[0] * 10
    back = np.zeros((n, 24), dtype=int)
    for i in range(1, n):
        trans = cost[:, None] + switch_penalty * (1 - np.eye(24))
        back[i] = trans.argmin(axis=0)
        cost = trans.min(axis=0) - emissions[i] * 10
    path = [int(cost.argmin())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    path.reverse()
    keys = [Key(p % 12, "minor" if p >= 12 else "major") for p in path]
    # Fold short excursions and anything with the global signature back.
    out = [global_key if k.fifths == global_key.fifths else k for k in keys]
    i = 0
    while i < n:
        j = i
        while j < n and out[j].fifths == out[i].fifths:
            j += 1
        if out[i].fifths != global_key.fifths and j - i < min_bars:
            for k in range(i, j):
                out[k] = global_key
        i = j
    return out


# ---------------------------------------------------------------------------
# Chord fusion


@dataclass(frozen=True)
class HarmonyOptions:
    audio_weight: float = 1.0
    symbolic_weight: float = 1.0
    # Cost of changing chord, by metric position.
    change_cost_bar: float = 0.15
    change_cost_half_bar: float = 0.4
    change_cost_beat: float = 0.9
    # Share of the span's bass weight a non-root bass note needs for a slash
    # chord (Chordotomy: passing bass notes never make one).
    slash_min_share: float = 0.55
    slash_non_chord_tone_share: float = 0.8
    # Shortest chord, in beats.
    min_chord_beats: int = 2
    # Weight of a sung melody relative to accompaniment.
    melody_weight: float = 0.5


@dataclass
class BeatFrame:
    start: float  # seconds
    end: float
    beat_index: int  # 0-based from bar 1
    pcp: np.ndarray = field(default_factory=lambda: np.zeros(12))
    bass: np.ndarray = field(default_factory=lambda: np.zeros(12))
    audio: ChordLabel = NO_CHORD
    # False when `bass` is only the lowest accompaniment note (no bass part).
    bass_from_part: bool = True


def _role_weight(instrument: str, options: HarmonyOptions) -> float:
    role = spec_for(instrument).role
    if role == "drums":
        return 0.0
    if (
        role in ("voice", "winds", "brass")
        or spec_for(instrument).monophonic
        and role != "bass"
    ):
        return options.melody_weight
    return 1.0


def beat_frames(
    notes: list[PerformanceNote],
    beat_times: np.ndarray,
    audio_chords: list[tuple[float, float, str]] | None,
    options: HarmonyOptions,
) -> list[BeatFrame]:
    """Per-beat pitch-class and bass evidence. `beat_times[i]` = start of beat i."""
    frames = [
        BeatFrame(float(beat_times[i]), float(beat_times[i + 1]), i)
        for i in range(len(beat_times) - 1)
    ]
    if not frames:
        return frames
    starts = np.array([f.start for f in frames])
    bass_notes = [n for n in notes if spec_for(n.instrument).role == "bass"]
    have_bass_part = bool(bass_notes)

    for n in notes:
        w_role = _role_weight(n.instrument, options)
        if w_role == 0:
            continue
        first = max(0, int(np.searchsorted(starts, n.onset, side="right")) - 1)
        for i in range(first, len(frames)):
            f = frames[i]
            if f.start >= n.offset:
                break
            overlap = min(n.offset, f.end) - max(n.onset, f.start)
            if overlap <= 0:
                continue
            beat_len = f.end - f.start
            w = overlap / beat_len
            # Attacked in this beat, on the beat: stronger evidence.
            if f.start - 0.05 <= n.onset < f.end:
                w *= 1.5 if abs(n.onset - f.start) < 0.1 * beat_len else 1.15
            f.pcp[n.pitch % 12] += w * w_role
            if spec_for(n.instrument).role == "bass":
                f.bass[n.pitch % 12] += w

    if not have_bass_part:
        # The lowest sounding pitch of the accompaniment stands in for the bass.
        for f in frames:
            low = [
                n
                for n in notes
                if not n.is_drum
                and n.onset < f.end
                and n.offset > f.start
                and _role_weight(n.instrument, options) == 1.0
            ]
            if low:
                p = min(low, key=lambda n: n.pitch)
                f.bass[p.pitch % 12] += 1.0
            f.bass_from_part = False

    if audio_chords:
        for f in frames:
            mid_lo, mid_hi = f.start, f.end
            best, best_ov = NO_CHORD, 0.0
            for s, e, lab in audio_chords:
                ov = min(e, mid_hi) - max(s, mid_lo)
                if ov > best_ov:
                    best, best_ov = parse_harte(lab), ov
            f.audio = best
    return frames


def _symbolic_scores(pcp: np.ndarray) -> np.ndarray:
    total = pcp.sum()
    out = np.zeros(len(ALL_CHORDS))
    if total <= 1e-6:
        return out
    for j, c in enumerate(ALL_CHORDS):
        mask = np.zeros(12, dtype=bool)
        mask[list(c.pcs())] = True
        inside, outside = pcp[mask].sum(), pcp[~mask].sum()
        # Each chord tone should actually be there: a missing third (or a
        # missing 7th in a 7th chord) is penalised.
        missing = sum(1 for pc in c.pcs() if pcp[pc] < 0.08 * total)
        root_bonus = 0.1 * pcp[c.root] / total
        out[j] = (inside - 0.7 * outside) / total - 0.12 * missing + root_bonus
        out[j] -= QUALITY_BY_NAME[c.quality].complexity
    return out


def _audio_scores(label: ChordLabel) -> np.ndarray:
    out = np.zeros(len(ALL_CHORDS))
    if label.is_none:
        return out
    lp = label.pcs()
    for j, c in enumerate(ALL_CHORDS):
        cp = c.pcs()
        jac = len(cp & lp) / len(cp | lp)
        if c.root == label.root and c.quality == label.quality:
            out[j] = 1.0
        elif c.root == label.root:
            out[j] = 0.35 + 0.4 * jac
        else:
            out[j] = 0.45 * jac
    return out


def decode_chords(
    frames: list[BeatFrame], beats_per_bar: int, options: HarmonyOptions
) -> list[ChordLabel]:
    """The chord of every beat (Viterbi over the vocabulary plus no-chord)."""
    n, k = len(frames), len(ALL_CHORDS) + 1  # last state = N
    if n == 0:
        return []
    emis = np.zeros((n, k))
    for i, f in enumerate(frames):
        sym = _symbolic_scores(f.pcp) if f.pcp.sum() > 0 else np.zeros(k - 1)
        aud = _audio_scores(f.audio)
        emis[i, :-1] = options.symbolic_weight * sym + options.audio_weight * aud
        has_evidence = f.pcp.sum() > 0.3 or not f.audio.is_none
        emis[i, -1] = 0.2 if not has_evidence else -0.5

    def change_cost(beat_index: int) -> float:
        pos = beat_index % beats_per_bar
        if pos == 0:
            return options.change_cost_bar
        if beats_per_bar in (4, 6) and pos == beats_per_bar // 2:
            return options.change_cost_half_bar
        return options.change_cost_beat

    score = emis[0].copy()
    back = np.zeros((n, k), dtype=int)
    for i in range(1, n):
        c = change_cost(frames[i].beat_index)
        best_prev = int(score.argmax())
        stay = score
        switch = score[best_prev] - c
        take_switch = switch > stay
        back[i] = np.where(take_switch, best_prev, np.arange(k))
        score = np.maximum(stay, switch) + emis[i]
    path = [int(score.argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    path.reverse()
    return [ALL_CHORDS[j] if j < k - 1 else NO_CHORD for j in path]


@dataclass
class ChordSegment:
    start_beat: int
    end_beat: int  # exclusive
    chord: ChordLabel
    start: float  # seconds
    end: float


def segments_from_beats(
    chords: list[ChordLabel], frames: list[BeatFrame], min_beats: int
) -> list[ChordSegment]:
    segs: list[ChordSegment] = []
    for i, (c, f) in enumerate(zip(chords, frames)):
        if segs and segs[-1].chord == c:
            segs[-1].end_beat = i + 1
            segs[-1].end = f.end
        else:
            segs.append(ChordSegment(i, i + 1, c, f.start, f.end))
    # Absorb segments shorter than min_beats into the previous one.
    merged: list[ChordSegment] = []
    for s in segs:
        if merged and s.end_beat - s.start_beat < min_beats:
            merged[-1].end_beat, merged[-1].end = s.end_beat, s.end
        else:
            merged.append(s)
    # And join neighbours that became equal.
    out: list[ChordSegment] = []
    for s in merged:
        if out and out[-1].chord == s.chord:
            out[-1].end_beat, out[-1].end = s.end_beat, s.end
        else:
            out.append(s)
    return out


def assign_bass(
    segments: list[ChordSegment], frames: list[BeatFrame], options: HarmonyOptions
) -> list[ChordSegment]:
    """Slash chords where the bass line holds a note other than the root."""
    for s in segments:
        if s.chord.is_none:
            continue
        bass = np.zeros(12)
        for i in range(s.start_beat, s.end_beat):
            # The first beat of the chord counts double: that is where an
            # inversion is heard.
            bass += frames[i].bass * (2.0 if i == s.start_beat else 1.0)
        total = bass.sum()
        audio_bass = {frames[i].audio.bass for i in range(s.start_beat, s.end_beat)} - {
            None
        }
        if not frames[s.start_beat].bass_from_part:
            # Without a bass part the lowest accompaniment note is too weak a
            # witness on its own (Em7 and G6/E are the same notes): only
            # confirm an inversion the audio model also hears.
            if len(audio_bass) == 1:
                b = next(iter(audio_bass))
                if (
                    b in s.chord.pcs()
                    and b != s.chord.root
                    and total > 0
                    and bass[b] / total >= options.slash_min_share
                ):
                    s.chord = ChordLabel(s.chord.root, s.chord.quality, b)
            continue
        if total <= 0:
            if len(audio_bass) == 1:
                b = audio_bass.pop()
                if b in s.chord.pcs() and b != s.chord.root:
                    s.chord = ChordLabel(s.chord.root, s.chord.quality, b)
            continue
        pc = int(bass.argmax())
        share = bass[pc] / total
        if pc == s.chord.root:
            continue
        in_chord = pc in s.chord.pcs()
        if (
            in_chord and share >= options.slash_min_share
        ) or share >= options.slash_non_chord_tone_share:
            s.chord = ChordLabel(s.chord.root, s.chord.quality, pc)
    return segments


def recognize_chords(
    notes: list[PerformanceNote],
    beat_times: np.ndarray,
    beats_per_bar: int,
    audio_chords: list[tuple[float, float, str]] | None,
    options: HarmonyOptions = HarmonyOptions(),
) -> list[ChordSegment]:
    frames = beat_frames(notes, beat_times, audio_chords, options)
    per_beat = decode_chords(frames, beats_per_bar, options)
    segs = segments_from_beats(per_beat, frames, options.min_chord_beats)
    return assign_bass(segs, frames, options)


# ---------------------------------------------------------------------------
# Spelling chord symbols in a key

# Semitones above the tonic → scale-degree steps (0 = tonic letter).
_DEGREE_STEPS = {
    0: 0,
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 3,
    6: 3,
    7: 4,
    8: 5,
    9: 5,
    10: 6,
    11: 6,
}
# Interval above a chord root → letter steps, for spelling the bass of a slash chord.
_INTERVAL_STEPS = {
    0: 0,
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 3,
    6: 4,
    7: 4,
    8: 5,
    9: 5,
    10: 6,
    11: 6,
}
_LETTERS = "CDEFGAB"


def _tonic_spelling(key: Key) -> tuple[str, int]:
    from muscriptor.sheet.notation import spell

    p = spell(60 + key.tonic, key.fifths, key.mode)
    return p.step, p.alter


def _spell_from(step: str, alter: int, semitones: int, steps: int) -> tuple[str, int]:
    base_pc = (_NOTE_PC[step] + alter) % 12
    target_step = _LETTERS[(_LETTERS.index(step) + steps) % 7]
    target_pc = (base_pc + semitones) % 12
    diff = (target_pc - _NOTE_PC[target_step] + 6) % 12 - 6
    return target_step, diff


def spell_chord(chord: ChordLabel, key: Key) -> tuple[str, int, str | None, int]:
    """(root step, root alter, bass step, bass alter) of a chord in `key`."""
    t_step, t_alter = _tonic_spelling(key)
    semis = (chord.root - key.tonic) % 12
    root_step, root_alter = _spell_from(t_step, t_alter, semis, _DEGREE_STEPS[semis])
    if abs(root_alter) > 1:  # e.g. a double flat: respell enharmonically
        root_step, root_alter = _spell_from(
            t_step, t_alter, semis, _DEGREE_STEPS[semis] + (1 if root_alter < 0 else -1)
        )
    bass_step, bass_alter = None, 0
    if chord.bass is not None:
        iv = (chord.bass - chord.root) % 12
        steps = _INTERVAL_STEPS[iv]
        if chord.quality == "aug" and iv == 8:
            steps = 4
        bass_step, bass_alter = _spell_from(root_step, root_alter, iv, steps)
    return root_step, root_alter, bass_step, bass_alter


def _accidental_text(alter: int) -> str:
    return {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}[alter]


def chord_symbol(chord: ChordLabel, key: Key, offset: Fraction) -> ChordSymbol:
    q = QUALITY_BY_NAME[chord.quality]
    rs, ra, bs, ba = spell_chord(chord, key)
    label = rs + _accidental_text(ra) + q.suffix
    if bs is not None:
        label += "/" + bs + _accidental_text(ba)
    return ChordSymbol(
        offset=offset,
        root_step=rs,
        root_alter=ra,
        kind=q.kind,
        text=q.suffix,
        bass_step=bs,
        bass_alter=ba,
        label=label,
        degrees=list(q.degrees),
    )


def run_audio_chords(
    audio_path: str, chord_dict: str = "submission"
) -> list[tuple[float, float, str]]:
    """lv-chordia on an audio file → [(start, end, Harte label)].

    Raises ImportError when lv-chordia is not installed.
    """
    import os

    from lv_chordia.chord_recognition import chord_recognition

    result = chord_recognition(os.path.abspath(audio_path), chord_dict)
    return [
        (float(r["start_time"]), float(r["end_time"]), str(r["chord"])) for r in result
    ]
