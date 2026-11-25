"""Data model shared by the sheet-music stages.

Two levels, deliberately kept apart:

* :class:`PerformanceNote` — what was *played*: seconds, MIDI pitch, the
  transcribed instrument. Everything up to and including cleanup works on
  these.
* :class:`Score` (the Notation IR) — what is *written*: measures, staves,
  voices, durations as exact fractions of a quarter note, spelled pitches,
  ties, tuplets, chord symbols, key/time signatures and tempo marks. The
  MusicXML writer is a pure function of a Score, so everything that decides
  how the music looks happens before it and can be tested on the IR alone.

All IR times are `fractions.Fraction` in quarter notes (so a 4/4 bar is 4 and
an eighth-note triplet is 1/3): exact arithmetic means a measure's durations
sum to exactly its length, which MusicXML requires and float rounding breaks.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Performance level (seconds)


@dataclass
class PerformanceNote:
    pitch: int  # MIDI pitch; GM drum key for drums
    onset: float  # seconds
    offset: float  # seconds
    instrument: str  # MuScriptor group name, "drums" for drums
    velocity: int = 80
    # Where the note came from: "mix" or the stem name ("vocals", "bass", ...).
    source: str = "mix"

    @property
    def is_drum(self) -> bool:
        return self.instrument == "drums"

    @property
    def duration(self) -> float:
        return self.offset - self.onset


def save_performance(notes: list[PerformanceNote], path: Path) -> None:
    path.write_text(
        json.dumps([dataclasses.asdict(n) for n in notes], indent=0), encoding="utf-8"
    )


def load_performance(path: Path) -> list[PerformanceNote]:
    return [PerformanceNote(**d) for d in json.loads(path.read_text(encoding="utf-8"))]


# ---------------------------------------------------------------------------
# Beat level: quantized notes, still unsplit into measures


@dataclass
class QuantizedNote:
    """A note snapped to the beat grid; times in quarter notes from bar 1."""

    pitch: int
    onset: Fraction
    offset: Fraction
    instrument: str
    velocity: int = 80
    # Steps per beat of the grid the onset was snapped to (2, 3, 4, 6, ...).
    grid: int = 4
    staff: int = 1
    voice: int = 1

    @property
    def duration(self) -> Fraction:
        return self.offset - self.onset


# ---------------------------------------------------------------------------
# Notation IR


@dataclass
class Pitch:
    step: str  # "C".."B"
    alter: int  # -2..2
    octave: int
    midi: int
    # Display accidental, decided per measure ("sharp", "flat", "natural", ...)
    accidental: str | None = None


@dataclass
class DrumHead:
    """An unpitched note on a percussion staff."""

    midi: int  # GM drum key (instrument id)
    step: str
    octave: int
    notehead: str


@dataclass
class Tuplet:
    actual: int  # e.g. 3 notes ...
    normal: int  # ... in the time of 2
    start: bool = False
    stop: bool = False


@dataclass
class Event:
    """A note, chord or rest in one voice. `pitches` empty = rest."""

    offset: Fraction  # from the start of the measure
    duration: Fraction  # sounding length in quarter notes (tuplet-adjusted)
    pitches: list[Pitch] = field(default_factory=list)
    drums: list[DrumHead] = field(default_factory=list)
    tie_start: bool = False
    tie_stop: bool = False
    tuplet: Tuplet | None = None
    # A whole-measure rest (written without a type, centred in the bar).
    measure_rest: bool = False
    # Invisible spacer rest (fills a voice that only plays part of a measure).
    hidden: bool = False

    @property
    def is_rest(self) -> bool:
        return not self.pitches and not self.drums


@dataclass
class Voice:
    number: int  # MusicXML voice number, unique within the part
    staff: int  # 1-based
    events: list[Event] = field(default_factory=list)


@dataclass
class ChordSymbol:
    offset: Fraction  # from the start of the measure
    root_step: str
    root_alter: int
    kind: str  # MusicXML <kind> value, e.g. "minor-seventh"
    text: str  # display suffix, e.g. "m7"
    bass_step: str | None = None
    bass_alter: int = 0
    label: str = ""  # full printed symbol, e.g. "Am7/G"
    confidence: float = 1.0
    # Added/altered degrees for kinds MusicXML has no word for:
    # (degree value, alter, "add" | "alter" | "subtract"), e.g. add9 = (9, 0, "add").
    degrees: list[tuple[int, int, str]] = field(default_factory=list)


@dataclass
class Clef:
    sign: str  # "G", "F", "C", "percussion"
    line: int
    octave_change: int = 0


@dataclass
class Measure:
    number: int
    length: Fraction  # quarter notes
    time_signature: tuple[int, int] | None = None  # only where it changes
    key_fifths: int | None = None  # only where it changes
    key_mode: str = "major"
    tempo: float | None = None  # metronome mark, only where it changes
    voices: list[Voice] = field(default_factory=list)
    chords: list[ChordSymbol] = field(default_factory=list)
    rehearsal: str | None = None
    # Free text above the staff (e.g. "Swing").
    words: str | None = None
    # Double bar line at the end (section boundary).
    double_bar: bool = False
    # Recording time of the downbeat, for cross-referencing with the audio.
    start_seconds: float | None = None


@dataclass
class Part:
    id: str
    name: str
    abbreviation: str
    instrument: str  # MuScriptor group
    role: str
    clefs: list[Clef]
    gm_program: int  # 0-based
    is_drum: bool = False
    # Written pitch = sounding + 12 * written_octave_shift (guitar, bass: +1).
    written_octave_shift: int = 0
    measures: list[Measure] = field(default_factory=list)
    # GM drum keys used, for the percussion instrument list.
    drum_keys: list[int] = field(default_factory=list)

    @property
    def staves(self) -> int:
        return len(self.clefs)


@dataclass
class Score:
    title: str
    parts: list[Part]
    composer: str | None = None
    source: str | None = None
    # Free-form provenance and analysis summary (key, tempo, swing...).
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# JSON (debug artifact). Fractions are written as "n/d" strings.


def _encode(obj: Any) -> Any:
    if isinstance(obj, Fraction):
        return str(obj)
    if dataclasses.is_dataclass(obj):
        return {k: _encode(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    return obj


def score_to_json(score: Score) -> str:
    # asdict recurses and turns nested dataclasses into dicts but keeps the
    # Fractions, which _encode then stringifies.
    return json.dumps(_encode(dataclasses.asdict(score)), indent=1, ensure_ascii=False)
