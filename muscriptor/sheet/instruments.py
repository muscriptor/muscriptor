"""What each transcribed instrument is, as far as notation is concerned.

MuScriptor labels every note with one of its instrument groups
(`muscriptor.tokenizer.mt3.MT3_FULL_PLUS_GROUP_NAMES`). Notation needs more
than a label: which staff layout, which clef, which written octave, whether the
instrument plays one note at a time, and the range outside which a note is far
more likely a transcription error (bleed, octave error) than a real note.

Nothing here invents instruments: a part only exists if the transcription
produced notes for it. Groups the table does not know fall back to a generic
treble/grand-staff part named after the group.
"""

from dataclasses import dataclass
from typing import Literal

Role = Literal[
    "keyboard",
    "guitar",
    "bass",
    "strings",
    "string_section",
    "voice",
    "winds",
    "brass",
    "synth",
    "drums",
    "percussion",
    "other",
]

StaffLayout = Literal[
    "grand", "treble", "treble_8vb", "bass", "bass_8vb", "alto", "percussion"
]


@dataclass(frozen=True)
class InstrumentSpec:
    group: str  # MuScriptor group name
    name: str  # printed part name
    abbreviation: str
    role: Role
    layout: StaffLayout
    gm_program: int  # 0-based General MIDI program (ignored for drums)
    low: int  # lowest plausible sounding MIDI pitch
    high: int  # highest plausible sounding MIDI pitch
    monophonic: bool = False
    max_polyphony: int = 10
    # Re-articulations of the same pitch closer than this (seconds) are one
    # sustained note that the model split (vibrato, bowing, breath).
    merge_gap: float = 0.03
    # Order of parts in a score, top to bottom (conventional score order:
    # vocals, winds, brass, keyboards/guitars, strings, bass, drums).
    order: int = 50


_SPECS = [
    InstrumentSpec(
        "voice", "Voice", "Vo.", "voice", "treble", 52, 40, 84, True, 1, 0.08, 0
    ),
    InstrumentSpec(
        "flutes", "Flute", "Fl.", "winds", "treble", 73, 59, 98, True, 1, 0.06, 10
    ),
    InstrumentSpec(
        "oboe", "Oboe", "Ob.", "winds", "treble", 68, 58, 93, True, 1, 0.06, 11
    ),
    InstrumentSpec(
        "english_horn",
        "English Horn",
        "E.H.",
        "winds",
        "treble",
        69,
        52,
        84,
        True,
        1,
        0.06,
        12,
    ),
    InstrumentSpec(
        "clarinet", "Clarinet", "Cl.", "winds", "treble", 71, 50, 94, True, 1, 0.06, 13
    ),
    InstrumentSpec(
        "bassoon", "Bassoon", "Bsn.", "winds", "bass", 70, 34, 77, True, 1, 0.06, 14
    ),
    InstrumentSpec(
        "soprano_and_alto_sax",
        "Alto Saxophone",
        "A.Sx.",
        "winds",
        "treble",
        65,
        49,
        84,
        True,
        1,
        0.06,
        15,
    ),
    InstrumentSpec(
        "tenor_sax",
        "Tenor Saxophone",
        "T.Sx.",
        "winds",
        "treble_8vb",
        66,
        44,
        79,
        True,
        1,
        0.06,
        16,
    ),
    InstrumentSpec(
        "baritone_sax",
        "Baritone Saxophone",
        "B.Sx.",
        "winds",
        "bass",
        67,
        36,
        72,
        True,
        1,
        0.06,
        17,
    ),
    InstrumentSpec(
        "trumpet", "Trumpet", "Tpt.", "brass", "treble", 56, 54, 86, True, 1, 0.06, 20
    ),
    InstrumentSpec(
        "french_horn", "Horn", "Hn.", "brass", "treble", 60, 34, 77, True, 1, 0.06, 21
    ),
    InstrumentSpec(
        "trombone", "Trombone", "Tbn.", "brass", "bass", 57, 34, 75, True, 1, 0.06, 22
    ),
    InstrumentSpec(
        "tuba", "Tuba", "Tba.", "brass", "bass", 58, 26, 65, True, 1, 0.06, 23
    ),
    InstrumentSpec(
        "brass_section",
        "Brass",
        "Br.",
        "brass",
        "grand",
        61,
        34,
        86,
        False,
        4,
        0.06,
        24,
    ),
    InstrumentSpec(
        "synth_lead",
        "Synth Lead",
        "Syn.",
        "synth",
        "treble",
        80,
        36,
        96,
        True,
        1,
        0.05,
        25,
    ),
    InstrumentSpec(
        "acoustic_piano",
        "Piano",
        "Pno.",
        "keyboard",
        "grand",
        0,
        21,
        108,
        False,
        10,
        0.02,
        30,
    ),
    InstrumentSpec(
        "electric_piano",
        "Electric Piano",
        "E.Pno.",
        "keyboard",
        "grand",
        4,
        28,
        103,
        False,
        10,
        0.02,
        31,
    ),
    InstrumentSpec(
        "organ", "Organ", "Org.", "keyboard", "grand", 16, 24, 96, False, 10, 0.05, 32
    ),
    InstrumentSpec(
        "chromatic_percussion",
        "Mallets",
        "Mal.",
        "keyboard",
        "grand",
        11,
        41,
        108,
        False,
        4,
        0.02,
        33,
    ),
    InstrumentSpec(
        "orchestral_harp",
        "Harp",
        "Hp.",
        "keyboard",
        "grand",
        46,
        24,
        103,
        False,
        8,
        0.02,
        34,
    ),
    InstrumentSpec(
        "synth_pad",
        "Synth Pad",
        "Pad",
        "synth",
        "grand",
        88,
        28,
        96,
        False,
        8,
        0.06,
        35,
    ),
    InstrumentSpec(
        "acoustic_guitar",
        "Acoustic Guitar",
        "A.Gtr.",
        "guitar",
        "treble_8vb",
        25,
        40,
        88,
        False,
        6,
        0.02,
        40,
    ),
    InstrumentSpec(
        "clean_electric_guitar",
        "Electric Guitar",
        "E.Gtr.",
        "guitar",
        "treble_8vb",
        27,
        40,
        88,
        False,
        6,
        0.02,
        41,
    ),
    InstrumentSpec(
        "distorted_electric_guitar",
        "Distortion Guitar",
        "D.Gtr.",
        "guitar",
        "treble_8vb",
        30,
        40,
        88,
        False,
        6,
        0.02,
        42,
    ),
    InstrumentSpec(
        "violin", "Violin", "Vln.", "strings", "treble", 40, 55, 100, False, 2, 0.08, 50
    ),
    InstrumentSpec(
        "viola", "Viola", "Vla.", "strings", "alto", 41, 48, 88, False, 2, 0.08, 51
    ),
    InstrumentSpec(
        "cello", "Violoncello", "Vc.", "strings", "bass", 42, 36, 81, False, 2, 0.08, 52
    ),
    InstrumentSpec(
        "string_ensemble",
        "Strings",
        "Str.",
        "string_section",
        "grand",
        48,
        28,
        100,
        False,
        6,
        0.08,
        53,
    ),
    InstrumentSpec(
        "synth_strings",
        "Synth Strings",
        "Syn.Str.",
        "string_section",
        "grand",
        50,
        28,
        100,
        False,
        6,
        0.08,
        54,
    ),
    InstrumentSpec(
        "orchestra_hit",
        "Orchestra Hit",
        "Hit",
        "other",
        "grand",
        55,
        28,
        100,
        False,
        6,
        0.02,
        55,
    ),
    InstrumentSpec(
        "timpani",
        "Timpani",
        "Timp.",
        "percussion",
        "bass",
        47,
        36,
        60,
        False,
        2,
        0.02,
        60,
    ),
    InstrumentSpec(
        "contrabass",
        "Contrabass",
        "Cb.",
        "bass",
        "bass_8vb",
        43,
        23,
        67,
        True,
        1,
        0.08,
        70,
    ),
    InstrumentSpec(
        "acoustic_bass",
        "Acoustic Bass",
        "A.Bs.",
        "bass",
        "bass_8vb",
        32,
        23,
        67,
        True,
        1,
        0.04,
        71,
    ),
    InstrumentSpec(
        "electric_bass",
        "Electric Bass",
        "E.Bs.",
        "bass",
        "bass_8vb",
        33,
        23,
        67,
        True,
        1,
        0.04,
        72,
    ),
    InstrumentSpec(
        "drums", "Drum Set", "Dr.", "drums", "percussion", 0, 0, 127, False, 4, 0.0, 90
    ),
]

SPECS: dict[str, InstrumentSpec] = {s.group: s for s in _SPECS}


def spec_for(group: str) -> InstrumentSpec:
    """The spec for a MuScriptor group, or a generic polyphonic one."""
    if group in SPECS:
        return SPECS[group]
    pretty = group.replace("_", " ").title()
    return InstrumentSpec(
        group,
        pretty,
        pretty[:4] + ".",
        "other",
        "grand",
        0,
        21,
        108,
        False,
        8,
        0.03,
        80,
    )


_HIGH_STRINGS = ("violin", "viola", "string_ensemble", "synth_strings")
_LOW_STRINGS = ("cello", "contrabass")
_STRINGS = ("violin", "viola", "cello", "string_ensemble", "synth_strings")
_WOODWINDS = (
    "flutes",
    "oboe",
    "english_horn",
    "clarinet",
    "bassoon",
    "soprano_and_alto_sax",
    "tenor_sax",
    "baritone_sax",
)
_BRASS = ("trumpet", "french_horn", "trombone", "tuba", "brass_section")

# Separated stems → the MuScriptor groups that may be decoded from them. Used
# to constrain transcription of a stem (so the bass stem can only yield bass
# notes) and to decide which notes of the full-mix transcription a stem
# transcription replaces.
#
# The 6-stem separators (htdemucs_6s, BS-Roformer-SW) leave strings, winds,
# brass and synths in "other". Constraining it to exactly those families is
# what keeps sustained strings from being labelled piano: with piano and
# guitar forbidden, the model must name the bowed line for what it is.
# Organ is left out on purpose: allowed, it is what the model reaches for
# next on a sustained string pad (on a piano + strings ballad it took 92% of
# the "other" stem), and the separators put keyboards in the piano stem.
# Orchestra hit is left out because conditioning on it derails decoding.
# The remaining names are stems a second-pass (refine) model may produce.
STEM_GROUPS: dict[str, tuple[str, ...]] = {
    "vocals": ("voice",),
    "bass": ("electric_bass", "acoustic_bass", "contrabass"),
    "drums": ("drums",),
    "piano": ("acoustic_piano", "electric_piano", "organ"),
    "guitar": ("acoustic_guitar", "clean_electric_guitar", "distorted_electric_guitar"),
    "other": _STRINGS
    + _WOODWINDS
    + _BRASS
    + (
        "orchestral_harp",
        "chromatic_percussion",
        "synth_lead",
        "synth_pad",
    ),
    "harmonic_other": _STRINGS
    + _WOODWINDS
    + _BRASS
    + ("orchestral_harp", "synth_lead", "synth_pad"),
    "strings": _STRINGS + ("contrabass",),
    "high_strings": _HIGH_STRINGS,
    "violin": ("violin", "viola", "string_ensemble"),
    "low_strings": _LOW_STRINGS,
    "cello": _LOW_STRINGS,
    "winds": _WOODWINDS,
    "flute": ("flutes",),
    "brass": _BRASS,
    "organ": ("organ",),
    "synth": ("synth_lead", "synth_pad", "synth_strings"),
}


# General MIDI percussion (channel 10) → how a drum set staff draws it.
# (display step, display octave, notehead, voice: 1 = hands/stems up,
#  2 = feet/stems down, printed name). Positions follow the common
# MuseScore / PAS drum-set legend.
@dataclass(frozen=True)
class DrumSpec:
    step: str
    octave: int
    notehead: str  # MusicXML notehead value
    voice: int
    name: str


DRUM_MAP: dict[int, DrumSpec] = {
    35: DrumSpec("E", 4, "normal", 2, "Acoustic Bass Drum"),
    36: DrumSpec("F", 4, "normal", 2, "Bass Drum"),
    37: DrumSpec("C", 5, "x", 1, "Side Stick"),
    38: DrumSpec("C", 5, "normal", 1, "Snare"),
    39: DrumSpec("C", 5, "x", 1, "Hand Clap"),
    40: DrumSpec("C", 5, "normal", 1, "Electric Snare"),
    41: DrumSpec("G", 4, "normal", 1, "Low Floor Tom"),
    42: DrumSpec("G", 5, "x", 1, "Closed Hi-Hat"),
    43: DrumSpec("A", 4, "normal", 1, "High Floor Tom"),
    44: DrumSpec("D", 4, "x", 2, "Pedal Hi-Hat"),
    45: DrumSpec("B", 4, "normal", 1, "Low Tom"),
    46: DrumSpec("G", 5, "circle-x", 1, "Open Hi-Hat"),
    47: DrumSpec("D", 5, "normal", 1, "Low-Mid Tom"),
    48: DrumSpec("E", 5, "normal", 1, "Hi-Mid Tom"),
    49: DrumSpec("A", 5, "x", 1, "Crash Cymbal"),
    50: DrumSpec("F", 5, "normal", 1, "High Tom"),
    51: DrumSpec("F", 5, "x", 1, "Ride Cymbal"),
    52: DrumSpec("A", 5, "x", 1, "Chinese Cymbal"),
    53: DrumSpec("F", 5, "diamond", 1, "Ride Bell"),
    54: DrumSpec("E", 5, "x", 1, "Tambourine"),
    55: DrumSpec("B", 5, "x", 1, "Splash Cymbal"),
    56: DrumSpec("F", 5, "triangle", 1, "Cowbell"),
    57: DrumSpec("A", 5, "x", 1, "Crash Cymbal 2"),
    59: DrumSpec("F", 5, "x", 1, "Ride Cymbal 2"),
}

# Anything else the model emits is drawn as the nearest family member.
_DRUM_FALLBACK = {
    "kick": 36,
    "snare": 38,
    "hat": 42,
    "tom": 45,
    "cymbal": 49,
}


def drum_spec(pitch: int) -> tuple[int, DrumSpec]:
    """(GM pitch actually drawn, spec) for a drum hit at `pitch`."""
    if pitch in DRUM_MAP:
        return pitch, DRUM_MAP[pitch]
    if pitch < 35:
        key = "kick"
    elif pitch <= 40:
        key = "snare"
    elif pitch <= 50:
        key = "tom"
    else:
        key = "cymbal"
    mapped = _DRUM_FALLBACK[key]
    return mapped, DRUM_MAP[mapped]
