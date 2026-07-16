"""MT3 MIDI tokenizer.

Adapted from YourMT3+ (https://github.com/mimbres/YourMT3) and audiocraft_trans.
"""

import difflib
import logging
from collections.abc import Iterable

from muscriptor.tokenizer.notes import (
    DRUM_PROGRAM,
    SPECIAL_TOKENS,
    build_event_vocab,
)

logger = logging.getLogger(__name__)


def get_group_program_map(
    instrument_vocabulary: str,
    misc_programs: str,
    is_mt3: bool = False,
    include_drums: bool = False,
) -> dict[int, list[int]]:
    if instrument_vocabulary == "ONLY_PIANO":
        ret = {0: list(range(128))}
    elif instrument_vocabulary == "FULL":
        ret = {i: [i] for i in range(128)}
    elif instrument_vocabulary == "MT3_MIDI_PLUS":
        ret = {
            0: list(range(8)),
            1: list(range(8, 16)),
            2: list(range(16, 24)),
            3: list(range(24, 32)),
            4: list(range(32, 40)),
            5: list(range(40, 56)),
            6: list(range(56, 64)),
            7: list(range(64, 72)),
            8: list(range(72, 80)),
            9: list(range(80, 88)),
            10: list(range(88, 96)),
            11: list(range(100, 102)),
        }
    elif instrument_vocabulary == "MT3_FULL_PLUS":
        ret = {
            0: [0, 1, 3, 6, 7],
            1: [2, 4, 5],
            2: list(range(8, 16)),
            3: list(range(16, 24)),
            4: [24, 25],
            5: [26, 27, 28],
            6: [29, 30, 31],
            7: [32, 35],
            8: [33, 34, 36, 37, 38, 39],
            9: [40],
            10: [41],
            11: [42],
            12: [43],
            13: [46],
            14: [47],
            15: [48, 49, 44, 45],
            16: [50, 51],
            17: [52, 53, 54],
            18: [55],
            19: [56, 59],
            20: [57],
            21: [58],
            22: [60],
            23: [61, 62, 63],
            24: [64, 65],
            25: [66],
            26: [67],
            27: [68],
            28: [69],
            29: [70],
            30: [71],
            31: list(range(72, 80)),
            32: list(range(80, 88)),
            33: list(range(88, 96)),
            34: [100],
            35: [101],
        }
    elif instrument_vocabulary == "OURS_INSTRUMENT_GROUPS":
        ret = {
            0: list(range(8)),
            1: list(range(24, 32)),
            2: list(range(32, 40)),
            3: list(range(40, 56)),
            4: list(range(56, 64)),
            5: list(range(16, 24)) + list(range(64, 80)),
            6: list(range(80, 96)),
            7: list(range(8, 16)) + list(range(112, 119)),
        }
    else:
        assert False, instrument_vocabulary

    if instrument_vocabulary == "MT3_FULL_PLUS" and not is_mt3:
        not_assigned = set(range(130)) - set([v for vs in ret.values() for v in vs])
    else:
        not_assigned = set(range(128)) - set([v for vs in ret.values() for v in vs])
    if include_drums:
        not_assigned = not_assigned.union({DRUM_PROGRAM})
    if misc_programs == "ONE_GROUP":
        ret[len(ret)] = list(not_assigned)
    elif misc_programs == "SINGLETON_GROUPS":
        for p in not_assigned:
            ret[len(ret)] = [p]
    else:
        assert misc_programs == "OMIT", misc_programs
    return ret


# Human-readable names for the MT3_FULL_PLUS instrument groups (see
# get_group_program_map). Used by the CLI's --instruments option and the
# web app's /instruments endpoint. The group IDs index the model's learned
# program groups and must not change; only the user-facing names do.
# Notes the model still decodes into an omitted group surface as "program_<n>".
MT3_FULL_PLUS_GROUP_NAMES: dict[str, int] = {
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


def instrument_group_from_names(names: Iterable[str]) -> str:
    """Map exact instrument group names to the model's conditioning string.

    The strict counterpart of :func:`resolve_instrument_names`: every name
    must appear verbatim in ``MT3_FULL_PLUS_GROUP_NAMES``. Raises ValueError
    listing the unknown names otherwise.
    """
    names = list(names)
    unknown = [n for n in names if n not in MT3_FULL_PLUS_GROUP_NAMES]
    if unknown:
        raise ValueError(
            f"unknown instrument name(s): {', '.join(map(repr, unknown))}; "
            f"valid names: {', '.join(MT3_FULL_PLUS_GROUP_NAMES)}"
        )
    return " ".join(str(MT3_FULL_PLUS_GROUP_NAMES[n]) for n in names)


def resolve_instrument_names(tokens: Iterable[str]) -> list[str]:
    """Resolve loosely-typed instrument tokens to canonical group names.

    Matching is case-insensitive; a token that is not an exact name may be
    any substring that matches exactly one group name (``"timp"`` →
    ``"timpani"``). Raises ValueError when a token is ambiguous (listing the
    candidates) or matches nothing (suggesting close spellings).
    """
    resolved = []
    for token in tokens:
        t = token.strip().lower()
        if t in MT3_FULL_PLUS_GROUP_NAMES:
            resolved.append(t)
            continue
        hits = [n for n in MT3_FULL_PLUS_GROUP_NAMES if t in n]
        if len(hits) == 1:
            resolved.append(hits[0])
        elif hits:
            raise ValueError(
                f"ambiguous instrument name {token!r}: "
                f"matches {', '.join(hits)}"
            )
        else:
            # Compare against each name AND its underscore-separated words,
            # so a typo like "pinao" still surfaces "acoustic_piano".
            def closeness(name: str) -> float:
                return max(
                    difflib.SequenceMatcher(None, t, part).ratio()
                    for part in (name, *name.split("_"))
                )

            ranked = sorted(MT3_FULL_PLUS_GROUP_NAMES, key=closeness, reverse=True)
            suggestions = [n for n in ranked[:3] if closeness(n) >= 0.6]
            hint = (
                f" — did you mean {', '.join(suggestions)}?"
                if suggestions
                else ""
            )
            raise ValueError(f"unknown instrument name {token!r}{hint}")
    return resolved


class MT3Tokenizer:
    def __init__(
        self,
        instrument_vocabulary: str = "FULL",
        max_shift_steps: int = 1001,
        frame_rate: int = 100,
    ):
        self.group_program_map = get_group_program_map(
            instrument_vocabulary, misc_programs="SINGLETON_GROUPS", is_mt3=True
        )
        self.frame_rate = frame_rate
        self._vocab = build_event_vocab(max_shift_steps)
        self._token_index = {(e.type, e.value): i for i, e in enumerate(self._vocab)}
        self.num_tokens = len(self._vocab)
        self.eos_id = SPECIAL_TOKENS.index("EOS")

        logger.info(f"MT3Tokenizer: {self.num_tokens} tokens")

    def tie_section_token_ids(
        self, open_note_keys: Iterable[tuple[int, int]]
    ) -> list[int]:
        """Encode a tie prologue declaring ``open_note_keys`` as sustained.

        ``open_note_keys`` are the ``(program, pitch)`` pairs of notes still
        sounding at a chunk boundary. The layout matches the training encoder
        (``note_event2event``): pairs sorted by (program, pitch), each program
        token emitted once for its run of pitches, terminated by the ``tie``
        token. Teacher-forcing these as the start of a chunk pins the model's
        tie section to the notes actually sustained from the previous chunk.
        """
        tokens: list[int] = []
        program_state: int | None = None
        for program, pitch in sorted(open_note_keys):
            if program != program_state:
                tokens.append(self._token_index[("program", program)])
                program_state = program
            tokens.append(self._token_index[("pitch", pitch)])
        tokens.append(self._token_index[("tie", 0)])
        return tokens

    def forbidden_token_ids(self, instruments: Iterable[str]) -> list[int]:
        """Token ids that must never be sampled when only ``instruments`` may
        appear in the transcription (the hard counterpart of the advisory
        instrument_group conditioning).

        ``instruments`` are exact MT3_FULL_PLUS group names (so this only makes
        sense on a tokenizer built with that vocabulary). A ``program`` token is
        forbidden unless it decodes to one of the given groups — i.e. it is the
        representative (first) program of an allowed group; ``drum`` tokens are
        forbidden unless "drums" is listed. Timing, pitch, velocity, tie and
        special tokens are never forbidden. Raises ValueError on unknown names.
        """
        names = list(instruments)
        unknown = [n for n in names if n not in MT3_FULL_PLUS_GROUP_NAMES]
        if unknown:
            raise ValueError(
                f"unknown instrument name(s): {', '.join(map(repr, unknown))}; "
                f"valid names: {', '.join(MT3_FULL_PLUS_GROUP_NAMES)}"
            )
        allow_drums = "drums" in names
        # Same representative-program convention as decoding
        # (transcription_model._build_instrument_for_program): the model emits
        # the first program of a group, so only that program is allowed.
        allowed_programs = set()
        for name in names:
            if name == "drums":
                continue
            gid = MT3_FULL_PLUS_GROUP_NAMES[name]
            if gid in self.group_program_map and self.group_program_map[gid]:
                allowed_programs.add(self.group_program_map[gid][0])
        forbidden = []
        for token_id, event in enumerate(self._vocab):
            if event.type == "program" and event.value not in allowed_programs:
                forbidden.append(token_id)
            elif event.type == "drum" and not allow_drums:
                forbidden.append(token_id)
        return forbidden
