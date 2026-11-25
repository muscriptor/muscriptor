"""Performance-level cleanup: turn raw transcription output into plausible playing.

Runs on :class:`~muscriptor.sheet.ir.PerformanceNote` (seconds), per
instrument, before anything is snapped to a grid. Each pass targets one kind
of transcription error that would otherwise become clutter on the page:

* duplicates      the same pitch decoded twice a few ms apart
* re-articulation one sustained note split in pieces (vibrato, bow change)
* ghost notes     very short notes, and short octave/12th "harmonic" echoes
                  of a louder note starting at the same time
* range           notes the instrument cannot play: octave errors are folded
                  back into range, bleed from other instruments is dropped
* octave jumps    isolated notes an octave off their melodic context in a
                  monophonic line
* monophony       bass, voice, winds and brass play one note at a time
* polyphony       no more simultaneous notes than the instrument can play
* sparse parts    instruments with only a handful of notes are merged into
                  a part of the same role, or dropped

All passes are pure functions returning new lists, and each logs what it
changed so the effect of the stage is visible in the pipeline log.
"""

from __future__ import annotations

import dataclasses
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass

from muscriptor.sheet.instruments import spec_for
from muscriptor.sheet.ir import PerformanceNote

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CleanupOptions:
    # Same-pitch onsets closer than this are one note decoded twice.
    duplicate_window: float = 0.035
    # Pitched notes shorter than this are dropped (seconds); sustained
    # instruments (voice, strings, winds) use `min_duration_sustained`.
    min_duration: float = 0.045
    min_duration_sustained: float = 0.07
    # A note shorter than this starting with a louder note 12/19/24 semitones
    # below is a harmonic the model heard as a separate note.
    ghost_max_duration: float = 0.15
    # Parts with fewer notes than this are merged into a part with the same
    # role, or dropped when there is none (and they have fewer than
    # `min_part_notes_alone`).
    min_part_notes: int = 40
    min_part_notes_alone: int = 40
    # ... unless it sounds through at least this share of the song.
    min_part_coverage: float = 0.25
    # Simultaneous onsets (for polyphony limits and monophony).
    chord_window: float = 0.04


# Roles written as a single part however many labels the model used.
SINGLE_PART_ROLES = ("bass", "voice")


def _by_instrument(notes: list[PerformanceNote]) -> dict[str, list[PerformanceNote]]:
    groups: dict[str, list[PerformanceNote]] = defaultdict(list)
    for n in notes:
        groups[n.instrument].append(n)
    for g in groups.values():
        g.sort(key=lambda n: (n.onset, n.pitch))
    return groups


def _copy(notes: list[PerformanceNote]) -> list[PerformanceNote]:
    return [dataclasses.replace(n) for n in notes]


# ---------------------------------------------------------------------------
# Instrument consolidation


def _coverage(notes: list[PerformanceNote]) -> dict[str, float]:
    """Share of the song's span during which each instrument sounds."""
    if not notes:
        return {}
    start = min(n.onset for n in notes)
    span = max(n.offset for n in notes) - start
    if span <= 0:
        return {}
    out = {}
    for inst, group in _by_instrument(notes).items():
        total, cur_s, cur_e = 0.0, None, None
        for n in sorted(group, key=lambda n: n.onset):
            if cur_e is None or n.onset > cur_e:
                if cur_e is not None:
                    total += cur_e - cur_s
                cur_s, cur_e = n.onset, n.offset
            else:
                cur_e = max(cur_e, n.offset)
        if cur_e is not None:
            total += cur_e - cur_s
        out[inst] = total / span
    return out


def consolidate_instruments(
    notes: list[PerformanceNote], options: CleanupOptions = CleanupOptions()
) -> list[PerformanceNote]:
    """Merge or drop instruments with too few notes to be a reliable part.

    A transcription model sprinkles a few notes of instruments that are not
    there (a piano chord heard as harp, a voice heard as flute). Writing each
    as a part produces staves of empty bars. A sparse instrument is relabelled
    as the busiest instrument of the same role when one exists — its notes are
    probably real but mislabelled — and dropped otherwise.
    """
    counts = Counter(n.instrument for n in notes)
    by_role: dict[str, list[str]] = defaultdict(list)
    for inst in counts:
        by_role[spec_for(inst).role].append(inst)

    relabel: dict[str, str | None] = {}
    # One bass player, one lead voice: a model alternating between "electric"
    # and "acoustic" bass on the same line is labelling noise, not two parts.
    for role in SINGLE_PART_ROLES:
        members = by_role.get(role, [])
        if len(members) > 1:
            target = max(members, key=lambda p: counts[p])
            for inst in members:
                if inst != target:
                    relabel[inst] = target
                    logger.info(
                        "cleanup: %d %s note(s) merged into %s (one %s part)",
                        counts[inst],
                        inst,
                        target,
                        role,
                    )
            counts[target] = sum(counts[m] for m in members)
            by_role[role] = [target]
    coverage = _coverage(notes)
    for inst, count in counts.items():
        if inst in relabel:
            continue
        # Sparse = few notes *and* little of the song covered: a bass in
        # whole notes or a string pad has few notes but is a real part.
        if (
            count >= options.min_part_notes
            or coverage.get(inst, 0.0) >= options.min_part_coverage
        ):
            continue
        role = spec_for(inst).role
        partners = [
            p
            for p in by_role[role]
            if p != inst and counts[p] >= options.min_part_notes
        ]
        if partners:
            target = max(partners, key=lambda p: counts[p])
            relabel[inst] = target
            logger.info("cleanup: %d %s note(s) merged into %s", count, inst, target)
        elif count < options.min_part_notes_alone:
            relabel[inst] = None
            logger.info(
                "cleanup: dropped %s (%d notes, too sparse for a part)", inst, count
            )

    out = []
    for n in notes:
        target = relabel.get(n.instrument, n.instrument)
        if target is None:
            continue
        out.append(
            dataclasses.replace(n, instrument=target) if target != n.instrument else n
        )
    return out


# ---------------------------------------------------------------------------
# Per-instrument passes


def remove_duplicates(
    notes: list[PerformanceNote], window: float
) -> list[PerformanceNote]:
    """Same pitch, onsets within `window`: keep one, spanning both."""
    by_pitch: dict[int, list[PerformanceNote]] = defaultdict(list)
    for n in notes:
        by_pitch[n.pitch].append(n)
    out = []
    for group in by_pitch.values():
        group.sort(key=lambda n: n.onset)
        kept: list[PerformanceNote] = []
        for n in group:
            if kept and n.onset - kept[-1].onset < window:
                kept[-1].offset = max(kept[-1].offset, n.offset)
                continue
            kept.append(dataclasses.replace(n))
        out.extend(kept)
    return sorted(out, key=lambda n: (n.onset, n.pitch))


def merge_rearticulations(
    notes: list[PerformanceNote], max_gap: float, max_fragment: float = 0.1
) -> list[PerformanceNote]:
    """Glue a sustained note back together where the model split it.

    Only merges when one side of the split is a fragment shorter than
    `max_fragment`: the transcription ends a note exactly where the same
    pitch is attacked again, so two full-length notes back to back are a
    re-strummed chord or a repeated syllable — real notes, kept apart.
    """
    if max_gap <= 0:
        return notes
    by_pitch: dict[int, list[PerformanceNote]] = defaultdict(list)
    for n in notes:
        by_pitch[n.pitch].append(n)
    out = []
    for group in by_pitch.values():
        group.sort(key=lambda n: n.onset)
        kept: list[PerformanceNote] = []
        for n in group:
            if (
                kept
                and n.onset - kept[-1].offset <= max_gap
                and min(n.duration, kept[-1].duration) < max_fragment
            ):
                kept[-1].offset = max(kept[-1].offset, n.offset)
                continue
            kept.append(dataclasses.replace(n))
        out.extend(kept)
    return sorted(out, key=lambda n: (n.onset, n.pitch))


def remove_short_notes(
    notes: list[PerformanceNote], min_duration: float
) -> list[PerformanceNote]:
    return [n for n in notes if n.duration >= min_duration]


def remove_harmonic_ghosts(
    notes: list[PerformanceNote], max_duration: float, window: float
) -> list[PerformanceNote]:
    """Drop short notes that are an overtone of a longer note starting with them."""
    out = []
    for n in notes:
        if n.duration < max_duration:
            is_ghost = any(
                m is not n
                and abs(m.onset - n.onset) <= window
                and (n.pitch - m.pitch) in (12, 19, 24)
                and m.duration >= 2 * n.duration
                for m in notes
                if abs(m.onset - n.onset) <= window
            )
            if is_ghost:
                continue
        out.append(n)
    return out


def fold_into_range(
    notes: list[PerformanceNote], low: int, high: int
) -> list[PerformanceNote]:
    """Fold octave errors back into [low, high]; drop what is far outside.

    A note one octave out is usually the model picking the wrong octave of a
    real note; further out it is bleed from another instrument.
    """
    out = []
    for n in notes:
        p = n.pitch
        if low <= p <= high:
            out.append(n)
        elif low - 12 <= p < low:
            out.append(dataclasses.replace(n, pitch=p + 12))
        elif high < p <= high + 12 and p - 12 >= low:
            # Above range: for low instruments (bass) this is bleed far more
            # often than an octave error, so only fold back for non-bass.
            if spec_for(n.instrument).role != "bass":
                out.append(dataclasses.replace(n, pitch=p - 12))
    return out


def fix_octave_jumps(
    notes: list[PerformanceNote], context: int = 3
) -> list[PerformanceNote]:
    """In a monophonic line, move notes an octave off their neighbours back.

    A note is moved when it is at least 10 semitones from the median of the
    `context` notes on each side, and moving it by an octave brings it within
    7 semitones of that median.
    """
    if len(notes) < 2 * context + 1:
        return notes
    notes = _copy(notes)
    pitches = [n.pitch for n in notes]
    for i, n in enumerate(notes):
        lo, hi = max(0, i - context), min(len(notes), i + context + 1)
        neighbours = sorted(pitches[lo:i] + pitches[i + 1 : hi])
        median = neighbours[len(neighbours) // 2]
        delta = n.pitch - median
        if abs(delta) >= 10:
            moved = n.pitch - 12 * (1 if delta > 0 else -1)
            if abs(moved - median) <= 7:
                n.pitch = moved
    return notes


def enforce_monophony(
    notes: list[PerformanceNote], prefer: str, chord_window: float
) -> list[PerformanceNote]:
    """One note at a time: simultaneous onsets keep one, overlaps are cut.

    `prefer` decides between notes starting together: "low" (bass lines),
    "high" (melodies) or "long".
    """
    if not notes:
        return notes
    notes = sorted(notes, key=lambda n: n.onset)
    clusters: list[list[PerformanceNote]] = [[notes[0]]]
    for n in notes[1:]:
        if n.onset - clusters[-1][0].onset <= chord_window:
            clusters[-1].append(n)
        else:
            clusters.append([n])

    def pick(cluster: list[PerformanceNote]) -> PerformanceNote:
        if prefer == "low":
            return min(cluster, key=lambda n: (n.pitch, -n.duration))
        if prefer == "high":
            return max(cluster, key=lambda n: (n.pitch, n.duration))
        return max(cluster, key=lambda n: n.duration)

    # Three or more different pitches starting together cannot come from one
    # voice / one horn / one bass string: that is a decoding artefact (the
    # model looping on a chord it heard in the stem), not a note to keep.
    plausible = [c for c in clusters if len({n.pitch for n in c}) < 3]
    if len(plausible) < len(clusters):
        logger.info(
            "cleanup: %s: dropped %d chord cluster(s) a monophonic instrument cannot play",
            notes[0].instrument,
            len(clusters) - len(plausible),
        )
    line = [dataclasses.replace(pick(c)) for c in plausible]
    for prev, nxt in zip(line, line[1:]):
        if prev.offset > nxt.onset:
            prev.offset = nxt.onset
    return [n for n in line if n.duration > 0]


def limit_polyphony(
    notes: list[PerformanceNote], max_voices: int, chord_window: float
) -> list[PerformanceNote]:
    """At most `max_voices` notes starting together; keep the outer and longest.

    The outer voices (lowest and highest) carry the bass line and melody, so
    they are kept first, then the longest of the rest.
    """
    if not notes:
        return notes
    notes = sorted(notes, key=lambda n: n.onset)
    out: list[PerformanceNote] = []
    cluster: list[PerformanceNote] = []

    def flush():
        if len(cluster) <= max_voices:
            out.extend(cluster)
            return
        ordered = sorted(cluster, key=lambda n: n.pitch)
        keep = {id(ordered[0]), id(ordered[-1])}
        for n in sorted(ordered[1:-1], key=lambda n: -n.duration):
            if len(keep) >= max_voices:
                break
            keep.add(id(n))
        out.extend(n for n in cluster if id(n) in keep)

    for n in notes:
        if cluster and n.onset - cluster[0].onset > chord_window:
            flush()
            cluster = []
        cluster.append(n)
    flush()
    return out


def clean_drums(notes: list[PerformanceNote], window: float) -> list[PerformanceNote]:
    """Drum hits: one hit per drum per `window` (flams decoded twice)."""
    by_key: dict[int, list[PerformanceNote]] = defaultdict(list)
    for n in notes:
        by_key[n.pitch].append(n)
    out = []
    for group in by_key.values():
        group.sort(key=lambda n: n.onset)
        last = None
        for n in group:
            if last is not None and n.onset - last < window:
                continue
            out.append(n)
            last = n.onset
    return sorted(out, key=lambda n: (n.onset, n.pitch))


# ---------------------------------------------------------------------------


def clean_instrument(
    notes: list[PerformanceNote], options: CleanupOptions = CleanupOptions()
) -> list[PerformanceNote]:
    """All per-instrument passes, for notes of a single instrument."""
    if not notes:
        return notes
    inst = notes[0].instrument
    spec = spec_for(inst)
    if spec.role == "drums":
        return clean_drums(notes, options.duplicate_window)

    sustained = spec.role in (
        "voice",
        "strings",
        "string_section",
        "winds",
        "brass",
        "synth",
    )
    min_dur = options.min_duration_sustained if sustained else options.min_duration

    out = remove_duplicates(notes, options.duplicate_window)
    out = merge_rearticulations(out, spec.merge_gap)
    out = remove_short_notes(out, min_dur)
    out = remove_harmonic_ghosts(out, options.ghost_max_duration, options.chord_window)
    out = fold_into_range(out, spec.low, spec.high)
    if spec.monophonic:
        prefer = (
            "low" if spec.role == "bass" else "high" if spec.role == "voice" else "long"
        )
        out = enforce_monophony(out, prefer, options.chord_window)
        out = fix_octave_jumps(out)
    else:
        out = limit_polyphony(out, spec.max_polyphony, options.chord_window)
    return out


def clean_notes(
    notes: list[PerformanceNote], options: CleanupOptions = CleanupOptions()
) -> list[PerformanceNote]:
    """The whole cleanup stage, for a multi-instrument transcription."""
    before = Counter(n.instrument for n in notes)
    notes = consolidate_instruments(notes, options)
    out: list[PerformanceNote] = []
    for inst, group in sorted(_by_instrument(notes).items()):
        cleaned = clean_instrument(group, options)
        logger.info(
            "cleanup: %-26s %5d → %5d notes",
            inst,
            before.get(inst, len(group)),
            len(cleaned),
        )
        out.extend(cleaned)
    # Cleaning can leave a part too thin to stand on its own: check again.
    out = consolidate_instruments(out, options)
    out.sort(key=lambda n: (n.onset, n.instrument, n.pitch))
    return out
