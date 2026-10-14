"""Note and NoteEvent types, tokenizer utilities.

Adapted from YourMT3+ (https://github.com/mimbres/YourMT3).
"""

import os
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from mido import Message, MetaMessage, MidiFile, MidiTrack, second2tick

from muscriptor.utils.beats import BAR_OFFSET_MARKER

DRUM_PROGRAM = 128
MINIMUM_NOTE_DURATION_SEC = 0.01


@dataclass
class Note:
    is_drum: bool
    program: int  # MIDI program number (0-127); 128 for drum
    onset: float  # onset time in seconds
    offset: float  # offset time in seconds (== onset for drums)
    pitch: int  # MIDI note number (0-127)


@dataclass
class NoteEvent:
    is_drum: bool
    program: int  # [0, 127], 128 for drum (ignored in tokenizer)
    time: float  # absolute time in seconds
    velocity: int  # 1 for onset, 0 for offset; drum has no offset
    pitch: int  # MIDI pitch


@dataclass
class TieNoteEvent:
    program: int  # [0, 127], 128 for drum (ignored in tokenizer)
    pitch: int  # MIDI pitch


@dataclass
class EventRange:
    type: str
    min_value: int
    max_value: int  # inclusive


@dataclass
class Event:
    type: str
    value: int


def sort_notes(notes: list[Note]):
    if len(notes) > 0:
        notes.sort(key=lambda n: (n.onset, n.is_drum, n.program, n.pitch, n.offset))


def sort_note_events(note_events: list[NoteEvent]):
    if len(note_events) > 0:
        note_events.sort(
            key=lambda n: (n.time, n.is_drum, n.program, n.velocity, n.pitch)
        )


def sort_tie_note_events(tie_note_events: list[TieNoteEvent]):
    if len(tie_note_events) > 0:
        tie_note_events.sort(key=lambda n: (n.program, n.pitch))


def validate_notes(
    notes: list[Note],
    minimum_offset: float | None = MINIMUM_NOTE_DURATION_SEC,
    fix: bool = True,
) -> list[Note]:
    if len(notes) > 0:
        for note in list(notes):
            if note.onset is None and fix:
                notes.remove(note)
            elif note.offset is None and fix:
                note.offset = note.onset + minimum_offset
            elif note.onset > note.offset:
                if fix:
                    note.offset = max(note.offset, note.onset + minimum_offset)
            elif note.is_drum is False and note.offset - note.onset < 0.01 and fix:
                note.offset = note.onset + minimum_offset
    return notes


def trim_overlapping_notes(notes: list[Note], sort: bool = True) -> list[Note]:
    if len(notes) <= 1:
        return notes
    trimmed_notes = []
    channels = set((note.program, note.pitch, note.is_drum) for note in notes)
    for program, pitch, is_drum in channels:
        channel_notes = [
            n
            for n in notes
            if n.pitch == pitch and n.program == program and n.is_drum == is_drum
        ]
        sorted_notes = sorted(channel_notes, key=lambda n: n.onset)
        for i in range(1, len(sorted_notes)):
            if sorted_notes[i - 1].offset > sorted_notes[i].onset:
                sorted_notes[i - 1].offset = sorted_notes[i].onset
        valid_notes = [n for n in sorted_notes if n.onset < n.offset]
        trimmed_notes.extend(valid_notes)
    if sort:
        sort_notes(trimmed_notes)
    return trimmed_notes


# Special tokens occupy the first indices of the vocabulary, in this order.
SPECIAL_TOKENS = ("PAD", "EOS", "UNK")


def build_event_vocab(max_shift_steps: int) -> list[Event]:
    """Return the token-index → :class:`Event` decode table.

    Index ``i`` maps to the event the model emits at token ``i``. The layout
    is fixed: special tokens, then ``shift``, then the note-event ranges.
    """
    ranges = (
        [EventRange(token, 0, 0) for token in SPECIAL_TOKENS]
        + [EventRange("shift", 0, max_shift_steps - 1)]
        + [
            EventRange("pitch", 0, 127),
            EventRange("velocity", 0, 1),
            EventRange("tie", 0, 0),
            EventRange("program", 0, 129),
            EventRange("drum", 0, 127),
        ]
    )
    vocab: list[Event] = []
    for er in ranges:
        for value in range(er.min_value, er.max_value + 1):
            vocab.append(Event(type=er.type, value=value))
    return vocab


def note_event2note(
    note_events: list[NoteEvent],
    tie_note_events: list[TieNoteEvent] | None = None,
    shorten_notes_above_n_sec: int = 10,
    fix_broken_notes: bool = True,
    trim_overlap: bool = True,
    force_offset_past_segment_end: float | None = None,
    force_onset_before_segment_start: float | None = None,
) -> tuple[list[Note], Counter]:
    notes: list[Note] = []
    active_note_events: dict[tuple[int, int], NoteEvent | TieNoteEvent] = {}
    err_cnt: Counter = Counter()

    if tie_note_events is not None:
        for ne in tie_note_events:
            active_note_events[(ne.program, ne.pitch)] = ne

    sort_note_events(note_events)
    for ne in note_events:
        try:
            if ne.time is None:
                continue
            elif ne.is_drum:
                if ne.velocity == 1:
                    notes.append(
                        Note(
                            is_drum=True,
                            program=DRUM_PROGRAM,
                            onset=ne.time,
                            offset=ne.time + MINIMUM_NOTE_DURATION_SEC,
                            pitch=ne.pitch,
                        )
                    )
                else:
                    continue
            else:
                active_ne = active_note_events.pop((ne.program, ne.pitch), None)
                if ne.velocity == 0 and active_ne is None:
                    raise ValueError("Err/onset not found")
                if active_ne is not None:
                    if type(active_ne) is NoteEvent:
                        notes.append(
                            Note(
                                is_drum=False,
                                program=active_ne.program,
                                onset=active_ne.time,
                                offset=ne.time,
                                pitch=active_ne.pitch,
                            )
                        )
                    else:  # TieNoteEvent
                        notes.append(
                            Note(
                                is_drum=False,
                                program=active_ne.program,
                                onset=force_onset_before_segment_start,
                                offset=ne.time,
                                pitch=active_ne.pitch,
                            )
                        )
                if ne.velocity == 1:
                    active_note_events[(ne.program, ne.pitch)] = ne
        except ValueError as ve:
            err_cnt[str(ve)] += 1

    for ne in active_note_events.values():
        try:
            if type(ne) is NoteEvent and ne.velocity == 1:
                if ne.program is None or ne.pitch is None:
                    raise ValueError("Err/active ne incomplete")
                elif ne.time is None:
                    continue
                else:
                    notes.append(
                        Note(
                            is_drum=False,
                            program=ne.program,
                            onset=ne.time,
                            offset=ne.time + MINIMUM_NOTE_DURATION_SEC
                            if force_offset_past_segment_end is None
                            else force_offset_past_segment_end,
                            pitch=ne.pitch,
                        )
                    )
        except ValueError as ve:
            err_cnt[str(ve)] += 1

    if shorten_notes_above_n_sec > 0:
        for n in list(notes):
            try:
                if n.offset - n.onset > shorten_notes_above_n_sec:
                    n.offset = n.onset + MINIMUM_NOTE_DURATION_SEC
                    raise ValueError(f"Err/long note > {shorten_notes_above_n_sec}s")
            except ValueError as ve:
                err_cnt[str(ve)] += 1
    if fix_broken_notes:
        notes = validate_notes(notes, fix=True)
    if trim_overlap:
        notes = trim_overlapping_notes(notes, sort=True)
    else:
        sort_notes(notes)
    return notes, err_cnt


def note2note_event(notes: list[Note]) -> list[NoteEvent]:
    note_events = []
    for note in notes:
        if note.program == 1024:
            note.is_drum = True
        note_events.append(
            NoteEvent(note.is_drum, note.program, note.onset, 1, note.pitch)
        )
        if not note.is_drum:
            note_events.append(
                NoteEvent(note.is_drum, note.program, note.offset, 0, note.pitch)
            )
    sort_note_events(note_events)
    return note_events


def _append_tempo_changes(
    track: MidiTrack,
    tempo_changes: list[tuple[int, int]],
    messages: list[tuple[int, Message]],
) -> None:
    """Append `messages` to `track` with `tempo_changes` woven in, as deltas.

    Both are (absolute tick, ...) and in order; a tempo change goes before the
    notes on its tick, which is the tempo they start at.
    """
    timeline = sorted(
        [(tick, 0, MetaMessage("set_tempo", tempo=t)) for tick, t in tempo_changes]
        + [(tick, 1, msg) for tick, msg in messages],
        key=lambda item: (item[0], item[1]),
    )
    last = 0
    for tick, _, msg in timeline:
        track.append(msg.copy(time=tick - last))
        last = tick


def note_event2midi(
    note_events: list[NoteEvent],
    output_file: str | os.PathLike | None = None,
    velocity: int = 100,
    ticks_per_beat: int = 480,
    tempo: int = 500000,
    program_names: dict[int, str] | None = None,
    beats_per_bar: int | None = None,
    offset_s: float = 0.0,
    to_tick: Callable[[float], int] | None = None,
    tempo_changes: list[tuple[int, int]] | None = None,
    drum_length: float = 0.01,
) -> MidiFile:
    """Convert NoteEvent list to a type-1 (multi-track) MIDI file.

    Each program gets its own named track so DAWs that split imports by
    track (e.g. Ableton, which ignores channels/programs) keep the
    instruments apart. Channel assignments match the earlier type-0 layout:
    programs claim channels 0-8 then 10-15 in order of first appearance
    (sharing 15 on overflow), drums live on channel 9.

    `program_names` maps a program number (DRUM_PROGRAM for drums) to the
    track name; unmapped programs fall back to "program <n>" / "drums".

    `beats_per_bar` writes a time signature (denominator 4).
    `offset_s` is recorded in a marker as the playback time at which the
    original audio starts (see BAR_OFFSET_MARKER).

    By default event times are seconds, written at the constant `tempo` and
    delayed by `offset_s`. A caller with a tempo map passes `to_tick`, mapping
    event times (in whatever unit, drum hits lasting `drum_length` of it) to
    ticks, and `tempo_changes` as (tick, microseconds per beat) pairs.
    """
    if to_tick is None:

        def to_tick(time: float) -> int:
            return round(second2tick(time + offset_s, ticks_per_beat, tempo))

    tempo_changes = tempo_changes or [(0, tempo)]

    midi = MidiFile(ticks_per_beat=ticks_per_beat, type=1)
    meta_track = MidiTrack()
    if beats_per_bar is not None:
        meta_track.append(
            MetaMessage(
                "time_signature", numerator=beats_per_bar, denominator=4, time=0
            )
        )
    if offset_s:
        # Doesn't do anything for the MIDI but we just mark "we had to shift by this
        # much to align the bars". Read by /auralize later
        meta_track.append(
            MetaMessage("marker", text=f"{BAR_OFFSET_MARKER}{offset_s:.4f}", time=0)
        )
    _append_tempo_changes(meta_track, tempo_changes, [])
    midi.tracks.append(meta_track)

    drum_offset_events = []
    for ne in note_events:
        if ne.is_drum:
            drum_offset_events.append(
                NoteEvent(
                    is_drum=True,
                    program=ne.program,
                    time=ne.time + drum_length,
                    pitch=ne.pitch,
                    velocity=0,
                )
            )
    note_events = list(note_events) + drum_offset_events
    sort_note_events(note_events)

    program_names = program_names or {}
    program_to_channel: dict[int, int] = {}
    available_channels = list(range(0, 9)) + list(range(10, 16))
    # Each track's note messages at their absolute ticks; deltas come last.
    tracks: dict[int, list[tuple[int, Message]]] = {}
    current_tick = 0
    for ne in note_events:
        absolute_tick = to_tick(ne.time)
        if absolute_tick < current_tick:
            raise ValueError(
                f"at ne.time {ne.time}, absolute_tick {absolute_tick} < current_tick {current_tick}"
            )
        current_tick = absolute_tick

        key = DRUM_PROGRAM if (ne.is_drum or ne.program == DRUM_PROGRAM) else ne.program
        if key not in tracks:
            tracks[key] = []
            if key == DRUM_PROGRAM:
                ne_channel = 9
            else:
                try:
                    ne_channel = available_channels.pop(0)
                except IndexError:
                    ne_channel = 15
            program_to_channel[key] = ne_channel
        ne_channel = program_to_channel[key]

        msg_note = "note_on" if ne.velocity > 0 else "note_off"
        msg_velocity = velocity if ne.velocity > 0 else 0
        tracks[key].append(
            (
                absolute_tick,
                Message(
                    msg_note, note=ne.pitch, velocity=msg_velocity, channel=ne_channel
                ),
            )
        )

    for key, messages in tracks.items():
        track = MidiTrack()
        midi.tracks.append(track)
        ne_channel = program_to_channel[key]
        if key == DRUM_PROGRAM:
            name = program_names.get(key, "drums")
            gm_program = 0
        else:
            name = program_names.get(key, f"program {key}")
            gm_program = key
        track.append(MetaMessage("track_name", name=name, time=0))
        track.append(
            Message("program_change", program=gm_program, time=0, channel=ne_channel)
        )
        # MuseScore ignores set_tempo in a conductor track that has no notes,
        # so repeat the tempo here. Harmless for hosts that read the meta track.
        _append_tempo_changes(track, tempo_changes, messages)

    if output_file is not None:
        midi.save(output_file)
    return midi
