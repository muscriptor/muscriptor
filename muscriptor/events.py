"""Public streaming events and per-chunk event builder.

`TranscriptionModel.transcribe` is a generator that yields these dataclasses
one at a time. Every :class:`NoteStartEvent` is guaranteed to be followed by
exactly one matching :class:`NoteEndEvent` (same `index`) later in the stream.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from muscriptor.tokenizer.notes import (
    MINIMUM_NOTE_DURATION_SEC,
    Event,
)


_DRUM_INSTRUMENT = "drums"


@dataclass
class NoteStartEvent:
    pitch: int
    start_time: float
    index: int
    instrument: str


@dataclass
class NoteEndEvent:
    end_time: float
    start_event: NoteStartEvent

    @property
    def start_event_index(self) -> int:
        return self.start_event.index


@dataclass
class ProgressEvent:
    """A coarse transcription-progress signal, woven into the event stream.

    Marks that ``completed`` of ``total`` fixed-size audio chunks have been
    transcribed (``completed == 0`` is emitted once up front so consumers learn
    ``total`` and get a timing baseline; ``completed == total`` marks the end).
    These are deliberately coarse anchors — the frontend smooths between them
    and derives an ETA, since wall-clock time per chunk is only observable
    there. Advisory only: consumers that build notes/MIDI ignore them.
    """

    completed: int
    total: int


@dataclass
class ChunkBoundary:
    """Marks the start of a new model-output chunk in the token stream.

    ``seek_time`` is the chunk's start time in seconds; ``next_seek_time`` is
    the following chunk's start (``None`` for the last chunk), used to drop
    events the model emits past its window.
    """

    seek_time: float
    next_seek_time: float | None


@dataclass
class _StartNote:
    """A note opens: (program, pitch) starts sounding at `time`."""

    program: int
    pitch: int
    time: float


@dataclass
class _EndNote:
    """An open note closes: (program, pitch) stops sounding at `time`."""

    program: int
    pitch: int
    time: float


@dataclass
class _DrumHit:
    """An instantaneous drum hit at `time`; never enters the open set."""

    pitch: int
    time: float


_NoteAction = _StartNote | _EndNote | _DrumHit


class OpenNoteTracker:
    """The chunk-decoding state machine for the model's token stream.

    :meth:`feed` consumes the interleaved :class:`ChunkBoundary` markers and
    token indices and returns the note actions they imply; :meth:`finish`
    flushes the end-of-stream closes. All decode rules live here — the tie
    prologue (open notes absent from the tie set close at the boundary),
    malformed chunks (a shift before the ``tie`` token closes everything and
    drops the rest), the next_seek_time window, and retriggers.

    Two consumers share it: :func:`decode_model_tokens` turns the actions into
    indexed NoteStart/NoteEnd events, and the prelude-forcing path
    (``TranscriptionModel._generate_token_stream``) ignores the actions and
    reads :meth:`open_keys` at chunk boundaries — the ``(program, pitch)``
    pairs the next chunk's tie prologue must declare as sustained (see
    ``MT3Tokenizer.tie_section_token_ids``). One state machine serving both
    keeps decoding and forcing consistent by construction.
    """

    def __init__(self, vocab: list[Event], frame_rate: int = 100):
        self._vocab = vocab
        self._frame_rate = frame_rate
        # (program, pitch) -> onset time. Insertion-ordered: end-of-stream
        # closes replay in onset order.
        self._open: dict[tuple[int, int], float] = {}
        # Per-chunk state, reset at every ChunkBoundary.
        self._seek_time = 0.0
        self._next_seek_time: float | None = None
        self._start_tick = 0
        self._tick_state = 0
        self._program: int | None = None
        self._velocity: int | None = None
        self._in_prologue = True
        self._skip_rest = False
        self._tie_set: set[tuple[int, int]] = set()
        self._chunk_started = False

    def feed(self, item: "int | ChunkBoundary") -> list[_NoteAction]:
        if isinstance(item, ChunkBoundary):
            actions: list[_NoteAction] = []
            # If the previous chunk never closed its tie prologue (malformed:
            # no `tie` token before it ended), treat its tie set as empty so
            # every still-open note ends at that chunk's boundary.
            if self._chunk_started and self._in_prologue:
                actions = self._end_all(self._seek_time)
            self._seek_time = item.seek_time
            self._next_seek_time = item.next_seek_time
            self._start_tick = round(item.seek_time * self._frame_rate)
            self._tick_state = self._start_tick
            self._program = None
            self._velocity = None
            self._in_prologue = True
            self._skip_rest = False
            self._tie_set = set()
            self._chunk_started = True
            return actions

        event = self._vocab[item]
        etype = event.type

        if self._in_prologue:
            if etype == "tie":
                # End of the tie section: close prior notes not sustained here.
                self._in_prologue = False
                self._velocity = None
                ended = [k for k in self._open if k not in self._tie_set]
                for key in ended:
                    del self._open[key]
                return [_EndNote(*key, self._seek_time) for key in ended]
            if etype == "shift":
                # No tie token: the chunk is malformed. Close all open notes at
                # the boundary and drop the rest of the chunk.
                self._in_prologue = False
                self._skip_rest = True
                return self._end_all(self._seek_time)
            if etype == "program":
                self._program = event.value
            elif etype == "pitch" and self._program is not None:
                self._tie_set.add((self._program, event.value))
            return []

        if self._skip_rest:
            return []

        if etype == "shift":
            if event.value > 0:
                self._tick_state = self._start_tick + event.value
        elif etype == "program":
            self._program = event.value
        elif etype == "velocity":
            self._velocity = event.value
        elif etype == "drum":
            time = self._tick_state / self._frame_rate
            if self._next_seek_time is None or time < self._next_seek_time:
                return [_DrumHit(event.value, time)]
        elif etype == "pitch":
            if self._program is None or self._velocity is None:
                return []
            time = self._tick_state / self._frame_rate
            if self._next_seek_time is not None and time >= self._next_seek_time:
                return []
            key = (self._program, event.value)
            actions = []
            if key in self._open:
                del self._open[key]
                actions.append(_EndNote(*key, time))
            if self._velocity > 0:
                self._open[key] = time
                actions.append(_StartNote(*key, time))
            return actions
        return []

    def finish(self) -> list[_NoteAction]:
        """End of stream: close anything still open.

        A well-formed final chunk uses the minimum-duration fallback; a chunk
        that ended mid-prologue closes at its boundary (matching the
        malformed-chunk rule in :meth:`feed`).
        """
        if self._chunk_started and self._in_prologue:
            return self._end_all(self._seek_time)
        actions = [
            _EndNote(*key, onset + MINIMUM_NOTE_DURATION_SEC)
            for key, onset in self._open.items()
        ]
        self._open.clear()
        return actions

    def _end_all(self, time: float) -> list[_NoteAction]:
        actions: list[_NoteAction] = [_EndNote(*key, time) for key in self._open]
        self._open.clear()
        return actions

    def open_keys(self) -> list[tuple[int, int]]:
        """Sorted ``(program, pitch)`` pairs currently held open."""
        return sorted(self._open)


def decode_model_tokens(
    stream: Iterator[int | ChunkBoundary | ProgressEvent],
    vocab: list[Event],
    instrument_for_program: Callable[[int], str],
    frame_rate: int = 100,
) -> Iterator[NoteStartEvent | NoteEndEvent | ProgressEvent]:
    """Stream model token indices straight into NoteStart/NoteEnd events.

    ``stream`` interleaves :class:`ChunkBoundary` markers with token indices:
    each boundary starts a new chunk, followed by that chunk's tokens (EOS and
    anything after it already stripped). Tokens are consumed strictly in
    order: no buffering, no end-of-chunk sort. Each chunk begins with a *tie
    prologue* — ``(program, pitch)`` pairs for notes sustained from the
    previous chunk, terminated by a ``tie`` token — after which any prior open
    note not in that tie set is closed at the chunk boundary. The rest of the
    chunk drives note onsets/offsets directly.

    The decode rules themselves live in :class:`OpenNoteTracker`; this
    generator only turns its actions into events — minting indices, naming
    instruments, and pairing every NoteEndEvent with its NoteStartEvent.
    """
    tracker = OpenNoteTracker(vocab, frame_rate)
    open_notes: dict[tuple[int, int], NoteStartEvent] = {}
    next_index = 0

    def mint(pitch: int, start_time: float, instrument: str) -> NoteStartEvent:
        nonlocal next_index
        ev = NoteStartEvent(
            pitch=pitch, start_time=start_time, index=next_index, instrument=instrument
        )
        next_index += 1
        return ev

    def events_for(
        actions: list[_NoteAction],
    ) -> Iterator[NoteStartEvent | NoteEndEvent]:
        for action in actions:
            if isinstance(action, _EndNote):
                start = open_notes.pop((action.program, action.pitch))
                yield NoteEndEvent(end_time=action.time, start_event=start)
            elif isinstance(action, _StartNote):
                start = mint(
                    action.pitch, action.time, instrument_for_program(action.program)
                )
                open_notes[(action.program, action.pitch)] = start
                yield start
            else:  # _DrumHit: an instantaneous start/end pair
                start = mint(action.pitch, action.time, _DRUM_INSTRUMENT)
                yield start
                yield NoteEndEvent(
                    end_time=action.time + MINIMUM_NOTE_DURATION_SEC, start_event=start
                )

    for item in stream:
        if isinstance(item, ProgressEvent):
            # Advisory progress signal — pass straight through, untouched by
            # the decode state machine.
            yield item
            continue
        yield from events_for(tracker.feed(item))
    yield from events_for(tracker.finish())
