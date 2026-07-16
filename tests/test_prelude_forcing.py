"""Tests for teacher-forced tie prologues (prelude forcing).

Covers the layers of the feature:
- MT3Tokenizer.tie_section_token_ids produces exactly the training encoder's
  tie-prologue layout;
- OpenNoteTracker (the single decode state machine) keeps an open-note set
  that agrees with the events decode_model_tokens builds from its actions
  (checked by feeding both the same streams);
- TranscriptionModel._resolve_batch_size makes forcing the quality default:
  batch size defaults to 1 with forcing on, and batching requires explicitly
  disabling the forcing;
- _generate_token_stream forces the prologue of every chunk after the first
  (batch_size == 1 only, and only when prelude_forcing is on), via the
  `prompt` argument of LMModel.generate — whose prompt path is exercised here
  with a tiny random model.
"""

from types import SimpleNamespace

import pytest
import torch

from muscriptor.events import (
    ChunkBoundary,
    NoteEndEvent,
    NoteStartEvent,
    OpenNoteTracker,
    ProgressEvent,
    decode_model_tokens,
)
from muscriptor.models.lm import LMModel
from muscriptor.modules.conditioners import ConditioningProvider
from muscriptor.tokenizer.mt3 import MT3Tokenizer
from muscriptor.tokenizer.notes import NoteEvent, TieNoteEvent, build_event_vocab
from muscriptor.transcription_model import TranscriptionModel
from tests.encode_helpers import encode_index_map, encode_note_events, note_event2event

_MAX_SHIFT_STEPS = 1001
_VOCAB = build_event_vocab(_MAX_SHIFT_STEPS)
_INDEX = encode_index_map(_MAX_SHIFT_STEPS)


def _tok(type_: str, value: int = 0) -> int:
    return _INDEX[(type_, value)]


_EOS = _tok("EOS")


@pytest.fixture(scope="module")
def tokenizer() -> MT3Tokenizer:
    return MT3Tokenizer(instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001)


def _on(time: float, program: int, pitch: int) -> NoteEvent:
    return NoteEvent(is_drum=False, program=program, time=time, velocity=1, pitch=pitch)


def _off(time: float, program: int, pitch: int) -> NoteEvent:
    return NoteEvent(is_drum=False, program=program, time=time, velocity=0, pitch=pitch)


def _drum(time: float, pitch: int) -> NoteEvent:
    return NoteEvent(is_drum=True, program=128, time=time, velocity=1, pitch=pitch)


# ---------------------------------------------------------------------------
# MT3Tokenizer.tie_section_token_ids
# ---------------------------------------------------------------------------


def test_tie_section_matches_training_encoder(tokenizer):
    keys = [(32, 60), (0, 64), (0, 55), (32, 70)]
    tie_notes = [TieNoteEvent(program=p, pitch=pi) for p, pi in keys]
    reference = note_event2event([], tie_note_events=tie_notes, start_time=0.0)
    expected = [_INDEX[(e.type, e.value)] for e in reference]
    assert tokenizer.tie_section_token_ids(keys) == expected


def test_tie_section_empty_is_just_the_tie_token(tokenizer):
    assert tokenizer.tie_section_token_ids([]) == [_tok("tie")]


def test_tie_section_sorts_and_dedupes_programs(tokenizer):
    got = tokenizer.tie_section_token_ids([(5, 62), (5, 60), (2, 40)])
    assert got == [
        _tok("program", 2),
        _tok("pitch", 40),
        _tok("program", 5),
        _tok("pitch", 60),
        _tok("pitch", 62),
        _tok("tie"),
    ]


# ---------------------------------------------------------------------------
# OpenNoteTracker — with parity checks against decode_model_tokens
# ---------------------------------------------------------------------------
#
# Each chunk below is (seek_time, next_seek_time, tokens). Both the tracker
# and the decoder consume the identical stream; the decoder's open set is
# recovered from its NoteStart/NoteEnd events *before* its end-of-stream
# cleanup runs (a trailing ProgressEvent marks that every token has been
# consumed while the generator is still suspended).


def _tracker_open_keys(chunks) -> list[tuple[int, int]]:
    tracker = OpenNoteTracker(_VOCAB)
    for seek, next_seek, tokens in chunks:
        tracker.feed(ChunkBoundary(seek, next_seek))
        for t in tokens:
            tracker.feed(t)
    return tracker.open_keys()


def _decoder_open_keys(chunks) -> list[tuple[int, int]]:
    marker = ProgressEvent(completed=1, total=1)

    def stream():
        for seek, next_seek, tokens in chunks:
            yield ChunkBoundary(seek, next_seek)
            yield from tokens
        yield marker

    open_: dict[int, tuple[int, int]] = {}
    # instrument_for_program=identity so the program number survives in
    # NoteStartEvent.instrument (drums come out as "drums" but never stay open).
    for ev in decode_model_tokens(stream(), _VOCAB, lambda program: program):
        if isinstance(ev, ProgressEvent):
            break
        if isinstance(ev, NoteStartEvent):
            open_[ev.index] = (ev.instrument, ev.pitch)
        else:
            open_.pop(ev.start_event_index)
    return sorted(open_.values())


def _assert_open(chunks, expected: list[tuple[int, int]]):
    assert _tracker_open_keys(chunks) == expected
    assert _decoder_open_keys(chunks) == expected


def _encode(note_events, tie_notes=None, start_time=0.0) -> list[int]:
    return encode_note_events(
        note_events,
        max_shift_steps=_MAX_SHIFT_STEPS,
        tie_note_events=tie_notes,
        start_time=start_time,
    )


def test_unfinished_note_stays_open():
    chunk = _encode([_on(0.1, 0, 60), _on(0.2, 0, 64), _off(0.4, 0, 64)])
    _assert_open([(0.0, 5.0, chunk)], [(0, 60)])


def test_note_closed_in_next_chunk_via_tie():
    chunk0 = _encode([_on(0.1, 0, 60)])
    chunk1 = _encode(
        [_off(5.5, 0, 60)], tie_notes=[TieNoteEvent(0, 60)], start_time=5.0
    )
    _assert_open([(0.0, 5.0, chunk0), (5.0, 10.0, chunk1)], [])


def test_note_missing_from_tie_section_is_closed_at_boundary():
    chunk0 = _encode([_on(0.1, 0, 60)])
    chunk1 = _encode([], tie_notes=None, start_time=5.0)  # empty tie section
    _assert_open([(0.0, 5.0, chunk0), (5.0, 10.0, chunk1)], [])


def test_onset_past_chunk_window_is_ignored():
    chunk = _encode([_on(0.1, 0, 60), _on(6.0, 0, 62)])
    _assert_open([(0.0, 5.0, chunk)], [(0, 60)])


def test_drums_never_stay_open():
    chunk = _encode([_drum(0.5, 38), _drum(1.0, 42)])
    _assert_open([(0.0, 5.0, chunk)], [])


def test_retriggered_note_stays_open_once():
    chunk = _encode([_on(0.1, 0, 60), _on(0.3, 0, 60)])
    _assert_open([(0.0, 5.0, chunk)], [(0, 60)])


def test_malformed_chunk_without_tie_closes_everything():
    chunk0 = _encode([_on(0.1, 0, 60)])
    # A shift before any tie token: the decoder closes all open notes and
    # drops the rest of the chunk.
    chunk1 = [
        _tok("shift", 10),
        _tok("program", 0),
        _tok("velocity", 1),
        _tok("pitch", 62),
    ]
    _assert_open([(0.0, 5.0, chunk0), (5.0, 10.0, chunk1)], [])


def test_chunk_ending_mid_prologue_closes_everything_at_next_boundary():
    chunk0 = _encode([_on(0.1, 0, 60)])
    chunk1 = [_tok("program", 0), _tok("pitch", 60)]  # never reaches its tie token
    chunk2 = _encode([], start_time=10.0)
    _assert_open([(0.0, 5.0, chunk0), (5.0, 10.0, chunk1), (10.0, None, chunk2)], [])


# ---------------------------------------------------------------------------
# _resolve_batch_size: prelude forcing is the quality default
# ---------------------------------------------------------------------------


def _resolve(device_type: str, batch_size, prelude_forcing) -> int:
    fake = SimpleNamespace(_device=SimpleNamespace(type=device_type))
    return TranscriptionModel._resolve_batch_size(fake, batch_size, prelude_forcing)


def test_default_batch_size_is_1_while_forcing():
    assert _resolve("cuda", None, True) == 1
    assert _resolve("cpu", None, True) == 1


def test_default_batch_size_without_forcing_follows_device():
    assert _resolve("cuda", None, False) == 4
    assert _resolve("cpu", None, False) == 1


def test_batching_with_forcing_raises():
    with pytest.raises(ValueError, match="prelude_forcing=False"):
        _resolve("cuda", 4, True)


def test_explicit_batch_size_1_keeps_forcing():
    assert _resolve("cuda", 1, True) == 1


# ---------------------------------------------------------------------------
# _generate_token_stream: prompt wiring (fake model, real tokenizer)
# ---------------------------------------------------------------------------


def _run_stream(scripts, tokenizer, *, seek_times, batch_size=1, prelude_forcing=True):
    """Drive _generate_token_stream with a fake generate().

    ``scripts`` holds, per expected generate() call, the rows the fake emits
    *after* echoing back any prompt (mimicking the real generate contract,
    which yields prompt tokens through the stream first). Each row is one
    token per chunk in the batch. Returns (stream_items, prompts) where
    ``prompts`` records the prompt passed to each call (as a list, or None).
    """
    prompts: list[list[int] | None] = []

    def generate(prompt=None, **kwargs):
        prompts.append(None if prompt is None else prompt[0].tolist())
        if prompt is not None:
            assert prompt.shape[0] == 1
            for t in prompt[0].tolist():
                yield torch.tensor([t])
        for row in scripts[len(prompts) - 1]:
            yield torch.tensor(row)

    fake = SimpleNamespace(
        _model=SimpleNamespace(generate=generate),
        _tokenizer=tokenizer,
        _device=torch.device("cpu"),
    )
    stream = list(
        TranscriptionModel._generate_token_stream(
            fake,
            [object()] * len(seek_times),
            seek_times,
            batch_size,
            max_gen_len=64,
            use_sampling=False,
            temperature=1.0,
            cfg_coef=1.0,
            no_eos_is_ok=True,
            prelude_forcing=prelude_forcing,
        )
    )
    return stream, prompts


def _rows(tokens: list[int]) -> list[list[int]]:
    return [[t] for t in tokens]


def test_second_chunk_prelude_is_forced(tokenizer):
    # Chunk 0 opens (program 0, pitch 60) and never closes it.
    chunk0 = _encode([_on(0.1, 0, 60)]) + [_EOS]
    # Chunk 1's post-prologue script closes it at t=5.5.
    chunk1 = [_tok("shift", 50), _tok("velocity", 0), _tok("pitch", 60), _EOS]
    stream, prompts = _run_stream(
        [_rows(chunk0), _rows(chunk1)], tokenizer, seek_times=[0.0, 5.0]
    )

    assert prompts[0] is None  # the first chunk is never forced
    assert prompts[1] == tokenizer.tie_section_token_ids([(0, 60)])

    # The forced tokens flow through the stream right after the boundary,
    # so the downstream decoder sees them like any generated prologue.
    i = stream.index(ChunkBoundary(5.0, None))
    assert stream[i + 1 : i + 1 + len(prompts[1])] == prompts[1]

    # End to end: the sustained note survives the boundary and ends at 5.5.
    events = [
        ev
        for ev in decode_model_tokens(
            iter(stream), tokenizer._vocab, lambda program: program
        )
        if not isinstance(ev, ProgressEvent)
    ]
    assert len(events) == 2
    start, end = events
    assert isinstance(start, NoteStartEvent) and start.pitch == 60
    assert start.start_time == 0.1
    assert isinstance(end, NoteEndEvent) and end.end_time == 5.5


def test_forced_prelude_accumulates_across_chunks(tokenizer):
    chunk0 = _encode([_on(0.1, 0, 60)]) + [_EOS]
    # Chunk 1 (its own prologue forced) opens a second note on program 5.
    chunk1 = [
        _tok("shift", 20),
        _tok("program", 5),
        _tok("velocity", 1),
        _tok("pitch", 70),
        _EOS,
    ]
    chunk2 = [_EOS]
    _, prompts = _run_stream(
        [_rows(chunk0), _rows(chunk1), _rows(chunk2)],
        tokenizer,
        seek_times=[0.0, 5.0, 10.0],
    )
    assert prompts[1] == tokenizer.tie_section_token_ids([(0, 60)])
    assert prompts[2] == tokenizer.tie_section_token_ids([(0, 60), (5, 70)])


def test_prelude_forcing_flag_off_disables_forcing(tokenizer):
    chunk0 = _encode([_on(0.1, 0, 60)]) + [_EOS]
    chunk1 = _encode([], start_time=5.0) + [_EOS]
    _, prompts = _run_stream(
        [_rows(chunk0), _rows(chunk1)],
        tokenizer,
        seek_times=[0.0, 5.0],
        prelude_forcing=False,
    )
    assert prompts == [None, None]


def test_batch_size_above_one_disables_forcing(tokenizer):
    eos = tokenizer.eos_id
    rows = [[10, 20], [eos, 21], [99, eos]]
    _, prompts = _run_stream([rows], tokenizer, seek_times=[0.0, 5.0], batch_size=2)
    assert prompts == [None]


# ---------------------------------------------------------------------------
# LMModel.generate prompt path (tiny random model, CPU)
# ---------------------------------------------------------------------------

CARD = 16


@pytest.fixture(scope="module")
def tiny_model() -> LMModel:
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = LMModel(
        condition_provider=ConditioningProvider(conditioners={}, device=device),
        card=CARD,
        dim=16,
        num_heads=2,
        hidden_scale=2,
        num_layers=1,
        max_period=10000,
        device=device,
    )
    model.eval()
    return model


def _tokens(steps) -> list[int]:
    return [int(s[0]) for s in steps]


def test_generate_yields_prompt_tokens_first(tiny_model):
    prompt = torch.tensor([[5, 3, 9]])
    tokens = _tokens(
        tiny_model.generate(
            prompt=prompt, max_gen_len=8, num_samples=1, use_sampling=False
        )
    )
    assert tokens[:3] == [5, 3, 9]
    assert len(tokens) == 8


def test_generate_with_own_greedy_prefix_as_prompt_is_a_noop(tiny_model):
    # Forcing the tokens greedy decoding would have produced anyway must not
    # change the continuation — validates the prompt path's KV-cache/offset
    # bookkeeping.
    free = _tokens(
        tiny_model.generate(max_gen_len=8, num_samples=1, use_sampling=False)
    )
    forced = _tokens(
        tiny_model.generate(
            prompt=torch.tensor([free[:3]]),
            max_gen_len=8,
            num_samples=1,
            use_sampling=False,
        )
    )
    assert forced == free


def test_generate_beam_search_respects_prompt(tiny_model):
    prompt = torch.tensor([[5, 3]])
    tokens = _tokens(
        tiny_model.generate(
            prompt=prompt,
            max_gen_len=8,
            num_samples=1,
            use_sampling=False,
            beam_size=2,
            early_stop_on_token=7,
        )
    )
    assert tokens[:2] == [5, 3]
