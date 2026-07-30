"""Tests for overlapping-window transcription and the gzip restart criterion.

Ported from audiocraft_trans. Overlap forcing generalizes tie-section forcing
(see test_prelude_forcing.py): instead of forcing only the tie prologue, each
chunk after the first is teacher-forced to replay the *whole* note-event
sequence the previous chunk predicted over the shared window. The restart
criterion generates an overlapping chunk twice (with / without the overlap
prompt) and keeps the un-prompted one when the prompted one goes degenerate.

Layers covered:
- MT3Tokenizer.overlap_prompt_token_ids matches the training encoder and
  reduces to tie_section_token_ids when the overlap is empty;
- transcribe() lays out overlapping windows (stride = 5 - overlap);
- TranscriptionModel._gzip_ratio / _select_with_gzip implement the restart
  criterion;
- _generate_token_stream forces each chunk's overlap region, and (with
  allow_reset) resets to the un-prompted continuation when needed.
"""

from types import SimpleNamespace

import pytest
import torch

from muscriptor.events import ChunkBoundary
from muscriptor.tokenizer.mt3 import MT3Tokenizer
from muscriptor.tokenizer.notes import NoteEvent, TieNoteEvent
from muscriptor.transcription_model import TranscriptionModel
from tests.encode_helpers import encode_index_map, encode_note_events, note_event2event
from tests.test_prelude_forcing import _run_stream

_MAX_SHIFT_STEPS = 1001
_INDEX = encode_index_map(_MAX_SHIFT_STEPS)
_EOS = _INDEX[("EOS", 0)]


@pytest.fixture(scope="module")
def tokenizer() -> MT3Tokenizer:
    return MT3Tokenizer(instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001)


def _ne(is_drum, program, time, velocity, pitch) -> NoteEvent:
    return NoteEvent(is_drum, program, time, velocity, pitch)


def _rows(tokens):
    return [[t] for t in tokens]


def _bare_model(tokenizer) -> TranscriptionModel:
    """An un-``__init__``'d model carrying only what the gzip helpers touch."""
    model = object.__new__(TranscriptionModel)
    model._tokenizer = tokenizer
    model._device = torch.device("cpu")
    return model


# ---------------------------------------------------------------------------
# MT3Tokenizer.overlap_prompt_token_ids
# ---------------------------------------------------------------------------


def test_empty_overlap_equals_tie_section(tokenizer):
    keys = [(0, 60), (5, 70), (5, 62)]
    assert tokenizer.overlap_prompt_token_ids(
        keys, [], seek_time=2.5
    ) == tokenizer.tie_section_token_ids(keys)


def test_overlap_prompt_matches_training_encoder(tokenizer):
    seek = 2.5
    tie_keys = [(0, 60)]
    # Note events inside the overlap window, absolute times, mixed program+drum.
    events = [
        _ne(False, 5, 2.7, 1, 70),  # onset, new program
        _ne(True, 128, 2.8, 1, 38),  # drum hit
        _ne(False, 5, 3.4, 0, 70),  # offset
    ]
    ref = note_event2event(
        [_ne(*(e.is_drum, e.program, e.time, e.velocity, e.pitch)) for e in events],
        tie_note_events=[TieNoteEvent(p, pi) for p, pi in tie_keys],
        start_time=seek,
    )
    expected = [_INDEX[(e.type, e.value)] for e in ref]
    got = tokenizer.overlap_prompt_token_ids(tie_keys, events, seek_time=seek)
    assert got == expected
    # And it is a strict superset of the tie-only prologue (same prefix).
    tie_only = tokenizer.tie_section_token_ids(tie_keys)
    assert got[: len(tie_only)] == tie_only


# ---------------------------------------------------------------------------
# transcribe(): overlapping window layout
# ---------------------------------------------------------------------------


def _captured_seek_times(duration_sec, overlap, tokenizer):
    """Run transcribe() with audio/model faked out; return (num_chunks, seek_times)."""
    captured = {}
    n_samples = int(duration_sec * 16000)

    class _Fake:
        _device = torch.device("cpu")
        _tokenizer = tokenizer
        _instrument_for_program = staticmethod(lambda program: "x")
        _resolve_batch_size = TranscriptionModel._resolve_batch_size

        def _load_wav(self, audio, sample_rate):
            return torch.zeros(1, n_samples)

        def _build_conditions(self, wav, instrument_group=None):
            return [SimpleNamespace()]

        def _generate_token_stream(self, all_conditions, seek_times, *rest):
            captured["seek_times"] = list(seek_times)
            captured["n_conditions"] = len(all_conditions)
            return iter([])

    list(TranscriptionModel.transcribe(_Fake(), "unused.wav", overlap=overlap))
    return captured["n_conditions"], captured["seek_times"]


def test_overlap_zero_layout_is_adjacent(tokenizer):
    n, seeks = _captured_seek_times(12.0, 0.0, tokenizer)
    assert n == 3
    assert seeks == [0.0, 5.0, 10.0]


def test_overlap_lays_out_strided_windows(tokenizer):
    # stride = 5 - 2.5 = 2.5; 12s => 1 + ceil((12-5)/2.5) = 4 windows.
    n, seeks = _captured_seek_times(12.0, 2.5, tokenizer)
    assert n == 4
    assert seeks == [0.0, 2.5, 5.0, 7.5]


def test_short_audio_is_a_single_window(tokenizer):
    n, seeks = _captured_seek_times(3.0, 2.5, tokenizer)
    assert n == 1
    assert seeks == [0.0]


def test_overlap_out_of_range_rejected(tokenizer):
    model = _bare_model(tokenizer)
    with pytest.raises(ValueError, match="overlap"):
        list(model.transcribe("x.wav", overlap=5.0))


def test_overlap_requires_prelude_forcing(tokenizer):
    model = _bare_model(tokenizer)
    with pytest.raises(ValueError, match="prelude_forcing"):
        list(model.transcribe("x.wav", overlap=2.5, prelude_forcing=False))


# ---------------------------------------------------------------------------
# _forcing_prompt_ids: capping a runaway-chunk prompt to the generation budget
# ---------------------------------------------------------------------------


def test_forcing_prompt_keeps_overlap_when_it_fits(tokenizer):
    model = _bare_model(tokenizer)
    prev = encode_note_events([_ne(False, 0, 2.6, 1, 60)], _MAX_SHIFT_STEPS)
    ids = model._forcing_prompt_ids(
        [(5, 70)], prev, 0.0, 2.5, overlap=2.5, max_gen_len=2000, chunk_index=3
    )
    events = model._overlap_note_events(prev, 0.0, 2.5, 2.5)
    assert ids == tokenizer.overlap_prompt_token_ids([(5, 70)], events, 2.5)
    assert ids  # no fallback


def test_forcing_prompt_caps_runaway_overlap_to_tie_only(tokenizer):
    model = _bare_model(tokenizer)
    # A previous chunk that predicted many onsets in the overlap window [2.5, 5)
    # yields an overlap prompt too long for a tiny generation budget.
    prev = encode_note_events(
        [_ne(False, 0, 2.5 + 0.1 * k, 1, 60 + k) for k in range(6)],
        _MAX_SHIFT_STEPS,
    )
    with pytest.warns(RuntimeWarning, match="exceeds the generation budget"):
        ids = model._forcing_prompt_ids(
            [], prev, 0.0, 2.5, overlap=2.5, max_gen_len=6, chunk_index=3
        )
    assert ids == tokenizer.tie_section_token_ids([])  # fell back to the tie prologue


def test_forcing_prompt_drops_forcing_when_even_tie_too_long(tokenizer):
    model = _bare_model(tokenizer)
    open_keys = [(p, 60) for p in range(20)]  # 41-token tie prologue
    with pytest.warns(RuntimeWarning, match="without forcing"):
        ids = model._forcing_prompt_ids(
            open_keys, [], 0.0, 2.5, overlap=0.0, max_gen_len=10, chunk_index=3
        )
    assert ids == []  # no forcing at all


# ---------------------------------------------------------------------------
# gzip restart criterion
# ---------------------------------------------------------------------------


def test_gzip_ratio_flags_repetition(tokenizer):
    model = _bare_model(tokenizer)
    motif = [
        _INDEX[("shift", 1)],
        _INDEX[("program", 0)],
        _INDEX[("velocity", 1)],
        _INDEX[("pitch", 60)],
    ]
    repetitive = motif * 40
    varied = encode_note_events(
        [_ne(False, i % 30, 0.1 * i, 1, 40 + i) for i in range(1, 30)],
        _MAX_SHIFT_STEPS,
    )
    assert model._gzip_ratio(repetitive) > model._gzip_ratio(varied)
    assert model._gzip_ratio([]) == -1.0


def test_select_resets_when_prompted_is_degenerate(tokenizer):
    model = _bare_model(tokenizer)
    degenerate = [
        _INDEX[("shift", 1)],
        _INDEX[("program", 0)],
        _INDEX[("velocity", 1)],
        _INDEX[("pitch", 60)],
    ] * 40
    clean = encode_note_events(
        [_ne(False, 0, 0.5, 1, 60), _ne(False, 0, 1.0, 0, 60)], _MAX_SHIFT_STEPS
    )
    with pytest.warns(RuntimeWarning, match="degenerate"):
        chosen = model._select_with_gzip(clean, True, degenerate, True, seek=2.5)
    assert chosen is clean


def test_select_keeps_prompted_when_healthy(tokenizer):
    model = _bare_model(tokenizer)
    with_tokens = encode_note_events(
        [_ne(False, 0, 0.5, 1, 60), _ne(False, 0, 1.0, 0, 60)], _MAX_SHIFT_STEPS
    )
    no_tokens = encode_note_events([_ne(False, 0, 0.5, 1, 62)], _MAX_SHIFT_STEPS)
    chosen = model._select_with_gzip(no_tokens, True, with_tokens, True, seek=2.5)
    assert chosen is with_tokens


def test_select_resets_when_prompted_never_ends(tokenizer):
    model = _bare_model(tokenizer)
    with_tokens = encode_note_events([_ne(False, 0, 0.5, 1, 60)], _MAX_SHIFT_STEPS)
    no_tokens = encode_note_events([_ne(False, 0, 0.5, 1, 62)], _MAX_SHIFT_STEPS)
    with pytest.warns(RuntimeWarning):
        chosen = model._select_with_gzip(no_tokens, True, with_tokens, False, seek=2.5)
    assert chosen is no_tokens


# ---------------------------------------------------------------------------
# _generate_token_stream: overlap forcing + reset wiring (fake model)
# ---------------------------------------------------------------------------


def test_overlap_prompt_replays_previous_chunk(tokenizer):
    # Chunk 0 (window [0,5)): a note that stays open past the boundary, plus a
    # note + drum inside the overlap region [2.5, 5).
    chunk0_events = [
        _ne(False, 0, 1.0, 1, 60),  # sustained across the boundary -> tie note
        _ne(False, 0, 3.0, 1, 62),  # onset inside overlap
        _ne(True, 128, 3.5, 1, 38),  # drum inside overlap
        _ne(False, 0, 4.0, 0, 62),  # offset inside overlap
    ]
    chunk0 = encode_note_events(chunk0_events, _MAX_SHIFT_STEPS) + [_EOS]
    chunk1 = [_EOS]

    stream, prompts = _run_stream(
        [_rows(chunk0), _rows(chunk1)],
        tokenizer,
        seek_times=[0.0, 2.5],
        overlap=2.5,
    )

    assert prompts[0] is None
    # Tie section = the sustained note; overlap events = the ones at t >= 2.5.
    expected = tokenizer.overlap_prompt_token_ids(
        [(0, 60)],
        [
            _ne(False, 0, 3.0, 1, 62),
            _ne(True, 128, 3.5, 1, 38),
            _ne(False, 0, 4.0, 0, 62),
        ],
        seek_time=2.5,
    )
    assert prompts[1] == expected
    # The forced tokens flow through the stream right after the boundary.
    i = stream.index(ChunkBoundary(2.5, None))
    assert stream[i + 1 : i + 1 + len(expected)] == expected


def test_allow_reset_generates_twice_and_can_reset(tokenizer):
    chunk0 = encode_note_events([_ne(False, 0, 1.0, 1, 60)], _MAX_SHIFT_STEPS) + [_EOS]
    # Chunk 1 generated twice: with-prompt collapses (repetitive), no-prompt clean.
    degenerate = [
        _INDEX[("shift", 1)],
        _INDEX[("program", 0)],
        _INDEX[("velocity", 1)],
        _INDEX[("pitch", 60)],
    ] * 40 + [_EOS]
    clean = encode_note_events(
        [_ne(False, 0, 3.0, 0, 60)],
        tie_note_events=[TieNoteEvent(0, 60)],
        max_shift_steps=_MAX_SHIFT_STEPS,
        start_time=2.5,
    ) + [_EOS]

    with pytest.warns(RuntimeWarning, match="degenerate"):
        stream, prompts = _run_stream(
            [_rows(chunk0), _rows(degenerate), _rows(clean)],
            tokenizer,
            seek_times=[0.0, 2.5],
            overlap=2.5,
            allow_reset=True,
        )

    # Three generate() calls: chunk0 (None), chunk1 with-prompt, chunk1 no-prompt.
    assert prompts[0] is None
    assert prompts[1] is not None  # overlap prompt
    assert prompts[2] is None  # the un-prompted reset run

    # The reset kept the clean continuation, so the degenerate token motif
    # never reaches the decoded stream after the second boundary.
    i = stream.index(ChunkBoundary(2.5, None))
    tail = [t for t in stream[i + 1 :] if isinstance(t, int)]
    assert tail == clean[:-1]  # EOS stripped


def test_allow_reset_keeps_prompted_when_healthy(tokenizer):
    chunk0 = encode_note_events([_ne(False, 0, 1.0, 1, 60)], _MAX_SHIFT_STEPS) + [_EOS]
    # Both runs healthy; with-prompt (a normal continuation) must be kept.
    with_prompt = encode_note_events(
        [_ne(False, 0, 3.0, 0, 60)],
        tie_note_events=[TieNoteEvent(0, 60)],
        max_shift_steps=_MAX_SHIFT_STEPS,
        start_time=2.5,
    ) + [_EOS]
    no_prompt = encode_note_events([_ne(False, 0, 3.0, 1, 90)], _MAX_SHIFT_STEPS) + [
        _EOS
    ]

    stream, prompts = _run_stream(
        [_rows(chunk0), _rows(with_prompt), _rows(no_prompt)],
        tokenizer,
        seek_times=[0.0, 2.5],
        overlap=2.5,
        allow_reset=True,
    )
    i = stream.index(ChunkBoundary(2.5, None))
    tail = [t for t in stream[i + 1 :] if isinstance(t, int)]
    # Kept the prompted run: the fake echoes the overlap prompt, then the body.
    assert tail == prompts[1] + with_prompt[:-1]
