"""Tests for TranscriptionModel._generate_token_stream.

These check the streaming contract of the token stream without a real model:
a fake `generate()` yields one row (`[batch]`) per timestep and records how
many timesteps have been pulled, so we can assert that a chunk's events come
out *as soon as that chunk finishes* — before the rest of the batch is even
generated.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from muscriptor.events import ChunkBoundary, ProgressEvent
from muscriptor.transcription_model import TranscriptionModel

EOS = 99


def test_transcribe_passes_public_generation_budget_to_token_stream(monkeypatch):
    captured: dict[str, int] = {}

    def generate_token_stream(*args):
        captured["max_generation_tokens"] = args[3]
        return iter(())

    fake = SimpleNamespace(
        _device=torch.device("cpu"),
        _tokenizer=SimpleNamespace(_vocab=[], frame_rate=100),
        _instrument_for_program=lambda _program: "piano",
        _resolve_batch_size=lambda _batch_size, _prelude_forcing: 1,
        _load_wav=lambda _audio, _sample_rate: torch.zeros(1, 80_000),
        _build_conditions=lambda _chunk, _instrument_group: [object()],
        _generate_token_stream=generate_token_stream,
    )
    monkeypatch.setattr(
        "muscriptor.transcription_model.decode_model_tokens",
        lambda stream, *_args, **_kwargs: stream,
    )
    monkeypatch.setattr("muscriptor.accelerator.synchronize", lambda: None)

    events = list(
        TranscriptionModel.transcribe(
            fake,
            "audio.wav",
            max_generation_tokens=321,
        )
    )

    assert events == [ProgressEvent(completed=0, total=1)]
    assert captured["max_generation_tokens"] == 321


@pytest.mark.parametrize("value", [0, -1, True])
def test_transcribe_rejects_invalid_generation_budget(value):
    fake = SimpleNamespace(_resolve_batch_size=Mock())

    with pytest.raises(ValueError, match="positive integer"):
        list(
            TranscriptionModel.transcribe(
                fake,
                "audio.wav",
                max_generation_tokens=value,
            )
        )

    fake._resolve_batch_size.assert_not_called()


def test_transcribe_to_midi_forwards_generation_budget():
    fake = SimpleNamespace(
        transcribe=Mock(return_value=iter(())),
        events_to_midi_bytes=Mock(return_value=b"midi"),
    )

    result = TranscriptionModel.transcribe_to_midi(
        fake,
        "audio.wav",
        max_generation_tokens=456,
    )

    assert result == b"midi"
    assert fake.transcribe.call_args.kwargs["max_generation_tokens"] == 456


def _run(batches, *, batch_size, seek_times, no_eos_is_ok=False):
    """Drive _generate_token_stream with a fake model.

    ``batches`` is one list of rows per expected ``generate()`` call; each row
    is the per-chunk token for one timestep. Returns ``(stream, pulled)`` where
    ``pulled`` grows by one entry every time the fake yields a timestep, so its
    length is how far generation has progressed.
    """
    pulled: list[list[int]] = []
    calls = iter(batches)

    def generate(**kwargs):
        for row in next(calls):
            pulled.append(row)
            yield torch.tensor(row)

    fake = SimpleNamespace(
        _model=SimpleNamespace(generate=generate),
        _tokenizer=SimpleNamespace(eos_id=EOS),
    )
    conditions = [object()] * len(seek_times)
    stream = TranscriptionModel._generate_token_stream(
        fake,
        conditions,
        seek_times,
        batch_size,
        max_gen_len=64,
        use_sampling=False,
        temperature=1.0,
        cfg_coef=2.0,
        no_eos_is_ok=no_eos_is_ok,
        # The fake tokenizer has no vocab; prelude forcing has its own tests
        # (test_prelude_forcing.py).
        prelude_forcing=False,
    )
    return stream, pulled


# ---------------------------------------------------------------------------
# Emitted as soon as possible
# ---------------------------------------------------------------------------


def test_first_chunk_streams_before_the_batch_finishes():
    # batch of 2 chunks: chunk 0 ends at row 2, chunk 1 only at row 4.
    rows = [[10, 20], [11, 21], [EOS, 22], [12, 23], [13, EOS]]
    stream, pulled = _run([rows], batch_size=2, seek_times=[0.0, 5.0])
    it = iter(stream)

    assert next(it) == ChunkBoundary(0.0, 5.0)
    assert len(pulled) == 0  # the boundary is emitted before any generation
    assert next(it) == 10
    assert len(pulled) == 1  # first token after a single timestep
    assert next(it) == 11
    # Chunk 0 is fully streamed having generated only its own timesteps —
    # chunk 1 (which finishes at row 4) has not been generated to completion.
    assert len(pulled) == 2


def test_single_chunk_streams_token_by_token():
    rows = [[10], [11], [12], [EOS]]
    stream, pulled = _run([rows], batch_size=1, seek_times=[0.0])
    it = iter(stream)

    assert next(it) == ChunkBoundary(0.0, None)
    assert len(pulled) == 0
    for expected, count in [(10, 1), (11, 2), (12, 3)]:
        assert next(it) == expected
        assert len(pulled) == count


# ---------------------------------------------------------------------------
# Ordering and buffering
# ---------------------------------------------------------------------------


def test_full_stream_order_for_a_batch():
    rows = [[10, 20], [11, 21], [EOS, 22], [12, 23], [13, EOS]]
    stream, _ = _run([rows], batch_size=2, seek_times=[0.0, 5.0])
    assert list(stream) == [
        ChunkBoundary(0.0, 5.0),
        10,
        11,
        ChunkBoundary(5.0, None),
        20,
        21,
        22,
        23,
        # End of the (only) batch: both chunks done.
        ProgressEvent(completed=2, total=2),
    ]


def test_later_chunk_finishing_first_is_buffered_until_its_turn():
    # chunk 1 hits EOS (row 1) before chunk 0 (row 3); its tokens must wait.
    rows = [[10, 20], [11, EOS], [12, 88], [EOS, 88]]
    stream, _ = _run([rows], batch_size=2, seek_times=[0.0, 5.0])
    assert list(stream) == [
        ChunkBoundary(0.0, 5.0),
        10,
        11,
        12,
        ChunkBoundary(5.0, None),
        20,
        ProgressEvent(completed=2, total=2),
    ]


def test_chunks_across_multiple_batches_stay_in_order():
    # batch_size=1 → one generate() call per chunk.
    batches = [[[10], [11], [EOS]], [[20], [EOS]]]
    stream, _ = _run(batches, batch_size=1, seek_times=[0.0, 5.0])
    assert list(stream) == [
        ChunkBoundary(0.0, 5.0),
        10,
        11,
        # batch_size=1 => a completion anchor trails each chunk.
        ProgressEvent(completed=1, total=2),
        ChunkBoundary(5.0, None),
        20,
        ProgressEvent(completed=2, total=2),
    ]


# ---------------------------------------------------------------------------
# Missing EOS
# ---------------------------------------------------------------------------


def test_missing_eos_raises_by_default():
    rows = [[10, 20], [11, 21]]  # neither chunk emits EOS
    stream, _ = _run([rows], batch_size=2, seek_times=[0.0, 5.0])
    with pytest.raises(RuntimeError, match="did not emit EOS"):
        list(stream)


def test_missing_eos_warns_and_still_emits_when_allowed():
    rows = [[10, 20], [11, 21]]
    stream, _ = _run([rows], batch_size=2, seek_times=[0.0, 5.0], no_eos_is_ok=True)
    with pytest.warns(RuntimeWarning, match="did not emit EOS"):
        events = list(stream)
    assert events == [
        ChunkBoundary(0.0, 5.0),
        10,
        11,
        ChunkBoundary(5.0, None),
        20,
        21,
        ProgressEvent(completed=2, total=2),
    ]
