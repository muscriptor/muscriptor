"""Tests for TranscriptionModel.transcribe_batch (several inputs per generate call).

A fake ``generate()`` follows the real contract: each row first echoes its
prompt (right-padded with the "ungenerated" token, which the real model
teacher-forces row by row), then emits the scripted tokens of the chunk the
row holds. The fake audio encodes ``(input, chunk)`` in its samples, so the
fake knows which chunk each row is. That lets us check that batching inputs
together gives exactly the events :meth:`transcribe` gives one input at a
time, including the forced tie prologues.
"""

from types import SimpleNamespace

import pytest
import torch

from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent
from muscriptor.tokenizer.mt3 import MT3Tokenizer
from muscriptor.tokenizer.notes import NoteEvent
from muscriptor.transcription_model import TranscriptionModel
from tests.encode_helpers import encode_index_map, encode_note_events

_MAX_SHIFT_STEPS = 1001
_INDEX = encode_index_map(_MAX_SHIFT_STEPS)
_EOS = _INDEX[("EOS", 0)]
_UNGENERATED = -1
_CHUNK_SAMPLES = 5 * 16000


def _tok(type_: str, value: int = 0) -> int:
    return _INDEX[(type_, value)]


def _on(time: float, program: int, pitch: int) -> NoteEvent:
    return NoteEvent(is_drum=False, program=program, time=time, velocity=1, pitch=pitch)


def _off_at(shift_steps: int, pitch: int) -> list[int]:
    """Post-prologue tokens of a chunk closing a held note `shift_steps` in."""
    return [_tok("shift", shift_steps), _tok("velocity", 0), _tok("pitch", pitch)]


def _first_chunk(note_events) -> list[int]:
    return encode_note_events(note_events, max_shift_steps=_MAX_SHIFT_STEPS)


# Per input, per chunk: the tokens the model emits after its (forced) prologue.
# Input 0 holds a note across chunk 0 -> 1 -> 2; input 1 has a single chunk;
# input 2 holds two notes into its second chunk.
_SCRIPTS = [
    [
        _first_chunk([_on(0.1, 0, 60)]) + [_EOS],
        [
            _tok("shift", 20),
            _tok("program", 5),
            _tok("velocity", 1),
            _tok("pitch", 70),
            _EOS,
        ],
        _off_at(30, 60) + _off_at(40, 70) + [_EOS],
    ],
    [_first_chunk([_on(1.0, 32, 40)]) + [_EOS]],
    [
        _first_chunk([_on(0.5, 0, 50), _on(2.0, 24, 64)]) + [_EOS],
        _off_at(10, 50) + [_EOS],
    ],
]


@pytest.fixture(scope="module")
def tokenizer() -> MT3Tokenizer:
    return MT3Tokenizer(instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001)


def _model(tokenizer, scripts, calls):
    """A TranscriptionModel around a fake generate(); `calls` records, per
    generate() call, the (input, chunk) of each row and its prompt."""

    def generate(prompt=None, conditions=(), **kwargs):
        keys = [c.key for c in conditions]
        prompts = []
        for j in range(len(keys)):
            row = [] if prompt is None else prompt[j].tolist()
            prompts.append([t for t in row if t != _UNGENERATED])
        calls.append(list(zip(keys, prompts)))
        rows = [p + scripts[i][k] for (i, k), p in zip(keys, prompts)]
        for t in range(max(map(len, rows))):
            yield torch.tensor([r[t] if t < len(r) else _EOS for r in rows])

    model = TranscriptionModel.__new__(TranscriptionModel)
    model._model = SimpleNamespace(generate=generate, ungenerated_token_id=_UNGENERATED)
    model._tokenizer = tokenizer
    model._device = torch.device("cpu")
    model._instrument_for_program = lambda program: program
    # Samples carry 100 * input + chunk, so each chunk's condition knows its key.
    model._load_wav = lambda audio, sample_rate: audio
    model._build_conditions = lambda chunk, instrument_group=None: [
        SimpleNamespace(key=divmod(int(chunk[0, 0]), 100))
    ]
    return model


def _audio(index: int, num_chunks: int) -> torch.Tensor:
    return torch.cat(
        [torch.full((1, _CHUNK_SAMPLES), 100.0 * index + k) for k in range(num_chunks)],
        dim=-1,
    )


def _notes(events):
    out = []
    for ev in events:
        if isinstance(ev, NoteStartEvent):
            out.append(("start", ev.index, ev.pitch, ev.instrument, ev.start_time))
        elif isinstance(ev, NoteEndEvent):
            out.append(("end", ev.start_event_index, ev.end_time))
    return out


@pytest.fixture
def audios():
    return [_audio(i, len(s)) for i, s in enumerate(_SCRIPTS)]


def test_batch_matches_one_input_at_a_time(tokenizer, audios):
    calls = []
    model = _model(tokenizer, _SCRIPTS, calls)
    batched = dict(model.transcribe_batch(audios, batch_size=2))
    assert sorted(batched) == [0, 1, 2]
    for i, audio in enumerate(audios):
        single = [
            ev
            for ev in model.transcribe((audio, 16000))
            if not isinstance(ev, ProgressEvent)
        ]
        assert _notes(batched[i]) == _notes(single)
        assert _notes(batched[i])  # the scripts do produce notes


def test_prelude_forcing_takes_one_chunk_per_input_and_forces_each_prologue(
    tokenizer, audios
):
    calls = []
    model = _model(tokenizer, _SCRIPTS, calls)
    list(model.transcribe_batch(audios, batch_size=2))
    # Input 1 (one chunk) finishes after the first call; input 2 takes its slot.
    assert [[key for key, _ in call] for call in calls] == [
        [(0, 0), (1, 0)],
        [(0, 1), (2, 0)],
        [(0, 2), (2, 1)],
    ]
    prompts = {key: prompt for call in calls for key, prompt in call}
    assert prompts[(0, 0)] == prompts[(1, 0)] == prompts[(2, 0)] == []
    assert prompts[(0, 1)] == tokenizer.tie_section_token_ids([(0, 60)])
    assert prompts[(0, 2)] == tokenizer.tie_section_token_ids([(0, 60), (5, 70)])
    assert prompts[(2, 1)] == tokenizer.tie_section_token_ids([(0, 50), (24, 64)])


def test_without_prelude_forcing_an_input_fills_the_batch(tokenizer, audios):
    calls = []
    model = _model(tokenizer, _SCRIPTS, calls)
    list(model.transcribe_batch(audios, batch_size=4, prelude_forcing=False))
    assert [[key for key, _ in call] for call in calls] == [
        [(0, 0), (0, 1), (0, 2), (1, 0)],
        [(2, 0), (2, 1)],
    ]
    assert all(prompt == [] for call in calls for _, prompt in call)


def test_inputs_are_loaded_lazily(tokenizer):
    calls, loaded = [], []
    model = _model(tokenizer, _SCRIPTS, calls)

    def inputs():
        for i, script in enumerate(_SCRIPTS):
            loaded.append(i)
            yield _audio(i, len(script)), 16000

    results = model.transcribe_batch(inputs(), batch_size=2)
    assert next(results)[0] == 1  # the one-chunk input finishes first
    assert loaded == [0, 1]
    assert sorted(i for i, _ in results) == [0, 2]
    assert loaded == [0, 1, 2]


def test_empty_input_yields_no_events(tokenizer):
    calls = []
    model = _model(tokenizer, _SCRIPTS, calls)
    results = list(model.transcribe_batch([torch.zeros(1, 0), _audio(1, 1)]))
    assert results[0] == (0, [])
    assert [i for i, _ in results] == [0, 1]


def test_missing_eos_warns_with_the_input_index(tokenizer):
    scripts = [[[_tok("shift", 10)] * 3]]  # never emits EOS

    def generate(prompt=None, conditions=(), **kwargs):
        for t in scripts[0][0]:
            yield torch.tensor([t])

    model = _model(tokenizer, scripts, [])
    model._model.generate = generate
    with pytest.warns(
        RuntimeWarning, match=r"input 0 chunk 0 \(seek=0.0s\) did not emit EOS"
    ):
        list(model.transcribe_batch([_audio(0, 1)]))
    with pytest.raises(RuntimeError, match="input 0 chunk 0"):
        list(model.transcribe_batch([_audio(0, 1)], no_eos_is_ok=False))


def test_beam_search_with_forcing_needs_batch_size_1(tokenizer, audios):
    model = _model(tokenizer, _SCRIPTS, [])
    with pytest.raises(ValueError, match="batch_size=1"):
        list(model.transcribe_batch(audios, batch_size=2, beam_size=2))


# ---------------------------------------------------------------------------
# LMModel.generate with per-row prompts of different lengths (tiny model, CPU)
# ---------------------------------------------------------------------------


def test_generate_padded_prompts_match_one_row_at_a_time():
    """transcribe_batch relies on this: a prompt right-padded with the
    "ungenerated" token is forced where it is filled and generated after."""
    from muscriptor.models.lm import LMModel
    from muscriptor.modules.conditioners import ConditioningProvider

    torch.manual_seed(0)
    device = torch.device("cpu")
    lm = LMModel(
        condition_provider=ConditioningProvider(conditioners={}, device=device),
        card=16,
        dim=16,
        num_heads=2,
        hidden_scale=2,
        num_layers=1,
        max_period=10000,
        device=device,
    ).eval()

    def run(prompt):
        steps = lm.generate(prompt=prompt, max_gen_len=10, use_sampling=False)
        return torch.stack(list(steps), dim=1)  # [rows, T]

    prompts = [[5, 3, 9, 1], [], [7]]
    padded = torch.full((3, 4), lm.ungenerated_token_id, dtype=torch.long)
    for j, p in enumerate(prompts):
        padded[j, : len(p)] = torch.tensor(p, dtype=torch.long)
    together = run(padded)
    for j, p in enumerate(prompts):
        alone = run(torch.tensor([p]) if p else None)
        assert together[j].tolist() == alone[0].tolist()
        assert together[j, : len(p)].tolist() == p
