"""Tests for strict instrument constraints (forbidding unlisted instruments).

Covers the tokenizer's forbidden-token computation, the LMModel logit masking
across the greedy / sampling / beam decode paths (with a tiny random model, no
weights), and TranscriptionModel argument validation.
"""

from types import SimpleNamespace

import pytest
import torch

from muscriptor.models.lm import LMModel
from muscriptor.modules.conditioners import ConditioningProvider
from muscriptor.tokenizer.mt3 import MT3_FULL_PLUS_GROUP_NAMES, MT3Tokenizer
from muscriptor.transcription_model import TranscriptionModel


@pytest.fixture(scope="module")
def tokenizer() -> MT3Tokenizer:
    return MT3Tokenizer(instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001)


# ---------------------------------------------------------------------------
# MT3Tokenizer.forbidden_token_ids
# ---------------------------------------------------------------------------


def _representative_program(tokenizer: MT3Tokenizer, name: str) -> int:
    return tokenizer.group_program_map[MT3_FULL_PLUS_GROUP_NAMES[name]][0]


def test_forbidden_ids_never_touch_timing_or_special_tokens(tokenizer):
    forbidden = set(tokenizer.forbidden_token_ids(["violin"]))
    for token_id, event in enumerate(tokenizer._vocab):
        if event.type in ("PAD", "EOS", "UNK", "shift", "pitch", "velocity", "tie"):
            assert token_id not in forbidden, (token_id, event)


def test_forbidden_ids_keep_only_listed_programs(tokenizer):
    forbidden = set(tokenizer.forbidden_token_ids(["violin", "cello"]))
    allowed_programs = {
        _representative_program(tokenizer, "violin"),
        _representative_program(tokenizer, "cello"),
    }
    for token_id, event in enumerate(tokenizer._vocab):
        if event.type == "program":
            if event.value in allowed_programs:
                assert token_id not in forbidden, event
            else:
                assert token_id in forbidden, event


def test_forbidden_ids_mask_drums_unless_listed(tokenizer):
    drum_ids = {
        token_id
        for token_id, event in enumerate(tokenizer._vocab)
        if event.type == "drum"
    }
    without_drums = set(tokenizer.forbidden_token_ids(["violin"]))
    assert drum_ids <= without_drums
    with_drums = set(tokenizer.forbidden_token_ids(["violin", "drums"]))
    assert not (drum_ids & with_drums)


def test_forbidden_ids_drums_only_masks_every_program(tokenizer):
    forbidden = set(tokenizer.forbidden_token_ids(["drums"]))
    for token_id, event in enumerate(tokenizer._vocab):
        if event.type == "program":
            assert token_id in forbidden, event


def test_forbidden_ids_rejects_unknown_names(tokenizer):
    with pytest.raises(ValueError, match="unknown instrument name"):
        tokenizer.forbidden_token_ids(["violin", "kazoo"])


# ---------------------------------------------------------------------------
# LMModel.generate masking (tiny random model, CPU)
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


def _generate_tokens(model: LMModel, **kwargs) -> list[int]:
    steps = model.generate(max_gen_len=8, num_samples=1, **kwargs)
    return [int(t) for step in steps for t in step.tolist()]


def test_greedy_never_emits_forbidden_tokens(tiny_model):
    allowed = 3
    forbidden = [t for t in range(CARD) if t != allowed]
    tokens = _generate_tokens(
        tiny_model, use_sampling=False, forbidden_tokens=forbidden
    )
    assert tokens and set(tokens) == {allowed}


def test_sampling_never_emits_forbidden_tokens(tiny_model):
    torch.manual_seed(1234)
    allowed = {3, 7}
    forbidden = [t for t in range(CARD) if t not in allowed]
    tokens = _generate_tokens(
        tiny_model, use_sampling=True, temp=2.0, forbidden_tokens=forbidden
    )
    assert tokens and set(tokens) <= allowed


def test_beam_search_never_emits_forbidden_tokens(tiny_model):
    allowed = {3, 7}
    forbidden = [t for t in range(CARD) if t not in allowed]
    tokens = _generate_tokens(
        tiny_model,
        use_sampling=False,
        beam_size=2,
        early_stop_on_token=7,
        forbidden_tokens=forbidden,
    )
    assert tokens and set(tokens) <= allowed


def test_generate_unmasked_uses_full_vocabulary(tiny_model):
    # Sanity check that the assertions above bite: without a mask, sampling at
    # high temperature over random weights spreads over more than the two
    # tokens the masked tests allow.
    torch.manual_seed(1234)
    tokens = _generate_tokens(tiny_model, use_sampling=True, temp=2.0)
    assert len(set(tokens)) > 2


# ---------------------------------------------------------------------------
# TranscriptionModel argument validation
# ---------------------------------------------------------------------------


def test_transcribe_strict_requires_instruments():
    fake = SimpleNamespace(_device=torch.device("cpu"))
    stream = TranscriptionModel.transcribe(
        fake, "unused.wav", strict_instruments=True
    )
    with pytest.raises(ValueError, match="strict_instruments"):
        next(stream)
