"""Tests for loading target and draft checkpoints."""

from types import SimpleNamespace

import pytest
import torch

import muscriptor.transcription_model as transcription_model_module
from muscriptor.tokenizer.mt3 import MT3Tokenizer
from muscriptor.transcription_model import TranscriptionModel


class _FakeLM:
    def __init__(self, card: int, device: str = "cpu"):
        self.card = card
        self.emb = SimpleNamespace(weight=torch.empty(0, device=device))


def test_load_model_configures_separate_draft_checkpoint(monkeypatch):
    target = _FakeLM(card=1395)
    draft = _FakeLM(card=1393)
    models = iter((target, draft))
    loads = []

    def load(weights_path, device, dtype):
        loads.append((weights_path, device, dtype))
        return next(models)

    monkeypatch.setattr(transcription_model_module, "_load_lm", load)

    loaded = TranscriptionModel.load_model(
        "large",
        device="cpu",
        dtype="float32",
        draft_weights_path="small",
        speculative_k=4,
    )

    assert (
        loaded._model,
        loaded._draft_model,
        loaded._speculative_k,
        loads,
    ) == (
        target,
        draft,
        4,
        [
            ("large", torch.device("cpu"), torch.float32),
            ("small", torch.device("cpu"), torch.float32),
        ],
    )


def test_invalid_speculative_k_is_rejected_before_loading(monkeypatch):
    loads = []
    monkeypatch.setattr(
        transcription_model_module,
        "_load_lm",
        lambda *args: loads.append(args),
    )

    with pytest.raises(ValueError, match="between 1 and 4"):
        TranscriptionModel.load_model(
            "large",
            device="cpu",
            draft_weights_path="small",
            speculative_k=8,
        )

    assert loads == []


def test_nondefault_speculative_k_requires_draft_checkpoint(monkeypatch):
    loads = []
    monkeypatch.setattr(
        transcription_model_module,
        "_load_lm",
        lambda *args: loads.append(args),
    )

    with pytest.raises(ValueError, match="requires draft_weights_path"):
        TranscriptionModel.load_model(
            "large",
            device="cpu",
            speculative_k=4,
        )

    assert loads == []


def test_draft_checkpoint_must_cover_shared_output_vocabulary():
    tokenizer = MT3Tokenizer(
        instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001
    )

    with pytest.raises(ValueError, match="shared output vocabulary"):
        TranscriptionModel(
            _FakeLM(card=1395),
            tokenizer,
            torch.device("cpu"),
            draft_model=_FakeLM(card=1000),
        )


def test_target_and_draft_must_use_same_device():
    tokenizer = MT3Tokenizer(
        instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001
    )

    with pytest.raises(ValueError, match="same device"):
        TranscriptionModel(
            _FakeLM(card=1395, device="cpu"),
            tokenizer,
            torch.device("cpu"),
            draft_model=_FakeLM(card=1393, device="meta"),
        )
