"""The fixed-shape decode path behind CUDA graphs must match eager decoding.

The graph itself only exists on CUDA; on CPU the decoder runs the same
fixed-shape attention eagerly, which is what these tests compare against the
regular dynamic-shape KV-cache path.
"""

import pytest
import torch

from muscriptor.models.lm import LMModel
from muscriptor.modules.conditioners import ConditioningProvider

CARD = 64


@pytest.fixture
def model() -> LMModel:
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = LMModel(
        condition_provider=ConditioningProvider(conditioners={}, device=device),
        card=CARD,
        dim=32,
        num_heads=4,
        hidden_scale=2,
        num_layers=3,
        max_period=10000,
        device=device,
    )
    model.eval()
    return model


def _generate(model, **kwargs) -> list[list[int]]:
    return [step.tolist() for step in model.generate(use_sampling=False, **kwargs)]


@pytest.fixture
def small_buckets(monkeypatch):
    """Buckets of 8 so that short generations cross several of them."""
    monkeypatch.setattr("muscriptor.modules.cuda_graph.BUCKET", 8)


@pytest.mark.parametrize("prompt", [None, torch.tensor([[5, 3, 9, 1]])])
def test_static_decode_matches_eager(model, prompt, small_buckets):
    kwargs = dict(prompt=prompt, max_gen_len=37, num_samples=1)
    eager = _generate(model, **kwargs)
    model.cuda_graphs = True
    static = _generate(model, **kwargs)
    assert static == eager
    assert model._graph_decoder is not None
    assert not model._graph_decoder.in_use


def test_decoder_is_reused_and_reset_between_generations(model, small_buckets):
    model.cuda_graphs = True
    first = _generate(model, max_gen_len=20, num_samples=1)
    decoder = model._graph_decoder
    second = _generate(model, max_gen_len=20, num_samples=1)
    assert model._graph_decoder is decoder
    assert first == second


def test_decoder_reallocated_for_a_new_shape(model):
    model.cuda_graphs = True
    _generate(model, max_gen_len=20, num_samples=1)
    decoder = model._graph_decoder
    eager_model_out = _generate(model, max_gen_len=30, num_samples=2)
    assert model._graph_decoder is not decoder
    model.cuda_graphs = False
    assert _generate(model, max_gen_len=30, num_samples=2) == eager_model_out


def test_stop_token_ends_generation(model):
    eager = _generate(model, max_gen_len=40, num_samples=1)
    stop = eager[5][0]
    model.cuda_graphs = True
    static = _generate(model, max_gen_len=40, num_samples=1, early_stop_on_token=stop)
    assert static == eager[: static.index([stop]) + 1]
    assert static[-1] == [stop]
