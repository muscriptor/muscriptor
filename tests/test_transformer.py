"""Tests for muscriptor/modules/transformer.py — CPU only, small tensors."""

import torch
from torch.nn import functional as F

from muscriptor.modules.streaming import increment_steps, init_states
from muscriptor.modules.transformer import (
    create_sin_embedding,
    StreamingMultiheadAttention,
    StreamingTransformer,
)


# ---------------------------------------------------------------------------
# Sinusoidal embeddings
# ---------------------------------------------------------------------------


def test_create_sin_embedding_shape():
    positions = torch.arange(10).float().view(1, 10, 1)  # [B, T, 1]
    emb = create_sin_embedding(positions, dim=16)
    assert emb.shape == (1, 10, 16)


def test_create_sin_embedding_dim_even():
    positions = torch.arange(5).float().view(1, 5, 1)
    emb = create_sin_embedding(positions, dim=8)
    assert emb.shape == (1, 5, 8)


def test_create_sin_embedding_different_positions():
    pos1 = torch.tensor([[[0.0]]])  # [1, 1, 1]
    pos2 = torch.tensor([[[1.0]]])
    e1 = create_sin_embedding(pos1, dim=8)
    e2 = create_sin_embedding(pos2, dim=8)
    assert not torch.allclose(e1, e2)


# ---------------------------------------------------------------------------
# Streaming attention state
# ---------------------------------------------------------------------------


def test_attention_state_does_not_initialize_unread_cache_storage(monkeypatch):
    attention = StreamingMultiheadAttention(embed_dim=8, num_heads=2)
    shape = (2, 3, 7, 2, 4)
    sentinel = 123.0
    allocator_storage = torch.full(shape, sentinel)
    original_empty = torch.empty

    def seeded_empty(*args, **kwargs):
        if args == (shape,):
            assert kwargs == {
                "device": attention.in_proj_weight.device,
                "dtype": attention.in_proj_weight.dtype,
            }
            return allocator_storage
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", seeded_empty)

    state = attention.init_state(batch_size=3, sequence_length=7)

    assert state["cache"] is allocator_storage
    assert state["offset"] == 0
    assert torch.all(state["cache"] == sentinel)


def _reference_qkv(attention, query):
    projected = F.linear(query, attention.in_proj_weight)
    q, k, v = projected.split(attention.embed_dim, dim=-1)
    shape = (*query.shape[:2], attention.num_heads, attention.dim_per_head)
    return q.reshape(shape), k.reshape(shape), v.reshape(shape)


def _reference_attention_output(attention, q, k, v, *, is_causal):
    attended = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal=is_causal,
        dropout_p=0.0,
    )
    merged = attended.transpose(1, 2).flatten(start_dim=2)
    return F.linear(merged, attention.out_proj.weight, attention.out_proj.bias)


class _RecordingCache:
    def __init__(self, tensor):
        self.tensor = tensor
        self.writes = []

    def __getitem__(self, key):
        return self.tensor[key]

    def __setitem__(self, key, value):
        self.writes.append(key)
        self.tensor[key] = value


def test_attention_prefill_matches_cpu_float32_reference_for_multiple_batches():
    torch.manual_seed(0)
    attention = StreamingMultiheadAttention(embed_dim=8, num_heads=2).eval()
    query = torch.randn(2, 3, 8)
    state = init_states(attention, batch_size=2, sequence_length=5)
    cache = _RecordingCache(state[""]["cache"])
    state[""]["cache"] = cache
    unwritten = -321.0
    cache.tensor.fill_(unwritten)

    with torch.no_grad():
        output = attention(query, model_state=state)
        q, k, v = _reference_qkv(attention, query)
        expected_output = _reference_attention_output(
            attention, q, k, v, is_causal=True
        )

    assert (output.device.type, output.dtype) == ("cpu", torch.float32)
    assert torch.equal(output, expected_output)
    assert torch.equal(cache.tensor[:, :, :3], torch.stack((k, v)))
    assert torch.all(cache.tensor[:, :, 3:] == unwritten)
    assert len(cache.writes) == 2


def test_attention_decode_updates_one_cache_slice_at_nonzero_offset():
    torch.manual_seed(1)
    attention = StreamingMultiheadAttention(embed_dim=8, num_heads=2).eval()
    prefill = torch.randn(2, 2, 8)
    decode = torch.randn(2, 1, 8)
    state = init_states(attention, batch_size=2, sequence_length=6)
    cache = _RecordingCache(state[""]["cache"])
    state[""]["cache"] = cache
    unwritten = -654.0
    cache.tensor.fill_(unwritten)

    with torch.no_grad():
        attention(prefill, model_state=state)
        increment_steps(attention, state, increment=prefill.shape[1])
        cache.writes.clear()
        output = attention(decode, model_state=state)

        _, prefill_k, prefill_v = _reference_qkv(attention, prefill)
        decode_q, decode_k, decode_v = _reference_qkv(attention, decode)
        expected_k = torch.cat((prefill_k, decode_k), dim=1)
        expected_v = torch.cat((prefill_v, decode_v), dim=1)
        expected_output = _reference_attention_output(
            attention,
            decode_q,
            expected_k,
            expected_v,
            is_causal=False,
        )

    assert torch.equal(output, expected_output)
    assert torch.equal(cache.tensor[:, :, :3], torch.stack((expected_k, expected_v)))
    assert torch.all(cache.tensor[:, :, 3:] == unwritten)
    assert state[""]["offset"] == 2
    assert len(cache.writes) == 1


# ---------------------------------------------------------------------------
# StreamingTransformer forward
# ---------------------------------------------------------------------------


def _make_transformer(**kwargs):
    defaults = dict(d_model=32, num_heads=2, num_layers=2, dim_feedforward=64)
    defaults.update(kwargs)
    return StreamingTransformer(**defaults)


def test_streaming_transformer_output_shape():
    model = _make_transformer()
    model.eval()
    x = torch.randn(2, 10, 32)  # [B, T, D]
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 10, 32)


def test_streaming_transformer_streaming_mode():
    """Feed tokens one at a time with explicit state; check output shape."""
    model = _make_transformer()
    model.eval()
    x = torch.randn(1, 6, 32)

    model_state = init_states(model, batch_size=1, sequence_length=x.shape[1])
    streaming_outs = []
    with torch.no_grad():
        for t in range(x.shape[1]):
            out_t = model(x[:, t : t + 1, :], model_state=model_state)
            streaming_outs.append(out_t)
            increment_steps(model, model_state, increment=1)
    streaming_out = torch.cat(streaming_outs, dim=1)

    assert streaming_out.shape == (1, 6, 32)


def test_streaming_transformer_fresh_state():
    model = _make_transformer()
    model.eval()
    x = torch.randn(1, 3, 32)
    state = init_states(model, batch_size=1, sequence_length=x.shape[1])
    with torch.no_grad():
        model(x, model_state=state)
    # Allocating a new state starts from scratch.
    state = init_states(model, batch_size=1, sequence_length=x.shape[1])
    with torch.no_grad():
        out = model(x, model_state=state)
    assert out.shape == (1, 3, 32)
