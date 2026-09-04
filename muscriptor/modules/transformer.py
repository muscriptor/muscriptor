"""Causal streaming transformer for muscriptor inference."""

import torch
from torch import nn
from torch.nn import functional as F

from muscriptor.modules.streaming import ModelState, State, StatefulModule


def create_sin_embedding(
    positions: torch.Tensor,
    dim: int,
    max_period: float = 10000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    assert dim % 2 == 0
    half_dim = dim // 2
    positions = positions.to(dtype)
    adim = torch.arange(half_dim, device=positions.device, dtype=dtype).view(1, 1, -1)
    max_period_tensor = torch.full([], max_period, device=positions.device, dtype=dtype)
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


class StreamingMultiheadAttention(StatefulModule):
    """Causal multi-head self-attention with a preallocated KV cache."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_per_head = embed_dim // num_heads

        in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False, **factory_kwargs)
        self.in_proj_weight = in_proj.weight
        self.in_proj_bias = in_proj.bias
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, **factory_kwargs)

    def init_state(self, batch_size: int, sequence_length: int) -> State:
        weight = self.in_proj_weight
        return {
            # Forward writes each cache slice before exposing only the written
            # prefix, so initializing the unread tail would be wasted work.
            "cache": torch.empty(
                (2, batch_size, sequence_length, self.num_heads, self.dim_per_head),
                device=weight.device,
                dtype=weight.dtype,
            ),
            # Kept as a plain host int: it is advanced deterministically by the
            # host-side generate loop, and reading it from a device tensor
            # (`.item()`) would force a GPU sync per layer per decode step.
            "offset": 0,
        }

    def increment_step(self, state: State, increment: int = 1) -> None:
        state["offset"] = state["offset"] + increment

    def _complete_kv(self, kv, state: State | None):
        if state is None:
            return kv[0], kv[1]
        cache = state["cache"]
        end = state["offset"]
        T = kv.shape[2]
        if T == 1:
            cache[:, :, end : end + T] = kv
        else:
            cache[0, :, end : end + T] = kv[0]
            cache[1, :, end : end + T] = kv[1]
        return cache[0, :, : end + T], cache[1, :, : end + T]

    def forward(
        self,
        query: torch.Tensor,
        model_state: ModelState | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
    ):
        state = self.get_state(model_state)
        projected = nn.functional.linear(query, self.in_proj_weight)
        B, T, _ = projected.shape
        packed = projected.reshape(B, T, 3, self.num_heads, self.dim_per_head)
        q = packed[:, :, 0]
        kv = packed[:, :, 1:].permute(2, 0, 1, 3, 4)

        k, v = self._complete_kv(kv, state)
        dtype = q.dtype

        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        # Causality must be bottom-right aligned so streaming decode steps
        # (T_q=1, T_k=cache_len) attend to all past tokens; PyTorch's
        # is_causal=True is top-left aligned and would mask out all cached
        # tokens except position 0 when T_q < T_k. An explicit attn_mask
        # can prevent dispatch to the fused flash kernel, so only build one in
        # the rectangular case that actually needs it — single-token decode
        # and square prefill stay mask-free.
        T_q, T_k = q_t.shape[2], k_t.shape[2]
        if attention_mask is not None:
            x = F.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                attn_mask=attention_mask,
                dropout_p=0.0,
            )
        elif T_q == 1:
            # One query row, bottom-right aligned: nothing is masked.
            x = F.scaled_dot_product_attention(q_t, k_t, v_t, dropout_p=0.0)
        elif T_q == T_k:
            # Square: bottom-right and top-left alignment coincide.
            x = F.scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=True, dropout_p=0.0
            )
        else:
            # Cached blocks need an explicitly bottom-right-aligned mask.
            raise NotImplementedError(
                f"Streaming attention with T_q={T_q} and T_k={T_k} "
                "requires an attention mask."
            )
        x = x.transpose(1, 2).to(dtype)

        x = x.reshape(B, T, self.embed_dim)
        x = self.out_proj(x)
        return x


class StreamingTransformerLayer(nn.Module):
    """Pre-norm transformer block: self-attention + GELU FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = StreamingMultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, **factory_kwargs
        )
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5, **factory_kwargs)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False, **factory_kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False, **factory_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        model_state: ModelState | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
    ):
        x = x + self.self_attn(
            self.norm1(x),
            model_state=model_state,
            attention_mask=attention_mask,
        )
        x = x + self.linear2(F.gelu(self.linear1(self.norm2(x))))
        return x


class StreamingTransformer(StatefulModule):
    """Stack of causal streaming transformer layers with sinusoidal positions."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        max_period: float = 10_000,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.max_period = max_period
        self.layers = nn.ModuleList(
            [
                StreamingTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

    def init_state(self, batch_size: int, sequence_length: int) -> State:
        device = self.layers[0].norm2.weight.device
        return {
            "offsets": torch.zeros(batch_size, dtype=torch.long, device=device),
        }

    def increment_step(self, state: State, increment: int = 1) -> None:
        state["offsets"] = state["offsets"] + increment

    def forward(
        self,
        x: torch.Tensor,
        prepend_length: int = 0,
        model_state: ModelState | None = None,
    ):
        del prepend_length  # unused; positions come from state['offsets']
        B, T, C = x.shape
        state = self.get_state(model_state)
        offsets = (
            state["offsets"]
            if state is not None
            else torch.zeros(B, dtype=torch.long, device=x.device)
        )

        positions = torch.arange(T, device=x.device).view(1, -1, 1)
        positions = positions + offsets.view(-1, 1, 1)
        # Always compute the sinusoidal embedding in fp32: fp16 cannot even
        # represent odd integers above 2048, so half-precision positions would
        # collapse neighbouring timesteps to the same embedding.
        pos_emb = create_sin_embedding(
            positions, C, max_period=self.max_period, dtype=torch.float32
        )
        x = x + (pos_emb * (positions >= 0).float()).to(x.dtype)

        attention_mask = None
        if model_state is not None and T > 1:
            attention_state = self.layers[0].self_attn.get_state(model_state)
            if attention_state is not None:
                past_length = attention_state["offset"]
                if past_length > 0:
                    # Query row i is the absolute position past_length + i.
                    # Build this once and share it across all layers.
                    attention_mask = torch.ones(
                        T,
                        past_length + T,
                        device=x.device,
                        dtype=torch.bool,
                    ).tril(diagonal=past_length)

        for layer in self.layers:
            x = layer(
                x,
                model_state=model_state,
                attention_mask=attention_mask,
            )
        return x
