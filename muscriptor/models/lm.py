"""Language model for MIDI token generation.

Adapted from audiocraft/models/lm.py.
"""

import logging
import time
from collections.abc import Iterator

import torch
from torch import nn

import muscriptor.accelerator
from muscriptor.modules.conditioners import (
    ConditioningProvider,
    ConditioningAttributes,
    ConditionType,
    nullify_all_conditions,
)
from muscriptor.modules.streaming import (
    ModelState,
    increment_steps,
    init_states,
)
from muscriptor.modules.transformer import StreamingTransformer
import muscriptor.utils.sampling as utils


logger = logging.getLogger(__name__)
ConditionTensors = dict[str, ConditionType]


# ---------------------------------------------------------------------------
# ScaledEmbedding  (used for token embeddings, keeps weight compatible with ckpt)
# ---------------------------------------------------------------------------


class ScaledEmbedding(nn.Embedding):
    """Embedding that maps zero_idx (a negative index) to a zero vector."""

    def __init__(self, *args, zero_idx: int = -1, **kwargs):
        super().__init__(*args, **kwargs)
        assert zero_idx < 0
        self.zero_idx = zero_idx

    def forward(self, input, *args, **kwargs):
        is_zero = input == self.zero_idx
        input = input.clamp(min=0)
        y = super().forward(input, *args, **kwargs)
        return torch.where(is_zero[..., None], torch.zeros_like(y), y)


# ---------------------------------------------------------------------------
# TorchAutocast
# ---------------------------------------------------------------------------


class TorchAutocast:
    """Minimal autocast context manager (matches the audiocraft interface)."""

    def __init__(
        self,
        enabled: bool = False,
        device_type: str = "cuda",
        dtype: torch.dtype | None = None,
    ):
        self.enabled = enabled
        self.device_type = device_type
        self.dtype = dtype
        self._ctx = None

    def __enter__(self):
        if self.enabled:
            self._ctx = torch.autocast(device_type=self.device_type, dtype=self.dtype)
            self._ctx.__enter__()
        return self

    def __exit__(self, *args):
        if self.enabled and self._ctx is not None:
            self._ctx.__exit__(*args)


# ---------------------------------------------------------------------------
# LMModel
# ---------------------------------------------------------------------------


class LMModel(nn.Module):
    """Causal transformer LM for MIDI token generation.

    Single-stream
    Supports classifier-free guidance at inference time.
    """

    def __init__(
        self,
        condition_provider: ConditioningProvider,
        card: int = 1024,
        dim: int = 128,
        num_heads: int = 8,
        hidden_scale: int = 4,
        cfg_coef: float = 1.0,
        autocast: TorchAutocast | None = None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.condition_provider = condition_provider
        self.card = card
        self.dim = dim
        self.cfg_coef = cfg_coef
        self.autocast = (
            autocast if autocast is not None else TorchAutocast(enabled=False)
        )

        self.emb = ScaledEmbedding(
            self.card + 1,
            dim,
            device=device,
            dtype=dtype,
            zero_idx=self.zero_token_id,
        )

        self.transformer = StreamingTransformer(
            d_model=dim,
            num_heads=num_heads,
            dim_feedforward=int(hidden_scale * dim),
            device=device,
            dtype=dtype,
            **kwargs,
        )
        self.out_norm = nn.LayerNorm(dim, eps=1e-5)
        self.linear = nn.Linear(dim, card, bias=False)

    # ------------------------------------------------------------------
    # Token ID properties
    # ------------------------------------------------------------------

    @property
    def initial_token_id(self) -> int:
        return self.card

    @property
    def zero_token_id(self) -> int:
        return -1

    @property
    def ungenerated_token_id(self) -> int:
        return -2

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        sequence: torch.Tensor,  # [B, S]
        condition_tensors: ConditionTensors,
        first_step: bool = False,
        model_state: ModelState | None = None,
    ) -> torch.Tensor:  # [B, S, card]
        B, S = sequence.shape

        input_ = self.emb(sequence)  # [B, S, D]

        prepend_length = 0
        if first_step:
            for cond, _ in condition_tensors.values():
                # Conditioners run in fp32 even when the transformer runs in
                # half precision (mel numerics degrade in fp16) — cast at the
                # seam.
                input_ = torch.cat([cond.to(input_.dtype), input_], dim=1)
            prepend_length = input_.shape[1] - S

        transformer_out = self.transformer(
            input_,
            prepend_length=prepend_length,
            model_state=model_state,
        )
        if self.out_norm:
            transformer_out = self.out_norm(transformer_out)

        # Remove prepended conditioning tokens
        if prepend_length > 0:
            transformer_out = transformer_out[:, -S:]

        logits = self.linear(transformer_out)
        return logits  # [B, S, card]

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _compute_logits(
        self,
        sequence: torch.Tensor,
        cfg_conditions: ConditionTensors,
        model_state: ModelState,
        first_step: bool,
        cfg_coef: float | None = None,
        forbidden_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:  # [B, card]
        """Run the forward pass and return masked logits at the last timestep."""
        B = sequence.shape[0]
        cfg_coef = self.cfg_coef if cfg_coef is None else cfg_coef

        if cfg_coef == 1.0:
            logits = self(
                sequence,
                cfg_conditions,
                first_step=first_step,
                model_state=model_state,
            )
        else:
            doubled = torch.cat([sequence, sequence], dim=0)
            all_logits = self(
                doubled,
                cfg_conditions,
                first_step=first_step,
                model_state=model_state,
            )
            cond_logits, uncond_logits = all_logits.split(B, dim=0)
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_coef

        logits = logits[:, -1, :].float()  # [B, card] — last timestep
        logits[:, 1393:] = -torch.inf      # mask reserved / OOV tokens
        if forbidden_tokens is not None:
            logits[:, forbidden_tokens] = -torch.inf
        return logits

    def _sample_next_token(
        self,
        sequence: torch.Tensor,
        cfg_conditions: ConditionTensors,
        model_state: ModelState,
        first_step: bool,
        use_sampling: bool = False,
        temp: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        cfg_coef: float | None = None,
        forbidden_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:  # [B]
        logits = self._compute_logits(
            sequence, cfg_conditions, model_state, first_step, cfg_coef,
            forbidden_tokens=forbidden_tokens,
        )
        if use_sampling and temp > 0.0:
            probs = torch.softmax(logits / temp, dim=-1)
            next_tokens = utils.sample_from_probs(probs, top_p=top_p, top_k=top_k)[:, 0]
        else:
            next_tokens = torch.argmax(logits, dim=-1)  # [B]
        return next_tokens  # [B]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor | None = None,
        conditions: list[ConditioningAttributes] = [],
        num_samples: int | None = None,
        max_gen_len: int = 256,
        use_sampling: bool = True,
        temp: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        cfg_coef: float | None = None,
        early_stop_on_token: int | None = None,
        beam_size: int = 1,
        beam_length_score_alpha: float = 0.75,
        forbidden_tokens: torch.Tensor | list[int] | None = None,
    ) -> Iterator[torch.Tensor]:
        """Autoregressively generate tokens, yielding one timestep at a time.

        Each yield is a ``[num_samples]`` tensor. For beam_size == 1 (default),
        tokens are yielded as they are generated. For beam_size > 1, beam search
        is run non-streamingly and all tokens are yielded at the end.

        ``forbidden_tokens`` are token ids whose logits are forced to -inf at
        every step, so they can never be sampled (greedy, sampling or beam).
        """
        assert not self.training
        if beam_size > 1:
            assert early_stop_on_token is not None, "beam search requires early_stop_on_token"
        device = self.emb.weight.device

        if forbidden_tokens is not None and not isinstance(
            forbidden_tokens, torch.Tensor
        ):
            forbidden_tokens = torch.tensor(
                forbidden_tokens, device=device, dtype=torch.long
            )

        if num_samples is None:
            num_samples = (
                len(conditions)
                if conditions
                else (prompt.shape[0] if prompt is not None else 1)
            )

        cfg_coef = self.cfg_coef if cfg_coef is None else cfg_coef

        # Build condition tensors (with null conditions appended for CFG)
        if conditions:
            if cfg_coef == 1.0:
                prepared = self.condition_provider.tokenize(conditions)
                muscriptor.accelerator.synchronize()
                _t = time.perf_counter()
                cfg_conditions: ConditionTensors = self.condition_provider(prepared)
                muscriptor.accelerator.synchronize()
                print(
                    f"[muscriptor] encode conditions (total): {time.perf_counter() - _t:.3f}s"
                )
            else:
                null_conditions = nullify_all_conditions(conditions)
                all_conditions = conditions + null_conditions
                prepared = self.condition_provider.tokenize(all_conditions)
                print(
                    "[muscriptor] instrument_group tokens:",
                    prepared.get("instrument_group"),
                )
                print(
                    "[muscriptor] dataset_name tokens:    ",
                    prepared.get("dataset_name"),
                )
                muscriptor.accelerator.synchronize()
                _t = time.perf_counter()
                cfg_conditions = self.condition_provider(prepared)
                muscriptor.accelerator.synchronize()
                print(
                    f"[muscriptor] encode conditions (total): {time.perf_counter() - _t:.3f}s"
                )
        else:
            cfg_conditions = {}

        eff_batch = num_samples * beam_size

        # Expand conditions so each beam gets its own copy (interleaved for CFG).
        if beam_size > 1 and cfg_conditions:
            cfg_conditions = {
                k: (
                    torch.repeat_interleave(cond, beam_size, dim=0),
                    torch.repeat_interleave(mask, beam_size, dim=0),
                )
                for k, (cond, mask) in cfg_conditions.items()
            }

        # Initialise generation buffer (eff_batch rows = num_samples × beam_size)
        ungenerated = self.ungenerated_token_id
        gen_sequence = torch.full(
            (eff_batch, max_gen_len + 1),
            ungenerated,
            device=device,
            dtype=torch.long,
        )
        gen_sequence[:, 0] = self.initial_token_id

        start_offset = 0
        if prompt is not None:
            PT = prompt.shape[-1]
            if beam_size > 1:
                prompt = torch.repeat_interleave(prompt, beam_size, dim=0)
            gen_sequence[:, 1 : 1 + PT] = prompt
            ungenerated_steps = (gen_sequence == ungenerated).nonzero()[:, 1]
            start_offset = max(0, int(ungenerated_steps.amin()) - 1)

        prepend_length = sum(cond.shape[1] for cond, _ in cfg_conditions.values())
        cache_batch_size = eff_batch * (1 if cfg_coef == 1.0 else 2)
        cache_seq_len = prepend_length + max_gen_len
        model_state = init_states(
            self, batch_size=cache_batch_size, sequence_length=cache_seq_len
        )

        # Accumulated log-prob scores, one per beam row.
        beam_scores = torch.zeros(eff_batch, device=device, dtype=torch.float)

        # For greedy/sampling emit prompt steps now; beam search emits at the end.
        if beam_size == 1:
            for t in range(start_offset):
                yield gen_sequence[:, t + 1]

        last_offset = start_offset - 1
        with self.autocast:
            for offset in range(start_offset, max_gen_len):
                last_offset = offset
                first_iter = offset == start_offset
                input_ = (
                    gen_sequence[:, : offset + 1]
                    if first_iter
                    else gen_sequence[:, offset : offset + 1]
                )

                if beam_size == 1:
                    # ── Standard greedy / sampling path ──────────────────
                    if early_stop_on_token is not None:
                        done = (gen_sequence == early_stop_on_token).any(dim=1).all()
                        if done:
                            break

                    next_token = self._sample_next_token(
                        input_,
                        cfg_conditions,
                        model_state,
                        first_step=first_iter,
                        use_sampling=use_sampling,
                        temp=temp,
                        top_k=top_k,
                        top_p=top_p,
                        cfg_coef=cfg_coef,
                        forbidden_tokens=forbidden_tokens,
                    )  # [B]

                    input_T = input_.shape[-1]
                    increment_steps(
                        self.transformer,
                        model_state,
                        increment=input_T + (prepend_length if first_iter else 0),
                    )

                    this_gen_step = gen_sequence[:, offset + 1]
                    next_token = torch.where(
                        this_gen_step == ungenerated, next_token, this_gen_step
                    )
                    gen_sequence[:, offset + 1] = next_token

                    yield gen_sequence[:, offset + 1]  # [num_samples]

                else:
                    # ── Beam search step ──────────────────────────────────
                    logits = self._compute_logits(
                        input_, cfg_conditions, model_state,
                        first_step=first_iter, cfg_coef=cfg_coef,
                        forbidden_tokens=forbidden_tokens,
                    )  # [eff_batch, card]
                    input_T = input_.shape[-1]
                    increment_steps(
                        self.transformer,
                        model_state,
                        increment=input_T + (prepend_length if first_iter else 0),
                    )

                    log_probs = torch.log_softmax(logits.float(), dim=-1)

                    # Top beam_size candidate tokens per current beam
                    topk_scores, topk_tokens = torch.topk(log_probs, k=beam_size, dim=-1)

                    # Track which beams have already emitted EOS
                    eos_mask = gen_sequence == early_stop_on_token
                    beam_has_ended = eos_mask.any(dim=-1)
                    eos_pos = eos_mask.int().argmax(dim=-1).clamp(min=1)
                    beam_lengths = torch.where(
                        beam_has_ended, eos_pos,
                        torch.full_like(eos_pos, offset + 1),
                    )

                    # Finished beams: don't expand further
                    topk_scores = torch.where(
                        beam_has_ended.unsqueeze(-1),
                        torch.zeros_like(topk_scores),
                        topk_scores,
                    )

                    # Length-normalized candidate scores: [eff_batch, beam_size]
                    lp = 1.0 / (beam_lengths.float() ** beam_length_score_alpha)
                    cand = (beam_scores.unsqueeze(-1) + topk_scores) * lp.unsqueeze(-1)

                    # Reshape to [num_samples, beam_size²] for cross-beam selection
                    cand_2d = cand.reshape(num_samples, beam_size * beam_size)

                    if offset == start_offset:
                        # All beams identical at start — take first beam_size tokens
                        new_scores = cand_2d[:, :beam_size]
                        best_idx = (
                            torch.arange(beam_size, device=device)
                            .unsqueeze(0)
                            .expand(num_samples, -1)
                        )
                    else:
                        new_scores, best_idx = torch.topk(cand_2d, k=beam_size, dim=-1)

                    # Decode flat index → (prev_beam_within_sample, token_rank)
                    prev_local = (best_idx // beam_size).reshape(-1)
                    tok_rank = (best_idx % beam_size).reshape(-1)

                    # Map to global row indices in [eff_batch, …] tensors
                    sample_base = (
                        torch.arange(num_samples, device=device)
                        .repeat_interleave(beam_size) * beam_size
                    )
                    prev_global = sample_base + prev_local

                    # Token for each new beam
                    next_token = topk_tokens[prev_global, tok_rank]

                    # Update beam scores (store un-normalized for the next step)
                    beam_scores = new_scores.reshape(-1) / lp[prev_global]

                    # Reorder generation sequences to match winning beams
                    gen_sequence = gen_sequence[prev_global]

                    # Reorder KV caches — shape is [2, batch, T, heads, head_dim]
                    for state in model_state.values():
                        if "cache" in state:
                            cache = state["cache"]
                            if cache.shape[1] == 2 * eff_batch:  # CFG-doubled cache
                                reorder = torch.cat([prev_global, prev_global + eff_batch])
                            else:
                                reorder = prev_global
                            state["cache"] = cache[:, reorder, :, :, :]

                    # Write next token (respecting pre-filled prompt positions)
                    this_step = gen_sequence[:, offset + 1]
                    next_token = torch.where(this_step == ungenerated, next_token, this_step)
                    gen_sequence[:, offset + 1] = next_token

                    # Early stop when every beam in every sample has emitted EOS
                    if (gen_sequence == early_stop_on_token).any(dim=-1).all():
                        break

        # Beam search: select best beam per sample and yield all tokens at once
        if beam_size > 1:
            best_beam = beam_scores.reshape(num_samples, beam_size).argmax(dim=-1)
            best_global = (
                torch.arange(num_samples, device=device) * beam_size + best_beam
            )
            best_sequence = gen_sequence[best_global]  # [num_samples, T]
            for t in range(last_offset + 1):
                yield best_sequence[:, t + 1]
