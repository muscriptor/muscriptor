"""Greedy speculative decoding with a separate draft model."""

from collections.abc import Iterator, Sequence

import torch

from muscriptor.models.lm import EMITTABLE_VOCAB_SIZE, ConditionTensors, LMModel
from muscriptor.modules.conditioners import ConditioningAttributes
from muscriptor.modules.streaming import ModelState, increment_steps, init_states

DEFAULT_DRAFT_K = 2
MIN_DRAFT_K = 1
# K=8 changed an MPS/fp16 target argmax in the evaluated corpus because scalar
# and masked block attention can round differently. Keep the exposed range at
# the largest block size that preserved all target tokens in that evaluation.
MAX_DRAFT_K = 4


def validate_draft_k(draft_k: int) -> None:
    if not MIN_DRAFT_K <= draft_k <= MAX_DRAFT_K:
        raise ValueError(f"draft K must be between {MIN_DRAFT_K} and {MAX_DRAFT_K}")


class _GreedyDecodeSession:
    """One model's conditions and KV state for a batch-one greedy decode."""

    def __init__(
        self,
        model: LMModel,
        conditions: Sequence[ConditioningAttributes],
        max_gen_len: int,
        forbidden_tokens: torch.Tensor | list[int] | None,
    ) -> None:
        self.model = model
        self.device = model.emb.weight.device
        if conditions:
            prepared = model.condition_provider.tokenize(list(conditions))
            self.conditions: ConditionTensors = model.condition_provider(prepared)
        else:
            self.conditions = {}
        self.prepend_length = sum(
            condition.shape[1] for condition, _ in self.conditions.values()
        )
        self.state: ModelState = init_states(
            model,
            batch_size=1,
            sequence_length=self.prepend_length + max_gen_len,
        )
        if forbidden_tokens is None or isinstance(forbidden_tokens, torch.Tensor):
            self.forbidden_tokens = forbidden_tokens
        else:
            self.forbidden_tokens = torch.tensor(
                forbidden_tokens, device=self.device, dtype=torch.long
            )

    def prefill(self, prompt: torch.Tensor | None) -> torch.Tensor:
        bos = torch.tensor(
            [[self.model.initial_token_id]], device=self.device, dtype=torch.long
        )
        sequence = bos if prompt is None else torch.cat((bos, prompt), dim=1)
        logits = self.model._compute_logits(
            sequence,
            self.conditions,
            self.state,
            first_step=True,
            cfg_coef=1.0,
            forbidden_tokens=self.forbidden_tokens,
        )
        self.advance(self.prepend_length + sequence.shape[-1])
        return logits

    def scalar(self, token: torch.Tensor) -> torch.Tensor:
        logits = self.model._compute_logits(
            token.view(1, 1),
            self.conditions,
            self.state,
            first_step=False,
            cfg_coef=1.0,
            forbidden_tokens=self.forbidden_tokens,
        )
        self.advance(1)
        return logits

    def block(self, sequence: torch.Tensor) -> torch.Tensor:
        logits = self.model(
            sequence,
            self.conditions,
            first_step=False,
            model_state=self.state,
        ).float()
        logits[..., EMITTABLE_VOCAB_SIZE:] = -torch.inf
        if self.forbidden_tokens is not None:
            logits[..., self.forbidden_tokens] = -torch.inf
        return logits

    def advance(self, increment: int) -> None:
        increment_steps(self.model.transformer, self.state, increment=increment)


def _generate_verified_tokens(
    target: _GreedyDecodeSession,
    draft: _GreedyDecodeSession,
    *,
    prompt: torch.Tensor | None,
    target_limit: int,
    draft_k: int,
    early_stop_on_token: int | None,
) -> Iterator[torch.Tensor]:
    target_logits = target.prefill(prompt)
    draft.prefill(prompt)
    current = torch.argmax(target_logits, dim=-1)
    generated = 1
    yield current
    if early_stop_on_token is not None and int(current[0]) == early_stop_on_token:
        return

    while generated < target_limit:
        remaining = target_limit - generated
        if remaining == 1:
            yield torch.argmax(target.scalar(current), dim=-1)
            return

        proposal_count = min(draft_k, remaining - 1)
        proposal_tokens = []
        draft_input = current
        for _ in range(proposal_count):
            draft_input = torch.argmax(draft.scalar(draft_input), dim=-1)
            proposal_tokens.append(draft_input)
        proposals = torch.stack(proposal_tokens, dim=1)

        verifier_input = torch.cat((current.view(1, 1), proposals), dim=1)
        target_tokens = torch.argmax(target.block(verifier_input), dim=-1)
        # Move both tensors together so acceptance needs only one device sync.
        compared = torch.cat((target_tokens[0], proposals[0])).cpu().tolist()
        target_ids = compared[: proposal_count + 1]
        proposal_ids = compared[proposal_count + 1 :]
        mismatch = next(
            (
                index
                for index, (target_id, proposal_id) in enumerate(
                    zip(target_ids, proposal_ids, strict=False)
                )
                if target_id != proposal_id
            ),
            None,
        )
        commit_count = proposal_count + 1 if mismatch is None else mismatch + 1
        committed_ids = target_ids[:commit_count]
        stop = False
        if early_stop_on_token is not None and early_stop_on_token in committed_ids:
            commit_count = committed_ids.index(early_stop_on_token) + 1
            stop = True

        target.advance(commit_count)
        generated += commit_count
        if not stop and generated < target_limit:
            if mismatch is None:
                draft.scalar(proposals[:, -1])
            else:
                draft.advance(commit_count - proposal_count)

        for index in range(commit_count):
            yield target_tokens[:, index]
        if stop:
            return
        current = target_tokens[:, commit_count - 1]


@torch.inference_mode()
def generate_speculative_greedy(
    target_model: LMModel,
    draft_model: LMModel,
    *,
    prompt: torch.Tensor | None = None,
    conditions: Sequence[ConditioningAttributes] = (),
    max_gen_len: int = 256,
    early_stop_on_token: int | None = None,
    forbidden_tokens: torch.Tensor | list[int] | None = None,
    draft_k: int = DEFAULT_DRAFT_K,
) -> Iterator[torch.Tensor]:
    """Yield target-greedy tokens verified from a separate draft model."""
    assert not target_model.training and not draft_model.training
    validate_draft_k(draft_k)
    if len(conditions) > 1:
        raise ValueError("speculative decoding requires batch size 1")
    target_device = target_model.emb.weight.device
    if draft_model.emb.weight.device != target_device:
        raise ValueError("target and draft models must use the same device")
    if prompt is not None:
        if prompt.shape[0] != 1:
            raise ValueError("speculative decoding requires a batch-one prompt")
        prompt = prompt.to(target_device)
        for index in range(prompt.shape[1]):
            yield prompt[:, index]

    prompt_length = 0 if prompt is None else prompt.shape[1]
    target_limit = max_gen_len - prompt_length
    if target_limit <= 0:
        return

    target = _GreedyDecodeSession(
        target_model, conditions, max_gen_len, forbidden_tokens
    )
    draft = _GreedyDecodeSession(draft_model, conditions, max_gen_len, forbidden_tokens)
    with target_model.autocast, draft_model.autocast:
        yield from _generate_verified_tokens(
            target,
            draft,
            prompt=prompt,
            target_limit=target_limit,
            draft_k=draft_k,
            early_stop_on_token=early_stop_on_token,
        )
