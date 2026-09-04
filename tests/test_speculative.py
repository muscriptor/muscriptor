"""CPU tests for greedy decoding with a separate draft model."""

import copy
import types

import pytest
import torch

from muscriptor.models.lm import LMModel
from muscriptor.models.speculative import (
    _generate_verified_tokens,
    _GreedyDecodeSession,
    generate_speculative_greedy,
)
from muscriptor.modules.conditioners import ConditioningProvider


def _tiny_model(seed: int = 0, card: int = 16) -> LMModel:
    torch.manual_seed(seed)
    device = torch.device("cpu")
    model = LMModel(
        condition_provider=ConditioningProvider(conditioners={}, device=device),
        card=card,
        dim=16,
        num_heads=2,
        hidden_scale=2,
        num_layers=1,
        max_period=10_000,
        device=device,
    )
    return model.eval()


def _tokens(steps) -> list[int]:
    return [int(step[0]) for step in steps]


def _logits(tokens: list[int], card: int = 128) -> torch.Tensor:
    logits = torch.full((1, len(tokens), card), -torch.inf)
    for index, token in enumerate(tokens):
        logits[0, index, token] = 0.0
    return logits


@pytest.mark.parametrize("draft_k", [1, 2, 4])
def test_identical_draft_matches_scalar_target_generation(draft_k):
    target = _tiny_model()
    draft = copy.deepcopy(target)

    expected = _tokens(
        target.generate(max_gen_len=12, num_samples=1, use_sampling=False)
    )
    actual = _tokens(
        generate_speculative_greedy(
            target,
            draft,
            max_gen_len=12,
            draft_k=draft_k,
        )
    )

    assert actual == expected


def test_different_model_bos_ids_preserve_prompted_target_generation():
    target = _tiny_model(card=1395)
    draft = _tiny_model(seed=1, card=1393)
    prompt = torch.tensor([[5, 3, 9]])

    expected = _tokens(
        target.generate(
            prompt=prompt,
            max_gen_len=12,
            num_samples=1,
            use_sampling=False,
        )
    )
    actual = _tokens(
        generate_speculative_greedy(
            target,
            draft,
            prompt=prompt,
            max_gen_len=12,
            draft_k=2,
        )
    )

    assert actual == expected


def test_target_eos_inside_verification_block_stops_generation():
    target = _tiny_model()
    draft = copy.deepcopy(target)
    free = _tokens(target.generate(max_gen_len=12, num_samples=1, use_sampling=False))
    eos = free[1]

    expected = _tokens(
        target.generate(
            max_gen_len=12,
            num_samples=1,
            use_sampling=False,
            early_stop_on_token=eos,
        )
    )
    actual = _tokens(
        generate_speculative_greedy(
            target,
            draft,
            max_gen_len=12,
            early_stop_on_token=eos,
            draft_k=2,
        )
    )

    assert actual == expected


def test_speculative_generation_runs_in_inference_mode():
    target = _tiny_model()
    draft = copy.deepcopy(target)
    inference_states = []
    handle = target.register_forward_pre_hook(
        lambda *args: inference_states.append(torch.is_inference_mode_enabled())
    )

    try:
        _tokens(
            generate_speculative_greedy(
                target,
                draft,
                max_gen_len=12,
                draft_k=2,
            )
        )
    finally:
        handle.remove()

    assert inference_states and all(inference_states)


def test_partial_acceptance_rewinds_draft_to_committed_prefix():
    class _TargetSession:
        def __init__(self):
            self.blocks = iter((_logits([11, 12, 13, 14, 15]),))
            self.advances = []

        def prefill(self, prompt):
            return _logits([10])[:, 0]

        def block(self, sequence):
            return next(self.blocks)

        def advance(self, increment):
            self.advances.append(increment)

    class _DraftSession:
        def __init__(self):
            self.proposals = iter((11, 99, 99, 99))
            self.advances = []

        def prefill(self, prompt):
            return _logits([0])[:, 0]

        def scalar(self, token):
            self.advances.append(1)
            return _logits([next(self.proposals)])[:, 0]

        def advance(self, increment):
            self.advances.append(increment)

    target = _TargetSession()
    draft = _DraftSession()

    steps = _generate_verified_tokens(
        target,
        draft,
        prompt=None,
        target_limit=8,
        draft_k=4,
        early_stop_on_token=None,
    )
    actual = [int(next(steps)[0]) for _ in range(3)]

    assert (actual, target.advances, draft.advances) == (
        [10, 11, 12],
        [2],
        [1, 1, 1, 1, -2],
    )


def test_block_masks_reserved_and_forbidden_output_tokens():
    model = _tiny_model(card=1395)

    def reserved_token_wins(self, *args, **kwargs):
        logits = _logits([7], card=self.card)
        logits[..., 8] = 1.0
        logits[..., 1394] = 2.0
        return logits

    model.forward = types.MethodType(reserved_token_wins, model)
    session = _GreedyDecodeSession(model, (), max_gen_len=4, forbidden_tokens=[8])

    actual = torch.argmax(session.block(torch.tensor([[3]])), dim=-1).item()

    assert actual == 7


def test_rejected_draft_matches_scalar_target_generation():
    target = _tiny_model()
    draft = _tiny_model(seed=1)
    with torch.no_grad():
        draft.linear.weight.zero_()

    expected = _tokens(
        target.generate(max_gen_len=12, num_samples=1, use_sampling=False)
    )
    actual = _tokens(
        generate_speculative_greedy(
            target,
            draft,
            max_gen_len=12,
            draft_k=2,
        )
    )

    assert actual == expected
