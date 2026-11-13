"""CUDA-graph replay of the single-token decode step.

Autoregressive decoding at batch size 1 is launch-bound on a GPU: each step is
a few hundred tiny kernels (24-48 layers x ~12 ops), and the time goes into
Python and kernel-launch overhead rather than into the GPU doing math. Capturing
one decode step as a CUDA graph and replaying it turns all of those launches
into one.

A graph replays fixed tensors with fixed shapes, so the decoder owns a
preallocated model state that it hands to ``LMModel.generate``:

- the KV caches are allocated once, at a length rounded up to a multiple of
  ``BUCKET``, zero-filled (masked slots must not hold NaN, which would leak
  through the softmax as 0 * NaN);
- the write position lives on the device: every attention state gets a
  ``"position"`` entry, a 1-element view of the transformer's in-place
  ``"offsets"`` counter, which switches attention onto its fixed-shape path
  (attend to a cache prefix, mask what is past the position).

Attending to the whole cache on every step would read mostly empty slots (a
chunk rarely gets near the 2000-token cap), so one graph is captured per
``BUCKET``-long prefix, lazily, the first time a step needs it; each step
replays the shortest one that covers its position.

The prefill (first step: conditioning + prompt) still runs eagerly through the
same state; only the T=1 steps after it are replayed.
"""

import logging

import torch
from torch import nn

from muscriptor.modules.streaming import ModelState, init_states

logger = logging.getLogger(__name__)

BUCKET = 512


class CUDAGraphDecoder:
    """Owns a static model state and the captured decode steps for one shape."""

    def __init__(self, model: nn.Module, batch_size: int, sequence_length: int):
        self.model = model
        self.key = (batch_size, sequence_length)
        self.in_use = False
        self._bucket = BUCKET
        length = -(-sequence_length // self._bucket) * self._bucket
        self.model_state: ModelState = init_states(model, batch_size, length)

        offsets = None
        for state in self.model_state.values():
            if "offsets" in state:
                offsets = state["offsets"]
        if offsets is None:
            raise RuntimeError("model has no transformer offsets to decode from")
        self._offsets = offsets
        self._attention_states = [
            state for state in self.model_state.values() if "cache" in state
        ]
        position = offsets[:1]
        for state in self._attention_states:
            state["cache"].zero_()
            state["position"] = position

        device = offsets.device
        self._tokens = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        # attend length -> (graph, its static logits output)
        self._graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
        # Off CUDA (tests), or once a capture failed: run the fixed-shape path
        # eagerly instead (still correct, just without the launch savings).
        self._eager = device.type != "cuda"

    def reset(self) -> None:
        """Rewind to an empty cache for a new generation."""
        self._offsets.zero_()
        for state in self._attention_states:
            state["offset"] = 0

    def _forward(self, attend_length: int) -> torch.Tensor:
        for state in self._attention_states:
            state["attend_length"] = attend_length
        return self.model(
            self._tokens, {}, first_step=False, model_state=self.model_state
        )

    def _capture(self, attend_length: int) -> None:
        # Warm up on a side stream (cuBLAS workspaces, SDPA kernel selection)
        # before capturing, as torch.cuda.graphs recommends. The warm-up runs
        # write this step's K/V at the current position, which the replay
        # right after overwrites with the same values.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._forward(attend_length)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        # thread_local: CUDA work on other threads (e.g. the server's) during
        # the capture must not invalidate it, nor be refused because of it.
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            logits = self._forward(attend_length)
        self._graphs[attend_length] = (graph, logits)

    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        """Logits ``[B, 1, card]`` for one decode step of ``tokens`` ``[B, 1]``.

        The returned tensor may be a graph's static output buffer, overwritten
        by a later call: consume it before then.
        """
        self._tokens.copy_(tokens)
        # The host-side offset (advanced by increment_steps) is the position
        # this step writes at, so no device read is needed to pick a graph.
        position = self._attention_states[0]["offset"]
        attend_length = (position // self._bucket + 1) * self._bucket
        if self._eager:
            return self._forward(attend_length)
        if attend_length not in self._graphs:
            try:
                self._capture(attend_length)
            except Exception as e:  # pragma: no cover - depends on driver/GPU
                logger.warning("CUDA graph capture failed, decoding eagerly: %s", e)
                self._eager = True
                return self._forward(attend_length)
        graph, logits = self._graphs[attend_length]
        graph.replay()
        return logits
