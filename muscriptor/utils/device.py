"""Device helpers: auto-detection and cross-backend synchronization."""

import torch


def best_device() -> torch.device:
    """Pick the best available device: CUDA, then MPS (Apple Silicon), then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync() -> None:
    """Wait for pending kernels on the active accelerator.

    Used around timing measurements so they reflect real work, not just
    kernel-launch latency. No-op on CPU-only machines.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
