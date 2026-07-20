"""Device-agnostic accelerator helpers.

``torch.accelerator`` (``is_available``/``current_accelerator``/``synchronize``)
was only added in PyTorch 2.6; on older versions we fall back to checking CUDA
then MPS directly, in that order.
"""

import torch

_TORCH_VERSION = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
_HAS_TORCH_ACCELERATOR = _TORCH_VERSION >= (2, 6)


def is_available() -> bool:
    """Whether an accelerator (GPU) is available."""
    if _HAS_TORCH_ACCELERATOR:
        return torch.accelerator.is_available()
    return torch.cuda.is_available() or torch.backends.mps.is_available()


def current_accelerator() -> torch.device:
    """The current accelerator device.

    Raises ``RuntimeError`` if no accelerator is available; check
    :func:`is_available` first.
    """
    if _HAS_TORCH_ACCELERATOR:
        return torch.accelerator.current_accelerator()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    raise RuntimeError("No available accelerator detected.")


def synchronize() -> None:
    """Wait for all kernels on the current accelerator to complete.

    No-op if no accelerator is available.
    """
    if _HAS_TORCH_ACCELERATOR:
        torch.accelerator.synchronize()
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
