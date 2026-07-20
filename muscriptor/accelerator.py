"""Device-agnostic accelerator helpers.

``torch.accelerator`` (``is_available``/``current_accelerator``/``synchronize``)
was only added in PyTorch 2.6; on older versions we fall back to checking CUDA
then MPS directly, in that order.
"""

import platform

from packaging.version import Version

import torch

_HAS_TORCH_ACCELERATOR = Version(torch.__version__.split("+")[0]) >= Version("2.6")


def _mps_available() -> bool:
    """Whether MPS is available and worth auto-selecting.

    torch <= 2.2 also reports MPS as available on Intel Macs with AMD GPUs, a
    backend that was never solid and has since been abandoned. Passing
    ``device="mps"`` explicitly still works there for those who want to try
    it; this only affects auto-detection.
    """
    return torch.backends.mps.is_available() and platform.machine() == "arm64"


def is_available() -> bool:
    """Whether an accelerator (GPU) is available."""
    if _HAS_TORCH_ACCELERATOR:
        return torch.accelerator.is_available()
    return torch.cuda.is_available() or _mps_available()


def current_accelerator() -> torch.device:
    """The current accelerator device.

    Raises ``RuntimeError`` if no accelerator is available; check
    :func:`is_available` first.
    """
    if _HAS_TORCH_ACCELERATOR:
        return torch.accelerator.current_accelerator()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
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
    elif _mps_available():
        torch.mps.synchronize()
