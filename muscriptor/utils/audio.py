"""Audio loading and resampling utilities. WAV is handled by the stdlib;
other formats fall back to `soundfile`."""

import wave
from pathlib import Path
from typing import IO

import numpy as np
import torch

from muscriptor.utils.resample import resample_frac


def _read_wav_file(source) -> tuple[torch.Tensor, int]:
    """Load a PCM WAV file using the stdlib `wave` module.

    `source` may be a filesystem path or a binary file-like object.

    Returns:
        (wav, sr) where wav has shape [C, T] and is float32 in [-1, 1].
    """
    if hasattr(source, "read"):
        opened = wave.open(source, "rb")
    else:
        opened = wave.open(str(source), "rb")
    with opened as wf:
        n_channels = wf.getnchannels()
        sr = wf.getframerate()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        as_int32 = (
            bytes_[:, 0].astype(np.int32)
            | (bytes_[:, 1].astype(np.int32) << 8)
            | (bytes_[:, 2].astype(np.int32) << 16)
        )
        as_int32 = np.where(as_int32 >= (1 << 23), as_int32 - (1 << 24), as_int32)
        data = as_int32.astype(np.float32) / float(1 << 23)
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    data = data.reshape(-1, n_channels)
    return torch.from_numpy(np.ascontiguousarray(data.T)), sr


def _read_non_wav_file(source: str | Path | IO[bytes]) -> tuple[torch.Tensor, int]:
    """Load a non-WAV audio file using `soundfile`.

    `source` may be a filesystem path or a binary file-like object (e.g. an
    ``io.BytesIO`` of an uploaded file), since libsndfile reads either.

    Returns:
        (wav, sr) where wav has shape [C, T] and is float32 in [-1, 1].
    """
    try:
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "soundfile is required to read non-WAV audio files. "
            "Install with: `pip install soundfile` or `uvx --with soundfile`"
        ) from e

    target = str(source) if isinstance(source, (str, Path)) else source
    data, sample_rate = sf.read(target, dtype="float32")
    if data.ndim == 1:
        data = data[:, None]
    wav = torch.from_numpy(np.ascontiguousarray(data.T))
    return wav, sample_rate


def resample(
    waveform: torch.Tensor,
    orig_freq: int,
    new_freq: int,
) -> torch.Tensor:
    """Sinc resampler via julius `resample_frac`. Operates along the last dim."""
    if orig_freq == new_freq:
        return waveform
    return resample_frac(waveform, int(orig_freq), int(new_freq))


def load_audio(path: str | Path, target_sr: int = 16000) -> torch.Tensor:
    """Load an audio file and return a mono float32 tensor at target_sr.

    PCM WAV files are read with the stdlib `wave` module. Other formats (mp3,
    flac, ogg, …) are decoded via `soundfile`. Dispatch is by content, not
    file extension, so misnamed files (e.g. an MP3 upload saved as .wav) still
    load.

    Returns:
        Tensor of shape [1, T] at target_sr.
    """
    filepath = Path(path)
    try:
        wav, sr = _read_wav_file(str(filepath))
    except (wave.Error, EOFError):
        wav, sr = _read_non_wav_file(str(filepath))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = resample(wav, sr, target_sr)
    return wav
