"""Source separation: split the mix into instrument stems before transcription.

Separating first keeps the bass, drums and vocals from bleeding into each
other's transcription. The original mix is still what the beat tracker and
the audio chord recognizer listen to (separation artefacts hurt them more
than the overlap does).

Backend: Demucs ``htdemucs_6s`` (Défossez et al., MIT code and weights) —
six stems: drums, bass, other, vocals, guitar, piano. The research document's
first choice, BS-RoFormer-SW via ``audio-separator``, pins an older
``rotary-embedding-torch`` than ``beat_this`` needs, so it cannot share an
environment with this project; the :class:`Separator` interface keeps it a
drop-in replacement for later.

Weights (~80 MB for htdemucs_6s) are fetched once by torch.hub into
``$TORCH_HOME`` (default ``~/.cache/torch``); run ``muscriptor download-models``
to do that ahead of time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class Stem:
    name: str
    path: Path
    # RMS level relative to the mix, in dB: how much of the song this stem is.
    level_db: float


class Separator(Protocol):
    name: str

    def separate(self, audio: Path, out_dir: Path) -> list[Stem]: ...


def _read_stereo(path: Path, sr: int) -> torch.Tensor:
    """[2, T] float32 at `sr`."""
    import soundfile as sf

    from muscriptor.utils.audio import resample

    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(np.ascontiguousarray(data.T))
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    if file_sr != sr:
        wav = resample(wav, file_sr, sr)
    return wav


def _rms_db(x: torch.Tensor) -> float:
    return float(20 * torch.log10(x.pow(2).mean().sqrt() + 1e-9))


class DemucsSeparator:
    name = "demucs"

    def __init__(
        self,
        model: str = "htdemucs_6s",
        device: str | None = None,
        shifts: int = 0,
        overlap: float = 0.25,
    ):
        self.model_name = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.shifts = shifts
        self.overlap = overlap

    def load(self):
        try:
            from demucs.pretrained import get_model
        except ImportError as e:
            raise RuntimeError(
                "Source separation needs Demucs: install the 'sheet' extra "
                "(uv pip install 'muscriptor[sheet]') or pass --separation none."
            ) from e
        model = get_model(self.model_name)
        model.eval()
        return model

    def separate(self, audio: Path, out_dir: Path) -> list[Stem]:
        import soundfile as sf
        from demucs.apply import apply_model

        model = self.load()
        sr = model.samplerate
        mix = _read_stereo(audio, sr)
        ref = mix.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            sources = apply_model(
                model,
                ((mix - mean) / std)[None],
                shifts=self.shifts,
                split=True,
                overlap=self.overlap,
                progress=False,
                device=self.device,
            )[0]
        sources = sources * std + mean
        mix_db = _rms_db(mix)
        out_dir.mkdir(parents=True, exist_ok=True)
        stems = []
        for name, wav in zip(model.sources, sources):
            path = out_dir / f"{name}.wav"
            sf.write(str(path), wav.cpu().numpy().T, sr, subtype="PCM_16")
            stems.append(Stem(name, path, _rms_db(wav) - mix_db))
            logger.info("separation: %-7s %+.1f dB", name, stems[-1].level_db)
        return stems


def stems_from_dir(out_dir: Path, mix: Path | None = None) -> list[Stem]:
    """Stems a previous run left in `out_dir` (for --resume)."""
    stems = []
    mix_db = None
    if mix is not None:
        mix_db = _rms_db(_read_stereo(mix, 44100))
    for path in sorted(out_dir.glob("*.wav")):
        level = 0.0
        if mix_db is not None:
            level = _rms_db(_read_stereo(path, 44100)) - mix_db
        stems.append(Stem(path.stem, path, level))
    return stems


def make_separator(
    method: str, model: str, device: str | None, shifts: int, overlap: float
) -> Separator | None:
    if method == "none":
        return None
    if method == "demucs":
        return DemucsSeparator(model, device, shifts, overlap)
    raise ValueError(
        f"unknown separation method {method!r} (expected 'demucs' or 'none')"
    )
