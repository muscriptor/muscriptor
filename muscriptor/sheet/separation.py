"""Source separation: split the mix into instrument stems before transcription.

Instrument identity has to be settled here, at the audio level. Transcribed
on the full mix, sustained strings under a piano come out labelled as piano,
and no cleanup rule downstream can tell which G4 was bowed. Each stem is
therefore transcribed with MuScriptor's instrument constraint (see
:mod:`muscriptor.sheet.transcription`): the ``other`` stem may not produce
piano, guitar, bass, voice or drums, so the strings and winds keep their own
labels.

The beat tracker and the audio chord recognizer still listen to the original
mix, because separation artefacts hurt them more than the overlap does.

Backends:

``demucs``   Demucs ``htdemucs_6s`` (Défossez et al., MIT code and weights):
             drums, bass, other, vocals, guitar, piano. Same environment as
             MuScriptor. The piano stem is weak.
``roformer`` Any model `python-audio-separator`_ can load, default
             ``BS-Roformer-SW`` (6 stems, much better piano and "other").
             audio-separator pins a ``rotary-embedding-torch`` that conflicts
             with ``beat_this``, so it runs as a subprocess in its *own*
             Python environment (``separation.python`` /
             ``$MUSCRIPTOR_SEPARATOR_PYTHON``). An optional second pass
             (``separation.refine_model``) re-separates the harmonic residual
             (by default piano + other), the cascade from the research notes:
             strip vocals/drums/bass first, then split what overlaps in the
             mid range.

Every stem also gets an *activity curve* (its level relative to the mix, per
second). Notes a stem transcription produces where the stem is near-silent
are separation bleed and are gated out before merging.

.. _python-audio-separator: https://github.com/nomadkaraoke/python-audio-separator
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100
# Activity curve resolution (seconds per frame).
ACTIVITY_HOP = 0.5
MANIFEST = "manifest.json"


@dataclass
class Activity:
    """A stem's level relative to the mix, in dB, per `hop` seconds."""

    hop: float
    db: np.ndarray

    def level(self, t0: float, t1: float, pad: float = 0.25) -> float:
        """Loudest frame overlapping [t0 - pad, t1 + pad]."""
        if len(self.db) == 0:
            return 0.0
        a = max(0, int((t0 - pad) / self.hop))
        b = min(len(self.db), int((t1 + pad) / self.hop) + 1)
        if b <= a:
            a, b = min(a, len(self.db) - 1), min(a, len(self.db) - 1) + 1
        return float(self.db[a:b].max())

    def active_ratio(self, threshold_db: float) -> float:
        return float((self.db >= threshold_db).mean()) if len(self.db) else 0.0


@dataclass
class Stem:
    name: str
    path: Path
    # RMS level relative to the mix, in dB: how much of the song this stem is.
    level_db: float
    activity: Activity | None = field(default=None, repr=False)


class Separator(Protocol):
    name: str

    def separate(self, audio: Path, out_dir: Path) -> list[Stem]: ...


# ---------------------------------------------------------------------------
# Audio helpers


def _read_stereo(path: Path, sr: int = SAMPLE_RATE) -> torch.Tensor:
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


def _write(path: Path, wav: torch.Tensor, sr: int = SAMPLE_RATE) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav.clamp(-1, 1).cpu().numpy().T, sr, subtype="PCM_16")


def _rms_db(x: torch.Tensor) -> float:
    return float(20 * torch.log10(x.pow(2).mean().sqrt() + 1e-9))


def _frame_db(wav: torch.Tensor, sr: int, hop: float) -> np.ndarray:
    """RMS in dB per non-overlapping `hop`-second frame of a [C, T] signal."""
    mono = wav.pow(2).mean(0)
    n = max(1, round(hop * sr))
    frames = mono.shape[-1] // n
    if frames == 0:
        return np.array([_rms_db(wav)])
    power = mono[: frames * n].reshape(frames, n).mean(1)
    return (10 * torch.log10(power + 1e-12)).numpy()


def ingest_audio(audio: Path, out_path: Path) -> Path:
    """Decode `audio` once to 44.1 kHz stereo 16-bit WAV at `out_path`.

    Every later stage (separators, the stem-level comparison, the second
    separation pass) reads this file, so a compressed or odd-rate input is
    decoded and resampled once instead of in memory by each of them.
    """
    audio = Path(audio)
    if (
        out_path.is_file()
        and out_path.stat().st_mtime >= audio.stat().st_mtime
        and out_path.stat().st_size > 44
    ):
        return out_path
    import soundfile as sf

    info = sf.info(str(audio))
    if (
        info.samplerate == SAMPLE_RATE
        and info.channels == 2
        and info.format == "WAV"
        and info.subtype == "PCM_16"
    ):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(audio, out_path)
    else:
        _write(out_path, _read_stereo(audio, SAMPLE_RATE))
    logger.info("ingest: %s → %s (44.1 kHz stereo)", audio.name, out_path)
    return out_path


def describe_stems(
    paths: dict[str, Path], mix: Path, hop: float = ACTIVITY_HOP
) -> list[Stem]:
    """Overall level and activity curve of each stem relative to `mix`."""
    mix_wav = _read_stereo(mix)
    mix_db = _rms_db(mix_wav)
    mix_frames = _frame_db(mix_wav, SAMPLE_RATE, hop)
    stems = []
    for name, path in paths.items():
        wav = _read_stereo(path)
        frames = _frame_db(wav, SAMPLE_RATE, hop)
        n = min(len(frames), len(mix_frames))
        # Relative to the mix in the same frame, but a quiet passage of the
        # whole song is not "inactive": floor the reference at mix level - 20.
        ref = np.maximum(mix_frames[:n], mix_db - 20)
        act = Activity(hop, frames[:n] - ref)
        stems.append(Stem(name, Path(path), _rms_db(wav) - mix_db, act))
        logger.info(
            "separation: %-12s %+.1f dB, active %.0f%% of the song",
            name,
            stems[-1].level_db,
            100 * act.active_ratio(-30.0),
        )
    return stems


# ---------------------------------------------------------------------------
# Demucs (in-process)


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
        paths = {}
        for name, wav in zip(model.sources, sources):
            paths[name] = out_dir / f"{name}.wav"
            _write(paths[name], wav, sr)
        return describe_stems(paths, audio)


# ---------------------------------------------------------------------------
# python-audio-separator (subprocess, own environment)

DRIVER = Path(__file__).with_name("_separator_driver.py")

# Stem names models use for the same thing.
_STEM_ALIASES = {
    "vocal": "vocals",
    "voice": "vocals",
    "drum": "drums",
    "instrument": "other",
    "instrumental": "other",
    "no_vocals": "other",
    "keys": "piano",
    "string": "strings",
    "wind": "winds",
    "woodwinds": "winds",
    # 17_HP-Wind_Inst-UVR (refine_inputs = ["other"]): what is not woodwind.
    "no_woodwinds": "harmonic_other",
}


def default_model_dir() -> str:
    """Where audio-separator models are cached ($MUSCRIPTOR_SEPARATOR_MODELS)."""
    return os.environ.get("MUSCRIPTOR_SEPARATOR_MODELS") or str(
        Path.home() / ".cache" / "muscriptor" / "separator-models"
    )


def separator_python_available(configured: str | None) -> bool:
    try:
        find_separator_python(configured)
    except RuntimeError:
        return False
    return True


def find_separator_python(configured: str | None) -> str:
    """The interpreter of the environment audio-separator is installed in."""
    candidate = configured or os.environ.get("MUSCRIPTOR_SEPARATOR_PYTHON")
    if candidate:
        if not Path(candidate).is_file() and shutil.which(candidate) is None:
            raise RuntimeError(f"separator Python not found: {candidate}")
        return candidate
    raise RuntimeError(
        "--separation roformer runs python-audio-separator in its own Python "
        "environment (its pinned rotary-embedding-torch conflicts with "
        "beat_this). Create one, e.g.\n"
        "  uv venv sep-env --python 3.12\n"
        '  uv pip install --python sep-env "audio-separator[gpu]" torch '
        "--index https://download.pytorch.org/whl/cu128 "
        "--extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match\n"
        "then set MUSCRIPTOR_SEPARATOR_PYTHON to its python (or "
        "[separation] python = ... in the config), or use --separation demucs."
    )


def run_audio_separator(
    python: str,
    model: str,
    audio: Path,
    work_dir: Path,
    model_dir: str | None = None,
    batch_size: int = 1,
    overlap: int = 8,
    autocast: bool = False,
) -> dict[str, Path]:
    """Separate `audio` with `model`; {lower-case stem name: wav path}."""
    req = {
        "input": str(audio),
        "output_dir": str(work_dir),
        "model": model,
        "model_dir": model_dir,
        "batch_size": batch_size,
        "overlap": overlap,
        "autocast": autocast,
    }
    logger.info("separation: %s on %s", model, audio.name)
    work_dir.mkdir(parents=True, exist_ok=True)
    req_path = work_dir / "request.json"
    req_path.write_text(json.dumps(req), encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [python, "-I", str(DRIVER), str(req_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,  # the error is reported below, with the separator's log
    )
    for line in proc.stderr.splitlines()[-200:]:
        logger.debug("audio-separator: %s", line)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise RuntimeError(f"audio-separator failed ({model}):\n{tail}")
    for line in proc.stdout.splitlines():
        if line.startswith("MUSCRIPTOR_STEMS "):
            raw = json.loads(line[len("MUSCRIPTOR_STEMS ") :])
            return {_STEM_ALIASES.get(k, k): Path(v) for k, v in raw.items()}
    raise RuntimeError(f"audio-separator printed no stems:\n{proc.stdout[-2000:]}")


class AudioSeparatorBackend:
    """Cascaded separation with audio-separator models (BS-/Mel-RoFormer, SCNet, …).

    Pass A splits the mix (default BS-Roformer-SW: vocals, drums, bass,
    guitar, piano, other). With `refine_model`, pass B re-separates the sum
    of the `refine_inputs` stems, and its stems replace them.
    """

    name = "roformer"

    def __init__(
        self,
        model: str,
        python: str | None = None,
        model_dir: str | None = None,
        refine_model: str | None = None,
        refine_inputs: tuple[str, ...] = ("piano", "other"),
        batch_size: int = 1,
        overlap: int = 8,
        autocast: bool = False,
    ):
        self.model = model
        self.python = find_separator_python(python)
        self.model_dir = model_dir
        self.refine_model = refine_model
        self.refine_inputs = tuple(refine_inputs)
        self.opts = {
            "model_dir": model_dir,
            "batch_size": batch_size,
            "overlap": overlap,
            "autocast": autocast,
        }

    def separate(self, audio: Path, out_dir: Path) -> list[Stem]:
        work = out_dir / "_work"
        if work.exists():
            shutil.rmtree(work)
        stems = run_audio_separator(
            self.python, self.model, audio, work / "a", **self.opts
        )
        if self.refine_model:
            inputs = [stems[n] for n in self.refine_inputs if n in stems]
            if inputs:
                mixed = sum(_read_stereo(p) for p in inputs)
                residual = work / "harmonic.wav"
                _write(residual, mixed)
                refined = run_audio_separator(
                    self.python, self.refine_model, residual, work / "b", **self.opts
                )
                for n in self.refine_inputs:
                    stems.pop(n, None)
                for n, p in refined.items():
                    # A pass-B "other" is what is left of the harmonic part.
                    stems["harmonic_other" if n == "other" else n] = p
            else:
                logger.warning(
                    "separation: no %s stem to refine", "/".join(self.refine_inputs)
                )
        out_dir.mkdir(parents=True, exist_ok=True)
        final = {}
        for name, path in stems.items():
            dest = out_dir / f"{name}.wav"
            # Normalise to 44.1 kHz stereo PCM16 like every other stem.
            _write(dest, _read_stereo(path))
            final[name] = dest
        shutil.rmtree(work, ignore_errors=True)
        return describe_stems(final, audio)


# ---------------------------------------------------------------------------


def separator_signature(cfg) -> dict:
    """What produced a stems directory (a cached one is reused only if it matches)."""
    sig = {"method": cfg.method, "model": cfg.model}
    if cfg.method == "roformer":
        sig["model"] = cfg.roformer_model
        sig["refine_model"] = cfg.refine_model
        sig["refine_inputs"] = list(cfg.refine_inputs) if cfg.refine_model else None
    return sig


def write_manifest(out_dir: Path, sig: dict) -> None:
    (out_dir / MANIFEST).write_text(json.dumps(sig, indent=1))


def cached_stems(out_dir: Path, mix: Path, sig: dict) -> list[Stem] | None:
    """Stems a previous run with the same separator left in `out_dir`."""
    if not out_dir.is_dir():
        return None
    paths = {p.stem: p for p in sorted(out_dir.glob("*.wav"))}
    if not paths:
        return None
    manifest = out_dir / MANIFEST
    if manifest.is_file():
        if json.loads(manifest.read_text()) != sig:
            logger.info("separation: cached stems are from another separator")
            return None
    elif sig.get("method") != "demucs" or len(paths) < 4:
        # Runs before the manifest existed only used Demucs.
        return None
    return describe_stems(paths, mix)


def stems_from_dir(out_dir: Path, mix: Path | None = None) -> list[Stem]:
    """Stems in `out_dir` (activity is only known if `mix` is given)."""
    paths = {p.stem: p for p in sorted(out_dir.glob("*.wav"))}
    if mix is not None:
        return describe_stems(paths, mix)
    return [Stem(n, p, 0.0) for n, p in paths.items()]


def make_separator(cfg, device: str | None) -> Separator | None:
    """The backend `cfg` (a :class:`~muscriptor.sheet.config.SeparationConfig`) asks for."""
    if cfg.method == "none":
        return None
    if cfg.method == "demucs":
        return DemucsSeparator(cfg.model, device, cfg.shifts, cfg.overlap)
    if cfg.method == "roformer":
        return AudioSeparatorBackend(
            cfg.roformer_model,
            python=cfg.python,
            model_dir=cfg.model_dir or default_model_dir(),
            refine_model=cfg.refine_model,
            refine_inputs=tuple(cfg.refine_inputs),
            batch_size=cfg.batch_size,
            overlap=cfg.roformer_overlap,
            autocast=cfg.autocast,
        )
    raise ValueError(
        f"unknown separation method {cfg.method!r} "
        "(expected 'demucs', 'roformer' or 'none')"
    )


__all__ = [
    "Activity",
    "AudioSeparatorBackend",
    "DemucsSeparator",
    "Separator",
    "Stem",
    "cached_stems",
    "describe_stems",
    "ingest_audio",
    "make_separator",
    "separator_signature",
    "stems_from_dir",
    "write_manifest",
]
