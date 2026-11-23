"""Configuration for the audio-to-sheet pipeline.

Precedence (later wins): built-in defaults → quality preset → TOML file
(`--config`) → environment variables → explicit CLI options.

Environment variables:
    MUSCRIPTOR_DEVICE         "auto", "cpu", "cuda", "cuda:1", ...
    MUSCRIPTOR_MODEL          transcription model: small | medium | large | path | URL
    MUSCRIPTOR_MUSESCORE      MuseScore 4 executable (also used by `--format sheets`)
    MUSCRIPTOR_OUTPUT_DIR     default parent directory for outputs
    MUSCRIPTOR_SEPARATION     "demucs" | "none"
    TORCH_HOME                where Demucs caches its weights (torch.hub)
    HF_HOME / HF_HUB_OFFLINE  HuggingFace cache / offline mode for MuScriptor weights

Example `sheet.toml`:

    quality = "standard"
    device = "cuda"

    [separation]
    method = "demucs"
    model = "htdemucs_6s"

    [transcription]
    model = "medium"
    stems = ["vocals", "bass"]

    [harmony]
    audio_chords = true

    [render]
    musescore = "C:/Program Files/MuseScore 4/bin/MuseScore4.exe"
    part_pdfs = true
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Quality = Literal["draft", "standard", "high"]


@dataclass
class SeparationConfig:
    method: Literal["demucs", "none"] = "demucs"
    # htdemucs_6s: drums, bass, other, vocals, guitar, piano (MIT weights).
    model: str = "htdemucs_6s"
    # Random-shift test-time augmentation; >0 is slower but cleaner.
    shifts: int = 0
    overlap: float = 0.25
    # A stem quieter than this (dB relative to the mix) is treated as absent.
    silence_db: float = -32.0


@dataclass
class TranscriptionConfig:
    model: str | None = None  # small | medium | large | path | URL (None = medium)
    dtype: str | None = None
    beam_size: int = 1
    # Stems re-transcribed with MuScriptor (instrument-constrained); their
    # notes replace that instrument family's notes from the full mix.
    stems: list[str] = field(default_factory=lambda: ["vocals", "bass"])
    # A stem transcription with fewer notes is ignored (the mix's notes stay).
    min_stem_notes: int = 30


@dataclass
class RhythmConfig:
    beats_per_bar: int | None = None  # override the detected meter
    beat_scale: float = 1.0  # 2 = detected tempo was half, 0.5 = double
    bpm: float | None = None  # force a constant tempo
    first_downbeat: float | None = None
    swing: Literal["auto", "on", "off"] = "auto"
    allow_triplets: bool = True
    finest: int = 4  # finest binary subdivision (4 = sixteenths)
    # Hold notes across silences up to this many beats (sustain / let ring).
    fill_gaps_beats: float = 0.5
    fill_gaps_beats_keyboard: float = 1.0


@dataclass
class CleanupConfig:
    min_part_notes: int = 40
    min_duration: float = 0.045
    min_duration_sustained: float = 0.07


@dataclass
class HarmonyConfig:
    audio_chords: bool = True  # lv-chordia on the mix
    chord_dict: str = "submission"
    min_chord_beats: int = 2
    local_keys: bool = True
    sections: bool = True


@dataclass
class RenderConfig:
    musescore: str | None = None  # path; None = $MUSCRIPTOR_MUSESCORE / auto-detect
    pdf: bool = True
    part_pdfs: bool = True


@dataclass
class PipelineConfig:
    quality: Quality = "standard"
    device: str = "auto"
    output_dir: str | None = None
    title: str | None = None
    # Reuse artifacts already in the output directory (stems, transcriptions,
    # audio chords) instead of recomputing them.
    resume: bool = True
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    rhythm: RhythmConfig = field(default_factory=RhythmConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    harmony: HarmonyConfig = field(default_factory=HarmonyConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    @property
    def torch_device(self) -> str | None:
        return None if self.device == "auto" else self.device


def apply_preset(cfg: PipelineConfig, quality: Quality) -> None:
    """Quality presets: model size and how much separation work is done."""
    cfg.quality = quality
    if quality == "draft":
        cfg.transcription.model = cfg.transcription.model or "small"
        cfg.separation.method = "none"
        cfg.transcription.stems = []
        cfg.render.part_pdfs = False
    elif quality == "high":
        cfg.transcription.model = cfg.transcription.model or "large"
        cfg.separation.shifts = max(cfg.separation.shifts, 1)
        cfg.transcription.stems = ["vocals", "bass", "drums", "piano", "guitar"]


def _merge(obj: Any, data: dict[str, Any], where: str) -> None:
    names = {f.name: f for f in dataclasses.fields(obj)}
    for key, value in data.items():
        if key not in names:
            raise ValueError(f"unknown config key {where}{key!r}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"config {where}{key} must be a table")
            _merge(current, value, f"{where}{key}.")
        else:
            setattr(obj, key, value)


def load_config(
    path: str | Path | None = None, quality: Quality | None = None
) -> PipelineConfig:
    cfg = PipelineConfig()
    data: dict[str, Any] = {}
    if path is not None:
        try:
            import tomllib
        except ImportError as e:  # Python 3.10
            raise RuntimeError("TOML config files need Python 3.11+") from e
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    preset = quality or data.get("quality") or "standard"
    apply_preset(cfg, preset)
    _merge(cfg, {k: v for k, v in data.items() if k != "quality"}, "")

    env = os.environ
    if env.get("MUSCRIPTOR_DEVICE"):
        cfg.device = env["MUSCRIPTOR_DEVICE"]
    if env.get("MUSCRIPTOR_MODEL"):
        cfg.transcription.model = env["MUSCRIPTOR_MODEL"]
    if env.get("MUSCRIPTOR_OUTPUT_DIR") and not cfg.output_dir:
        cfg.output_dir = env["MUSCRIPTOR_OUTPUT_DIR"]
    if env.get("MUSCRIPTOR_SEPARATION"):
        cfg.separation.method = env["MUSCRIPTOR_SEPARATION"]  # type: ignore[assignment]
    return cfg


def config_to_dict(cfg: PipelineConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)
