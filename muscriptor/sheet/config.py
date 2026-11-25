"""Configuration for the audio-to-sheet pipeline.

Precedence (later wins): built-in defaults → quality preset → TOML file
(`--config`) → environment variables → explicit CLI options.

Environment variables:
    MUSCRIPTOR_DEVICE         "auto", "cpu", "cuda", "cuda:1", ...
    MUSCRIPTOR_MODEL          transcription model: small | medium | large | path | URL
    MUSCRIPTOR_MUSESCORE      MuseScore 4 executable (also used by `--format sheets`)
    MUSCRIPTOR_OUTPUT_DIR     default parent directory for outputs
    MUSCRIPTOR_SEPARATION     "auto" | "demucs" | "roformer" | "none"
    MUSCRIPTOR_SEPARATOR_PYTHON  python of the audio-separator environment (roformer)
    MUSCRIPTOR_SEPARATOR_MODELS  where audio-separator models are cached
    TORCH_HOME                where Demucs caches its weights (torch.hub)
    HF_HOME / HF_HUB_OFFLINE  HuggingFace cache / offline mode for MuScriptor weights

Example `sheet.toml`:

    quality = "standard"
    device = "cuda"

    [separation]
    method = "roformer"            # or "demucs"
    python = "C:/sep-env/Scripts/python.exe"
    roformer_model = "BS-Roformer-SW.ckpt"

    [transcription]
    model = "medium"
    stems = ["auto", "-drums"]

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
    # auto = roformer when a separator environment is configured
    # (python / $MUSCRIPTOR_SEPARATOR_PYTHON), demucs otherwise.
    method: Literal["auto", "demucs", "roformer", "none"] = "auto"
    # Demucs model. htdemucs_6s: drums, bass, other, vocals, guitar, piano.
    model: str = "htdemucs_6s"
    # Random-shift test-time augmentation; >0 is slower but cleaner.
    shifts: int = 0
    overlap: float = 0.25
    # --- roformer (python-audio-separator, own environment) ---
    python: str | None = None  # that environment's interpreter
    model_dir: str | None = None  # model cache ($MUSCRIPTOR_SEPARATOR_MODELS)
    roformer_model: str = "BS-Roformer-SW.ckpt"
    # Optional second pass on the sum of `refine_inputs` (e.g. a strings or
    # woodwinds model); its stems replace those inputs.
    refine_model: str | None = None
    refine_inputs: list[str] = field(default_factory=lambda: ["piano", "other"])
    batch_size: int = 1
    roformer_overlap: int = 8
    autocast: bool = False
    # A stem quieter than this (dB relative to the mix) is treated as absent.
    silence_db: float = -32.0
    # Stem-transcription notes where the stem is this far below the mix
    # (per half second) are bleed from another instrument and are dropped.
    gate_db: float = -30.0


@dataclass
class TranscriptionConfig:
    model: str | None = None  # small | medium | large | path | URL (None = medium)
    dtype: str | None = None
    beam_size: int = 1
    # Stems re-transcribed with MuScriptor (instrument-constrained); their
    # notes replace that instrument family's notes from the full mix.
    # "auto" = every separated stem with a known instrument family;
    # "-name" removes one ("auto", "-drums").
    stems: list[str] = field(default_factory=lambda: ["auto", "-drums"])
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
    # Write a "Strings" section as separate parts: 3 = violin, viola, cello;
    # 2 = violin, cello (sections.py).
    split_strings: bool = True
    string_parts: int = 3
    # Split sustained string notes glued back together: "audio" where the
    # stem shows no new bow stroke, "legato" (every repeat), "off" (articulation.py).
    strings_articulation: Literal["audio", "legato", "off"] = "audio"


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
        cfg.transcription.stems = ["auto"]


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
        # utf-8-sig: Windows PowerShell 5.1 writes UTF-8 files with a BOM.
        data = tomllib.loads(Path(path).read_text(encoding="utf-8-sig"))
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
    if env.get("MUSCRIPTOR_SEPARATOR_PYTHON") and not cfg.separation.python:
        cfg.separation.python = env["MUSCRIPTOR_SEPARATOR_PYTHON"]
    validate_config(cfg)
    return cfg


_CHOICES: dict[str, tuple[object, ...]] = {
    "quality": ("draft", "standard", "high"),
    "separation.method": ("auto", "demucs", "roformer", "none"),
    "cleanup.strings_articulation": ("audio", "legato", "off"),
    "cleanup.string_parts": (2, 3),
    "rhythm.swing": ("auto", "on", "off"),
}


def validate_config(cfg: PipelineConfig) -> None:
    """Raise ValueError on a value no stage accepts (before anything runs)."""
    for key, allowed in _CHOICES.items():
        obj = cfg
        *path, name = key.split(".")
        for part in path:
            obj = getattr(obj, part)
        value = getattr(obj, name)
        if value not in allowed:
            raise ValueError(
                f"{key} = {value!r} is not one of {', '.join(map(repr, allowed))}"
            )


def resolve_separation_method(cfg: PipelineConfig) -> str:
    """'auto' → 'roformer' if its environment is configured, else 'demucs'."""
    if cfg.separation.method != "auto":
        return cfg.separation.method
    from muscriptor.sheet.separation import separator_python_available

    return "roformer" if separator_python_available(cfg.separation.python) else "demucs"


def resolve_stems(requested: list[str], available: list[str]) -> list[str]:
    """Expand "auto" / "-name" in `transcription.stems` against the separated stems."""
    from muscriptor.sheet.instruments import STEM_GROUPS

    out: list[str] = []
    for item in requested:
        if item == "auto":
            out += [s for s in available if s in STEM_GROUPS and s not in out]
        elif item.startswith("-"):
            out = [s for s in out if s != item[1:]]
        elif item not in out:
            out.append(item)
    return out


def config_to_dict(cfg: PipelineConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)
