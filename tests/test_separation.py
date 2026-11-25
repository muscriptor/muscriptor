"""Stem handling around source separation (no separator model is run)."""

import json
from pathlib import Path

import numpy as np
import pytest

from muscriptor.sheet.config import (
    PipelineConfig,
    load_config,
    resolve_separation_method,
    resolve_stems,
)
from muscriptor.sheet.instruments import STEM_GROUPS
from muscriptor.sheet.ir import PerformanceNote
from muscriptor.sheet.separation import (
    Activity,
    cached_stems,
    describe_stems,
    ingest_audio,
    run_audio_separator,
    separator_signature,
    write_manifest,
)
from muscriptor.sheet.transcription import gate_notes, merge_stem_notes

sf = pytest.importorskip("soundfile")
SR = 44100


def _tone(seconds, freq=440.0, amp=0.3, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    x = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([x, x], 1)


def _note(pitch, onset, offset, inst="violin", source="other"):
    return PerformanceNote(pitch, onset, offset, inst, source=source)


def test_other_stem_cannot_be_piano_or_guitar():
    for group in ("acoustic_piano", "electric_piano", "acoustic_guitar", "voice"):
        assert group not in STEM_GROUPS["other"]
    for group in ("violin", "cello", "string_ensemble", "flutes", "trumpet"):
        assert group in STEM_GROUPS["other"]


def test_resolve_stems():
    avail = ["bass", "drums", "guitar", "other", "piano", "vocals", "mystery"]
    assert resolve_stems(["auto", "-drums"], avail) == [
        "bass",
        "guitar",
        "other",
        "piano",
        "vocals",
    ]
    assert resolve_stems(["vocals", "bass"], avail) == ["vocals", "bass"]
    assert resolve_stems([], avail) == []


def test_auto_method_falls_back_to_demucs(monkeypatch):
    monkeypatch.delenv("MUSCRIPTOR_SEPARATOR_PYTHON", raising=False)
    cfg = PipelineConfig()
    assert resolve_separation_method(cfg) == "demucs"
    cfg.separation.method = "none"
    assert resolve_separation_method(cfg) == "none"


def test_auto_method_uses_configured_env(monkeypatch, tmp_path):
    py = tmp_path / "python.exe"
    py.write_text("")
    monkeypatch.setenv("MUSCRIPTOR_SEPARATOR_PYTHON", str(py))
    cfg = load_config(None, "standard")
    assert cfg.separation.python == str(py)
    assert resolve_separation_method(cfg) == "roformer"


def test_activity_level_and_gate():
    # Active for the first 2 s, then 50 dB down (bleed only).
    act = Activity(0.5, np.array([-5.0] * 4 + [-50.0] * 4))
    assert act.level(0.1, 0.4) == -5.0
    assert act.level(3.0, 3.5) == -50.0
    assert act.active_ratio(-30) == 0.5
    notes = [_note(67, 0.5, 1.0), _note(67, 3.0, 3.6), _note(69, 1.9, 2.4)]
    kept = gate_notes(notes, act, -30.0, "other")
    # The note straddling the boundary is kept (the pad reaches the active part).
    assert [n.onset for n in kept] == [0.5, 1.9]
    assert gate_notes(notes, None, -30.0) == notes


def test_other_stem_replaces_mix_strings_labelled_piano():
    mix = [
        _note(60, 0.0, 1.0, "acoustic_piano", "mix"),  # really the violin
        _note(48, 0.0, 1.0, "acoustic_piano", "mix"),
    ]
    stems = {
        "piano": [_note(48, 0.0, 1.0, "acoustic_piano", "piano")] * 30,
        "other": [_note(60, 0.0, 1.0, "violin", "other")] * 30,
    }
    merged = merge_stem_notes(mix, stems, min_stem_notes=10)
    assert {n.source for n in merged} == {"piano", "other"}
    assert {(n.pitch, n.instrument) for n in merged} == {
        (48, "acoustic_piano"),
        (60, "violin"),
    }


def test_ingest_and_describe(tmp_path):
    src = tmp_path / "song.wav"
    sf.write(str(src), _tone(2.0)[:, 0], 22050)  # mono, 22.05 kHz
    out = ingest_audio(src, tmp_path / "out" / "input.wav")
    info = sf.info(str(out))
    assert (info.samplerate, info.channels) == (SR, 2)
    assert ingest_audio(src, out) == out  # reused

    loud = tmp_path / "loud.wav"
    quiet = tmp_path / "quiet.wav"
    x = _tone(info.duration)
    sf.write(str(loud), x, SR)
    q = x * 0.0
    q[: SR // 2] = x[: SR // 2] * 0.5
    sf.write(str(quiet), q, SR)
    stems = {s.name: s for s in describe_stems({"a": loud, "b": quiet}, out)}
    assert abs(stems["a"].level_db) < 1.5
    assert stems["b"].activity.db[0] > -10
    assert stems["b"].activity.db[-1] < -60


def test_cached_stems_need_matching_manifest(tmp_path):
    mix = tmp_path / "input.wav"
    sf.write(str(mix), _tone(1.0), SR)
    stems = tmp_path / "stems"
    stems.mkdir()
    for name in ("vocals", "drums", "bass", "other", "piano", "guitar"):
        sf.write(str(stems / f"{name}.wav"), _tone(1.0, amp=0.05), SR)
    cfg = PipelineConfig().separation
    cfg.method = "demucs"  # the pipeline resolves "auto" first
    demucs = separator_signature(cfg)
    # Pre-manifest directories are Demucs output.
    assert cached_stems(stems, mix, demucs) is not None
    cfg.method = "roformer"
    rof = separator_signature(cfg)
    assert rof["model"] == "BS-Roformer-SW.ckpt"
    assert cached_stems(stems, mix, rof) is None
    write_manifest(stems, rof)
    assert len(cached_stems(stems, mix, rof)) == 6
    assert cached_stems(stems, mix, demucs) is None
    assert json.loads((stems / "manifest.json").read_text())["method"] == "roformer"


def test_run_audio_separator_parses_driver_output(tmp_path, monkeypatch):
    import subprocess

    def fake_run(cmd, **kw):
        req = json.loads(Path(cmd[-1]).read_text(encoding="utf-8"))
        assert req["model"] == "BS-Roformer-SW.ckpt"
        stems = {
            "vocals": str(tmp_path / "x_(Vocals).wav"),
            "instrumental": str(tmp_path / "x_(Instrumental).wav"),
            "no_woodwinds": str(tmp_path / "x_(No Woodwinds).wav"),
        }
        return subprocess.CompletedProcess(
            cmd, 0, stdout="noise\nMUSCRIPTOR_STEMS " + json.dumps(stems), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = run_audio_separator(
        "python", "BS-Roformer-SW.ckpt", tmp_path / "in.wav", tmp_path / "w"
    )
    assert set(out) == {"vocals", "other", "harmonic_other"}


class _FakeModel:
    """Labels every note with the first allowed group; records the calls."""

    def __init__(self, labels):
        self.labels = labels  # what an unconditioned pass "hears"
        self.calls = []

    def transcribe(self, audio, instruments=None, beam_size=1, **kw):
        from muscriptor.events import NoteEndEvent, NoteStartEvent

        cond = kw.get("condition_on_instruments", True)
        self.calls.append((list(instruments or []), cond))
        allowed = set(instruments or self.labels)
        labels = [g for g in self.labels if g in allowed] or sorted(allowed)
        for i, g in enumerate(labels * 30):
            start = NoteStartEvent(pitch=60, start_time=i * 0.5, index=i, instrument=g)
            yield start
            yield NoteEndEvent(end_time=i * 0.5 + 0.4, start_event=start)


def test_transcribe_stem_detects_then_conditions():
    from muscriptor.sheet.transcription import detect_instruments, transcribe_stem

    model = _FakeModel(["violin", "viola", "cello"])
    notes, detected = transcribe_stem(
        model, "other.wav", "other", list(STEM_GROUPS["other"])
    )
    assert detected == ["violin", "viola", "cello"]
    # Pass 1: whole family as a mask only; pass 2: the detected set, conditioned.
    assert model.calls[0] == (list(STEM_GROUPS["other"]), False)
    assert model.calls[1] == (detected, True)
    assert {n.source for n in notes} == {"other"}

    single = _FakeModel(["voice"])
    _, det = transcribe_stem(single, "v.wav", "vocals", ["voice"])
    assert det == ["voice"] and len(single.calls) == 1

    few = [_note(60, i, i + 1, "violin") for i in range(50)] + [
        _note(60, i, i + 1, "flutes") for i in range(3)
    ]
    assert detect_instruments(few) == ["violin"]


def test_silent_stem_drops_family_from_mix():
    mix = [_note(64, i, i + 0.5, "acoustic_guitar", "mix") for i in range(5)] + [
        _note(40, 0, 1, "electric_bass", "mix")
    ]
    merged = merge_stem_notes(mix, {}, min_stem_notes=10, absent={"guitar"})
    assert [n.instrument for n in merged] == ["electric_bass"]


def test_config_file_with_bom(tmp_path):
    path = tmp_path / "sheet.toml"
    path.write_bytes(b'\xef\xbb\xbf[cleanup]\nstrings_articulation = "legato"\n')
    assert load_config(path).cleanup.strings_articulation == "legato"
