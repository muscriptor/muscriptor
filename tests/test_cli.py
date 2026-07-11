"""CLI tests — checks that `-o -` writes a clean JSONL stream to stdout
and that all progress/timing chatter goes to stderr.

Patches the model loader with a fake so the test stays hermetic.
"""

import json
import wave
from pathlib import Path

import pytest
from typer.testing import CliRunner

import muscriptor.main as main_mod
from muscriptor.events import NoteEndEvent, NoteStartEvent


class _FakeInner:
    """Stand-in for the inner torch module; the CLI casts it with `.to(...)`."""

    def to(self, *_args, **_kwargs):
        return self


class _FakeModel:
    # kwargs of the most recent transcribe() call (reset by `patched_model`).
    last_kwargs: dict | None = None

    def __init__(self):
        self._model = _FakeInner()

    @classmethod
    def load_model(cls, **_):
        return cls()

    def transcribe(self, **kwargs):
        type(self).last_kwargs = kwargs
        s0 = NoteStartEvent(pitch=60, start_time=0.0, index=0, instrument="piano")
        s1 = NoteStartEvent(pitch=64, start_time=0.5, index=1, instrument="guitar")
        yield s0
        yield NoteEndEvent(end_time=0.4, start_event=s0)
        yield s1
        yield NoteEndEvent(end_time=0.9, start_event=s1)

    def transcribe_to_midi(self, **kwargs):
        type(self).last_kwargs = kwargs
        return b"FAKE_MIDI"


@pytest.fixture
def fake_audio(tmp_path: Path) -> Path:
    p = tmp_path / "silent.wav"
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 100)
    return p


@pytest.fixture
def patched_model(monkeypatch):
    _FakeModel.last_kwargs = None
    monkeypatch.setattr(main_mod, "TranscriptionModel", _FakeModel)


def test_jsonl_to_stdout_is_pure_jsonl(patched_model, fake_audio):
    """`-o -` with --format jsonl writes only JSON objects (one per line) to stdout."""
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        ["transcribe", str(fake_audio), "-f", "jsonl", "-o", "-"],
    )
    assert result.exit_code == 0, result.stderr

    # Every non-empty stdout line must be a complete JSON object.
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert len(lines) == 4
    parsed = [json.loads(ln) for ln in lines]
    assert parsed[0] == {
        "type": "start",
        "pitch": 60,
        "start_time": 0.0,
        "index": 0,
        "instrument": "piano",
    }
    assert parsed[1] == {"type": "end", "end_time": 0.4, "start_event_index": 0}


def test_jsonl_stdout_has_no_chatter(patched_model, fake_audio):
    """Stdout must contain nothing except JSON — no "Loading model…" etc."""
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        ["transcribe", str(fake_audio), "-f", "jsonl", "-o", "-"],
    )
    assert result.exit_code == 0
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        # Throws if the line isn't valid JSON — proves no banner / "Saved to" /
        # timing line leaked through.
        json.loads(line)


def test_progress_messages_go_to_stderr(patched_model, fake_audio):
    """Loading / transcribing banners must appear on stderr, never stdout."""
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        ["transcribe", str(fake_audio), "-f", "jsonl", "-o", "-"],
    )
    assert result.exit_code == 0
    assert "Loading model" in result.stderr
    assert "Transcribing" in result.stderr
    assert "Loading model" not in result.stdout
    assert "Transcribing" not in result.stdout


def test_jsonl_to_file_keeps_progress_on_stderr(patched_model, fake_audio, tmp_path):
    """Even when writing to a file, banners and "Saved JSONL to …" go to stderr."""
    out = tmp_path / "out.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        ["transcribe", str(fake_audio), "-f", "jsonl", "-o", str(out)],
    )
    assert result.exit_code == 0
    assert result.stdout == ""
    assert "Saved JSONL" in result.stderr
    assert out.read_text().splitlines() == [
        json.dumps(d)
        for d in [
            {
                "type": "start",
                "pitch": 60,
                "start_time": 0.0,
                "index": 0,
                "instrument": "piano",
            },
            {"type": "end", "end_time": 0.4, "start_event_index": 0},
            {
                "type": "start",
                "pitch": 64,
                "start_time": 0.5,
                "index": 1,
                "instrument": "guitar",
            },
            {"type": "end", "end_time": 0.9, "start_event_index": 1},
        ]
    ]


def test_strict_instruments_requires_instruments(patched_model, fake_audio):
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        ["transcribe", str(fake_audio), "--strict-instruments", "-f", "jsonl", "-o", "-"],
    )
    assert result.exit_code == 1
    assert "--strict-instruments requires --instruments" in result.stderr
    assert _FakeModel.last_kwargs is None


def test_strict_instruments_passed_to_model(patched_model, fake_audio):
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        [
            "transcribe",
            str(fake_audio),
            "--instruments",
            "violin,drums",
            "--strict-instruments",
            "-f",
            "jsonl",
            "-o",
            "-",
        ],
    )
    assert result.exit_code == 0, result.stderr
    assert _FakeModel.last_kwargs["instruments"] == ["violin", "drums"]
    assert _FakeModel.last_kwargs["strict_instruments"] is True


def test_tempo_bpm_passed_to_model(patched_model, fake_audio, tmp_path):
    out = tmp_path / "out.mid"
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        ["transcribe", str(fake_audio), "--tempo-bpm", "87.5", "-o", str(out)],
    )
    assert result.exit_code == 0, result.stderr
    assert out.read_bytes() == b"FAKE_MIDI"
    assert _FakeModel.last_kwargs["tempo_bpm"] == 87.5


def test_tempo_bpm_default_is_omitted(patched_model, fake_audio, tmp_path):
    """Without --tempo-bpm the model's own default (120) applies."""
    out = tmp_path / "out.mid"
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        ["transcribe", str(fake_audio), "-o", str(out)],
    )
    assert result.exit_code == 0, result.stderr
    assert "tempo_bpm" not in _FakeModel.last_kwargs


def test_tempo_bpm_requires_midi_format(patched_model, fake_audio):
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        ["transcribe", str(fake_audio), "--tempo-bpm", "90", "-f", "jsonl", "-o", "-"],
    )
    assert result.exit_code == 1
    assert "--tempo-bpm requires --format midi" in result.stderr
