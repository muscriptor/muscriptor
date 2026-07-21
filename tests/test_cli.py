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


def test_batching_requires_disabling_prelude_forcing(patched_model, fake_audio):
    """--batch-size > 1 without --no-prelude-forcing is rejected up front,
    before any model is loaded."""
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        ["transcribe", str(fake_audio), "-b", "4", "-f", "jsonl", "-o", "-"],
    )
    assert result.exit_code == 1
    assert "--no-prelude-forcing" in result.stderr
    assert "Loading model" not in result.stderr
    assert _FakeModel.last_kwargs is None


def test_batching_allowed_with_prelude_forcing_disabled(patched_model, fake_audio):
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        [
            "transcribe",
            str(fake_audio),
            "-b",
            "4",
            "--no-prelude-forcing",
            "-f",
            "jsonl",
            "-o",
            "-",
        ],
    )
    assert result.exit_code == 0, result.stderr
    assert _FakeModel.last_kwargs["batch_size"] == 4
    assert _FakeModel.last_kwargs["prelude_forcing"] is False


def test_instruments_passed_to_model(patched_model, fake_audio):
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        [
            "transcribe",
            str(fake_audio),
            "--instruments",
            "violin,drums",
            "-f",
            "jsonl",
            "-o",
            "-",
        ],
    )
    assert result.exit_code == 0, result.stderr
    assert _FakeModel.last_kwargs["instruments"] == ["violin", "drums"]


def test_generation_budget_passed_to_model(patched_model, fake_audio):
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        [
            "transcribe",
            str(fake_audio),
            "--max-generation-tokens",
            "512",
            "-f",
            "jsonl",
            "-o",
            "-",
        ],
    )

    assert result.exit_code == 0, result.stderr
    assert _FakeModel.last_kwargs["max_generation_tokens"] == 512


def test_generation_budget_must_be_positive_before_model_load(
    patched_model, fake_audio
):
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        [
            "transcribe",
            str(fake_audio),
            "--max-generation-tokens",
            "0",
            "-f",
            "jsonl",
            "-o",
            "-",
        ],
    )

    assert result.exit_code != 0
    assert "x>=1" in result.stderr
    assert "Loading model" not in result.stderr
    assert _FakeModel.last_kwargs is None


def test_strict_jsonl_file_preserves_existing_output_on_failure(
    monkeypatch, fake_audio, tmp_path
):
    class _FailingModel(_FakeModel):
        def transcribe(self, **kwargs):
            type(self).last_kwargs = kwargs
            yield NoteStartEvent(
                pitch=60,
                start_time=0.0,
                index=0,
                instrument="piano",
            )
            raise RuntimeError("did not emit EOS")

    monkeypatch.setattr(main_mod, "TranscriptionModel", _FailingModel)
    output = tmp_path / "out.jsonl"
    output.write_text("previous output\n")

    result = CliRunner().invoke(
        main_mod.app,
        [
            "transcribe",
            str(fake_audio),
            "--strict-eos",
            "-f",
            "jsonl",
            "-o",
            str(output),
        ],
    )

    assert result.exit_code == 1
    assert output.read_text() == "previous output\n"
    assert not list(tmp_path.glob(".out.jsonl.*.tmp"))


def test_strict_jsonl_stdout_emits_nothing_on_failure(monkeypatch, fake_audio):
    class _FailingModel(_FakeModel):
        def transcribe(self, **kwargs):
            type(self).last_kwargs = kwargs
            yield NoteStartEvent(
                pitch=60,
                start_time=0.0,
                index=0,
                instrument="piano",
            )
            raise RuntimeError("did not emit EOS")

    monkeypatch.setattr(main_mod, "TranscriptionModel", _FailingModel)

    result = CliRunner().invoke(
        main_mod.app,
        [
            "transcribe",
            str(fake_audio),
            "--strict-eos",
            "-f",
            "jsonl",
            "-o",
            "-",
        ],
    )

    assert result.exit_code == 1
    assert result.stdout == ""


def test_strict_jsonl_file_is_replaced_only_after_success(
    monkeypatch, fake_audio, tmp_path
):
    output = tmp_path / "out.jsonl"
    output.write_text("previous output\n")
    observed_during_transcription: list[str] = []

    class _ObservingModel(_FakeModel):
        def transcribe(self, **kwargs):
            type(self).last_kwargs = kwargs
            observed_during_transcription.append(output.read_text())
            yield from super().transcribe(**kwargs)
            observed_during_transcription.append(output.read_text())

    monkeypatch.setattr(main_mod, "TranscriptionModel", _ObservingModel)

    result = CliRunner().invoke(
        main_mod.app,
        [
            "transcribe",
            str(fake_audio),
            "--strict-eos",
            "-f",
            "jsonl",
            "-o",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.stderr
    assert observed_during_transcription == ["previous output\n", "previous output\n"]
    assert len(output.read_text().splitlines()) == 4
    assert not list(tmp_path.glob(".out.jsonl.*.tmp"))
