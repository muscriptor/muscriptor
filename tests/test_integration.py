"""Integration tests: load real checkpoint and run inference.

These tests require the safetensors weights file to exist.
They are slow (model is large) but run on CPU with short audio clips.

Run with: pytest tests/test_integration.py -v
"""

import tempfile
from pathlib import Path

import pytest
import torch

from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent
from .conftest import SONG_PATH

# Model constants
_SAMPLE_RATE = 16000
_SEGMENT_DURATION = 5  # seconds — must match transcription_model._SEGMENT_DURATION
# One full segment (5 s) is the natural test size; shorter clips are padded internally.
_DURATION_SEC = _SEGMENT_DURATION
# Use exactly two segments for the song clip so we exercise the multi-chunk path.
_SONG_CLIP_SEC = 10


@pytest.fixture(scope="module")
def silence():
    return torch.zeros(1, _SAMPLE_RATE * _DURATION_SEC)


@pytest.fixture(scope="module")
def noise():
    torch.manual_seed(0)
    return torch.randn(1, _SAMPLE_RATE * _DURATION_SEC) * 0.01


@pytest.fixture(scope="module")
def song_clip():
    """First 10 seconds of filling_the_void.wav resampled to 16 kHz mono."""
    if not SONG_PATH.exists():
        pytest.skip(f"Song not found at {SONG_PATH}")
    from muscriptor.utils.audio import load_audio

    wav = load_audio(SONG_PATH, target_sr=_SAMPLE_RATE)
    return wav[:, : _SAMPLE_RATE * _SONG_CLIP_SEC]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def test_model_loads(transcription_model):
    assert transcription_model is not None
    assert transcription_model._model is not None
    assert transcription_model._tokenizer is not None


def test_model_is_eval(transcription_model):
    assert not transcription_model._model.training


def test_model_device_is_cpu(transcription_model):
    p = next(transcription_model._model.parameters())
    assert p.device.type == "cpu"


# ---------------------------------------------------------------------------
# _load_wav preprocessing
# ---------------------------------------------------------------------------


def test_load_wav_1d_tensor(transcription_model):
    wav1d = torch.zeros(_SAMPLE_RATE)
    out = transcription_model._load_wav(wav1d, sample_rate=_SAMPLE_RATE)
    assert out.shape[0] == 1
    assert out.dim() == 2


def test_load_wav_2d_tensor(transcription_model):
    wav2d = torch.zeros(1, _SAMPLE_RATE)
    out = transcription_model._load_wav(wav2d, sample_rate=_SAMPLE_RATE)
    assert out.shape == (1, _SAMPLE_RATE)


def test_load_wav_stereo_to_mono(transcription_model):
    wav_stereo = torch.zeros(2, _SAMPLE_RATE)
    out = transcription_model._load_wav(wav_stereo, sample_rate=_SAMPLE_RATE)
    assert out.shape[0] == 1


def test_load_wav_resamples(transcription_model):
    wav_44k = torch.zeros(1, 44100)
    out = transcription_model._load_wav(wav_44k, sample_rate=44100)
    assert out.shape[-1] == _SAMPLE_RATE  # resampled to 1 second at 16 kHz


# ---------------------------------------------------------------------------
# _build_conditions
# ---------------------------------------------------------------------------


def test_build_conditions_structure(transcription_model):
    wav = torch.zeros(1, _SAMPLE_RATE)
    conditions = transcription_model._build_conditions(wav)
    assert len(conditions) == 1
    cond = conditions[0]
    assert "self_wav" in cond.wav
    assert "instrument_group" in cond.text
    assert "dataset_name" in cond.text


def test_build_conditions_wav_shape(transcription_model):
    wav = torch.zeros(1, _SAMPLE_RATE * _SEGMENT_DURATION)
    conditions = transcription_model._build_conditions(wav)
    wav_cond = conditions[0].wav["self_wav"]
    assert wav_cond.wav.shape == (1, 1, _SAMPLE_RATE * _SEGMENT_DURATION)
    assert wav_cond.length.item() == _SAMPLE_RATE * _SEGMENT_DURATION


# ---------------------------------------------------------------------------
# transcribe — event stream
# ---------------------------------------------------------------------------


def _collect(model, audio):
    return list(model.transcribe((audio, _SAMPLE_RATE)))


def _assert_start_end_invariants(events):
    starts = [e for e in events if isinstance(e, NoteStartEvent)]
    ends = [e for e in events if isinstance(e, NoteEndEvent)]
    assert len(starts) == len(ends), "every NoteStart must have a matching NoteEnd"
    start_ids = {s.index for s in starts}
    end_ids = {e.start_event_index for e in ends}
    assert start_ids == end_ids
    assert len(start_ids) == len(starts), "indices must be unique"


def test_transcribe_returns_iterator(transcription_model, silence):
    events = _collect(transcription_model, silence)
    assert isinstance(events, list)
    for ev in events:
        assert isinstance(ev, (NoteStartEvent, NoteEndEvent, ProgressEvent))


def test_transcribe_start_end_pair_invariant(transcription_model, noise):
    events = _collect(transcription_model, noise)
    _assert_start_end_invariants(events)


def test_transcribe_note_fields_in_range(transcription_model, noise):
    events = _collect(transcription_model, noise)
    for ev in events:
        if isinstance(ev, ProgressEvent):
            continue
        if isinstance(ev, NoteStartEvent):
            assert 0 <= ev.pitch <= 127
            assert ev.start_time >= 0.0
            assert ev.instrument
        else:
            assert ev.end_time >= ev.start_event.start_time


# ---------------------------------------------------------------------------
# transcribe_and_postprocess
# ---------------------------------------------------------------------------


def test_transcribe_and_postprocess_returns_bytes(transcription_model, silence):
    midi_bytes, _ = transcription_model.transcribe_and_postprocess(
        (silence, _SAMPLE_RATE)
    )
    assert isinstance(midi_bytes, bytes)
    assert len(midi_bytes) > 0


def test_transcribe_and_postprocess_is_valid(transcription_model, silence):
    from mido import MidiFile

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "out.mid"
        out.write_bytes(
            transcription_model.transcribe_and_postprocess((silence, _SAMPLE_RATE))[0]
        )
        midi = MidiFile(str(out))
        assert len(midi.tracks) > 0


def test_cli_json_output(transcription_model, silence, monkeypatch):
    """End-to-end: `muscriptor transcribe --format json` writes a valid event list."""
    import json
    import wave

    from typer.testing import CliRunner

    from muscriptor import main as cli_main

    monkeypatch.setattr(
        cli_main.TranscriptionModel,
        "load_model",
        classmethod(lambda cls, **kw: transcription_model),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "in.wav"
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(_SAMPLE_RATE)
            w.writeframes(b"\x00\x00" * (_SAMPLE_RATE * _DURATION_SEC))
        out_path = Path(tmpdir) / "out.json"

        result = CliRunner().invoke(
            cli_main.app,
            ["transcribe", str(wav_path), "-o", str(out_path), "--format", "json"],
        )
        assert result.exit_code == 0, result.output

        assert out_path.exists()
        events = json.loads(out_path.read_text())
        assert isinstance(events, list)
        seen_starts: dict[int, dict] = {}
        for e in events:
            assert e["type"] in {"start", "end"}
            if e["type"] == "start":
                assert set(e.keys()) == {
                    "type",
                    "pitch",
                    "start_time",
                    "index",
                    "instrument",
                }
                assert 0 <= e["pitch"] <= 127
                assert e["index"] not in seen_starts
                seen_starts[e["index"]] = e
            else:
                assert set(e.keys()) == {"type", "end_time", "start_event_index"}
                start = seen_starts.pop(e["start_event_index"])
                assert e["end_time"] >= start["start_time"]
        assert seen_starts == {}, "every start should have an end"


# ---------------------------------------------------------------------------
# transcribe — sampling vs greedy
# ---------------------------------------------------------------------------


def test_transcribe_greedy(transcription_model, silence):
    events = list(
        transcription_model.transcribe((silence, _SAMPLE_RATE), use_sampling=False)
    )
    assert isinstance(events, list)


def test_transcribe_sampling(transcription_model, noise):
    events = list(
        transcription_model.transcribe(
            (noise, _SAMPLE_RATE), use_sampling=True, temperature=1.0
        )
    )
    assert isinstance(events, list)


def test_transcribe_cfg_coef_zero(transcription_model, silence):
    # cfg_coef=0 means no guidance; should still run and yield (possibly empty) events
    events = list(transcription_model.transcribe((silence, _SAMPLE_RATE), cfg_coef=0.0))
    assert isinstance(events, list)


# ---------------------------------------------------------------------------
# Real audio: filling_the_void.wav (first 10 seconds)
# ---------------------------------------------------------------------------


def test_song_clip_shape(song_clip):
    assert song_clip.dim() == 2
    assert song_clip.shape[0] == 1
    assert song_clip.shape[1] == _SAMPLE_RATE * _SONG_CLIP_SEC


def test_transcribe_song_returns_events(transcription_model, song_clip):
    events = list(transcription_model.transcribe((song_clip, _SAMPLE_RATE)))
    assert isinstance(events, list)
    assert len(events) > 0, "expected at least some events from a real song"
    _assert_start_end_invariants(events)


def test_transcribe_emits_progress_anchors(transcription_model, song_clip):
    events = list(transcription_model.transcribe((song_clip, _SAMPLE_RATE)))
    progress = [e for e in events if isinstance(e, ProgressEvent)]
    assert progress, "expected at least one ProgressEvent"
    total = progress[0].total
    assert total >= 1
    assert all(p.total == total for p in progress)
    # An up-front 0 anchor, then one completion per chunk, ending at total.
    assert [p.completed for p in progress] == list(range(total + 1))


def test_transcribe_song_note_validity(transcription_model, song_clip):
    events = list(transcription_model.transcribe((song_clip, _SAMPLE_RATE)))
    for ev in events:
        if isinstance(ev, NoteStartEvent):
            assert 0 <= ev.pitch <= 127
            assert ev.start_time >= 0.0
        elif isinstance(ev, NoteEndEvent):
            assert ev.end_time >= ev.start_event.start_time


def test_transcribe_song_to_midi(transcription_model, song_clip):
    from mido import MidiFile

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "filling_the_void_10s.mid"
        out.write_bytes(
            transcription_model.transcribe_and_postprocess((song_clip, _SAMPLE_RATE))
        )
        assert out.exists()
        midi = MidiFile(str(out))
        assert len(midi.tracks) > 0


def test_transcribe_song_from_file_path(transcription_model):
    """Transcribe directly from the file path (exercises load_audio)."""
    if not SONG_PATH.exists():
        pytest.skip(f"Song not found at {SONG_PATH}")
    events = list(transcription_model.transcribe(SONG_PATH))
    assert isinstance(events, list)
    assert len(events) > 0
