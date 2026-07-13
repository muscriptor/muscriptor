"""Hermetic tests for the FastAPI transcription server.

Uses a fake transcriber so no weights / audio decoding is required.
"""

import base64
import json
import wave
from pathlib import Path
from unittest.mock import create_autospec

from fastapi.testclient import TestClient

from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent
from muscriptor.server import create_app, event_to_dict
from muscriptor.transcription_model import TranscriptionModel


FAKE_MIDI = b"FAKE_MIDI_BYTES"


def make_model(events=(), midi=FAKE_MIDI):
    """A mock standing in for TranscriptionModel.

    Autospec'd against the real class so the mock fakes isinstance and keeps
    method signatures in sync with what the server calls — but no weights or
    audio decoding are loaded.
    """
    model = create_autospec(TranscriptionModel, instance=True)
    model.transcribe.return_value = list(events)
    model.events_to_midi_bytes.return_value = midi
    return model


def _wav_bytes(tmp_path: Path) -> bytes:
    """Write a tiny silent WAV so the upload payload is a real file."""
    p = tmp_path / "silent.wav"
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)
    return p.read_bytes()


def _flac_bytes(sample_rate: int = 22050, n_frames: int = 1600) -> bytes:
    """Encode a tiny silent mono FLAC in memory via soundfile (non-WAV path)."""
    import io

    import numpy as np
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.zeros(n_frames, dtype="float32"), sample_rate, format="FLAC")
    return buf.getvalue()


def _parse_sse(body: str) -> list[dict]:
    """Parse SSE `data: <json>` lines into a list of dicts."""
    out: list[dict] = []
    for chunk in body.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        assert chunk.startswith("data: "), f"unexpected SSE chunk: {chunk!r}"
        out.append(json.loads(chunk[len("data: ") :]))
    return out


def test_event_to_dict_start_and_end():
    start = NoteStartEvent(pitch=60, start_time=0.5, index=0, instrument="piano")
    end = NoteEndEvent(end_time=1.5, start_event=start)
    assert event_to_dict(start) == {
        "type": "start",
        "pitch": 60,
        "start_time": 0.5,
        "index": 0,
        "instrument": "piano",
    }
    assert event_to_dict(end) == {
        "type": "end",
        "end_time": 1.5,
        "start_event_index": 0,
    }


def test_transcribe_streams_sse_events(tmp_path):
    s0 = NoteStartEvent(pitch=60, start_time=0.0, index=0, instrument="piano")
    s1 = NoteStartEvent(pitch=64, start_time=0.1, index=1, instrument="guitar")
    events = [
        s0,
        NoteEndEvent(end_time=0.5, start_event=s0),
        s1,
        NoteEndEvent(end_time=0.6, start_event=s1),
    ]
    model = make_model(events)
    client = TestClient(create_app(model))

    payload = _wav_bytes(tmp_path)
    resp = client.post(
        "/transcribe",
        files={"file": ("silent.wav", payload, "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    parsed = _parse_sse(resp.text)
    # Note events, then a trailing base64-encoded MIDI event.
    assert parsed[:-1] == [event_to_dict(e) for e in events]
    assert parsed[-1] == {
        "type": "midi",
        "data": base64.b64encode(FAKE_MIDI).decode("ascii"),
    }
    assert model.transcribe.call_count == 1


def test_transcribe_forwards_progress(tmp_path):
    s0 = NoteStartEvent(pitch=60, start_time=0.0, index=0, instrument="piano")
    events = [
        ProgressEvent(completed=0, total=2),
        s0,
        NoteEndEvent(end_time=0.5, start_event=s0),
        ProgressEvent(completed=1, total=2),
        ProgressEvent(completed=2, total=2),
    ]
    model = make_model(events)
    client = TestClient(create_app(model))

    resp = client.post(
        "/transcribe",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
    )
    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)

    # Progress events surface as their own SSE type, interleaved with notes.
    assert {"type": "progress", "completed": 0, "total": 2} in parsed
    assert {"type": "progress", "completed": 2, "total": 2} in parsed
    # ...but are kept out of the note list the MIDI file is built from.
    (built,) = model.events_to_midi_bytes.call_args.args
    assert all(not isinstance(e, ProgressEvent) for e in built)


def test_transcribe_empty_stream(tmp_path):
    model = make_model()
    client = TestClient(create_app(model))
    resp = client.post(
        "/transcribe",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
    )
    assert resp.status_code == 200
    # No notes, but the trailing MIDI event is still emitted.
    assert _parse_sse(resp.text) == [
        {"type": "midi", "data": base64.b64encode(FAKE_MIDI).decode("ascii")}
    ]


def test_transcribe_missing_file():
    client = TestClient(create_app(make_model()))
    resp = client.post("/transcribe")
    assert resp.status_code == 422


def test_transcribe_passes_tensor_not_path(tmp_path):
    """Server must hand the model an in-memory (tensor, sr) tuple — no disk."""
    import torch

    model = make_model()
    client = TestClient(create_app(model))
    resp = client.post(
        "/transcribe",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
    )
    assert resp.status_code == 200
    audio = model.transcribe.call_args.args[0]
    assert isinstance(audio, tuple)
    tensor, sr = audio
    assert isinstance(tensor, torch.Tensor)
    assert sr == 16000
    assert tensor.shape[-1] == 1600  # samples we wrote


def test_transcribe_rejects_invalid_wav():
    # An undecodable upload is the client's fault: the endpoint reports 400.
    client = TestClient(create_app(make_model()), raise_server_exceptions=False)
    resp = client.post(
        "/transcribe",
        files={"file": ("garbage.wav", b"not a wav at all", "audio/wav")},
    )
    assert resp.status_code == 400


def test_transcribe_accepts_non_wav_audio():
    """A non-WAV upload (FLAC) decodes via soundfile and reaches the model."""
    import torch

    model = make_model()
    client = TestClient(create_app(model))
    resp = client.post(
        "/transcribe",
        files={"file": ("clip.flac", _flac_bytes(sample_rate=22050), "audio/flac")},
    )
    assert resp.status_code == 200
    audio = model.transcribe.call_args.args[0]
    assert isinstance(audio, tuple)
    tensor, sr = audio
    assert isinstance(tensor, torch.Tensor)
    # Decoded by soundfile, not resampled by the server — sample rate preserved.
    assert sr == 22050
    assert tensor.shape[-1] == 1600


def test_transcribe_rejects_undecodable_file():
    """Bytes that are neither WAV nor anything libsndfile reads → 400."""
    client = TestClient(create_app(make_model()), raise_server_exceptions=False)
    resp = client.post(
        "/transcribe",
        files={"file": ("mystery.mp3", b"\x00\x01 not audio \x02\x03", "audio/mpeg")},
    )
    assert resp.status_code == 400


def test_transcribe_strict_without_instruments_is_rejected(tmp_path):
    """strict_instruments with an empty instrument list is a client error."""
    model = make_model()
    client = TestClient(create_app(model))
    resp = client.post(
        "/transcribe",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
        data={"strict_instruments": "true"},
    )
    assert resp.status_code == 400
    assert "strict_instruments" in resp.json()["detail"]
    model.transcribe.assert_not_called()


def test_transcribe_passes_strict_instruments(tmp_path):
    model = make_model()
    client = TestClient(create_app(model))
    resp = client.post(
        "/transcribe",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
        data={"instruments": ["violin", "drums"], "strict_instruments": "true"},
    )
    assert resp.status_code == 200
    kwargs = model.transcribe.call_args.kwargs
    assert kwargs["instruments"] == ["violin", "drums"]
    assert kwargs["strict_instruments"] is True


def test_transcribe_strict_defaults_to_false(tmp_path):
    model = make_model()
    client = TestClient(create_app(model))
    resp = client.post(
        "/transcribe",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
        data={"instruments": ["violin"]},
    )
    assert resp.status_code == 200
    assert model.transcribe.call_args.kwargs["strict_instruments"] is False


def test_transcribe_passes_tempo_bpm(tmp_path):
    model = make_model()
    client = TestClient(create_app(model))
    resp = client.post(
        "/transcribe",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
        data={"tempo_bpm": "87.5"},
    )
    assert resp.status_code == 200
    assert model.events_to_midi_bytes.call_args.kwargs["tempo_bpm"] == 87.5


def test_transcribe_tempo_bpm_defaults_to_120(tmp_path):
    model = make_model()
    client = TestClient(create_app(model))
    resp = client.post(
        "/transcribe",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
    )
    assert resp.status_code == 200
    assert model.events_to_midi_bytes.call_args.kwargs["tempo_bpm"] == 120.0


def test_transcribe_rejects_out_of_range_tempo(tmp_path):
    model = make_model()
    client = TestClient(create_app(model))
    for bad in ("0", "-3", "5000"):
        resp = client.post(
            "/transcribe",
            files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
            data={"tempo_bpm": bad},
        )
        assert resp.status_code == 400, bad
        assert "tempo_bpm" in resp.json()["detail"]
    model.transcribe.assert_not_called()


def test_transcribe_passes_sampling_params(tmp_path):
    model = make_model()
    client = TestClient(create_app(model))
    resp = client.post(
        "/transcribe",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
        data={"use_sampling": "true", "temperature": "0.8", "cfg_coef": "3.0"},
    )
    assert resp.status_code == 200
    kwargs = model.transcribe.call_args.kwargs
    assert kwargs["use_sampling"] is True
    assert kwargs["temperature"] == 0.8
    assert kwargs["cfg_coef"] == 3.0


def test_transcribe_sampling_defaults(tmp_path):
    model = make_model()
    client = TestClient(create_app(model))
    resp = client.post(
        "/transcribe",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
    )
    assert resp.status_code == 200
    kwargs = model.transcribe.call_args.kwargs
    assert kwargs["use_sampling"] is False
    assert kwargs["temperature"] == 1.0
    assert kwargs["cfg_coef"] == 1.0


def test_transcribe_rejects_out_of_range_temperature(tmp_path):
    model = make_model()
    client = TestClient(create_app(model))
    for bad in ("0", "-1", "100"):
        resp = client.post(
            "/transcribe",
            files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
            data={"temperature": bad},
        )
        assert resp.status_code == 400, bad
        assert "temperature" in resp.json()["detail"]
    model.transcribe.assert_not_called()


def test_transcribe_rejects_out_of_range_cfg_coef(tmp_path):
    model = make_model()
    client = TestClient(create_app(model))
    for bad in ("-1", "100"):
        resp = client.post(
            "/transcribe",
            files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
            data={"cfg_coef": bad},
        )
        assert resp.status_code == 400, bad
        assert "cfg_coef" in resp.json()["detail"]
    model.transcribe.assert_not_called()


def test_config_reports_max_beam_size():
    client = TestClient(create_app(make_model(), max_beam_size=4))
    assert client.get("/config").json() == {"max_beam_size": 4}
    # Default cap is 1 (beam search disabled).
    client = TestClient(create_app(make_model()))
    assert client.get("/config").json() == {"max_beam_size": 1}


def test_create_app_rejects_bad_max_beam_size():
    import pytest

    with pytest.raises(ValueError, match="max_beam_size"):
        create_app(make_model(), max_beam_size=0)


def test_transcribe_passes_beam_size(tmp_path):
    model = make_model()
    client = TestClient(create_app(model, max_beam_size=4))
    resp = client.post(
        "/transcribe",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
        data={"beam_size": "3"},
    )
    assert resp.status_code == 200
    assert model.transcribe.call_args.kwargs["beam_size"] == 3


def test_transcribe_beam_size_defaults_to_1(tmp_path):
    model = make_model()
    client = TestClient(create_app(model))
    resp = client.post(
        "/transcribe",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
    )
    assert resp.status_code == 200
    assert model.transcribe.call_args.kwargs["beam_size"] == 1


def test_transcribe_rejects_beam_size_over_cap(tmp_path):
    """beam_size beyond the server cap (including the default cap of 1) → 400."""
    for max_beam, bad in [(1, "2"), (4, "5"), (4, "0")]:
        model = make_model()
        client = TestClient(create_app(model, max_beam_size=max_beam))
        resp = client.post(
            "/transcribe",
            files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
            data={"beam_size": bad},
        )
        assert resp.status_code == 400, (max_beam, bad)
        assert "beam_size" in resp.json()["detail"]
        model.transcribe.assert_not_called()
