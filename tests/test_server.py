"""Hermetic tests for the FastAPI transcription server.

Uses a fake transcriber so no weights / audio decoding is required.
"""

import base64
import json
import threading
import time
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
    model.transcribe_to_midi.return_value = midi
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


def test_transcribe_passes_instruments(tmp_path):
    model = make_model()
    client = TestClient(create_app(model))
    resp = client.post(
        "/transcribe",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
        data={"instruments": ["violin", "drums"]},
    )
    assert resp.status_code == 200
    assert model.transcribe.call_args.kwargs["instruments"] == ["violin", "drums"]


def test_transcribe_midi_returns_bytes_with_headers(tmp_path):
    model = make_model()
    client = TestClient(create_app(model))
    resp = client.post(
        "/transcribe/midi",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/midi"
    assert resp.headers["content-disposition"] == 'attachment; filename="result.mid"'
    assert resp.content == FAKE_MIDI


def test_transcribe_midi_passes_tensor_and_instruments(tmp_path):
    import torch

    model = make_model()
    client = TestClient(create_app(model))
    resp = client.post(
        "/transcribe/midi",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
        data={"instruments": ["violin", "drums"]},
    )
    assert resp.status_code == 200
    (audio,) = model.transcribe_to_midi.call_args.args
    assert model.transcribe_to_midi.call_args.kwargs["instruments"] == [
        "violin",
        "drums",
    ]
    tensor, sr = audio
    assert isinstance(tensor, torch.Tensor)
    assert sr == 16000
    assert tensor.shape[-1] == 1600


def test_transcribe_midi_rejects_invalid_wav():
    client = TestClient(create_app(make_model()), raise_server_exceptions=False)
    resp = client.post(
        "/transcribe/midi",
        files={"file": ("garbage.wav", b"not a wav at all", "audio/wav")},
    )
    assert resp.status_code == 400


def test_transcribe_midi_rejects_unknown_instrument(tmp_path):
    client = TestClient(create_app(make_model()), raise_server_exceptions=False)
    resp = client.post(
        "/transcribe/midi",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
        data={"instruments": ["not_a_real_instrument"]},
    )
    assert resp.status_code == 400


def test_transcribe_midi_rejects_audio_over_duration_limit(tmp_path, monkeypatch):
    """Audio longer than the 15-minute cap is rejected with 413, before the
    model is ever touched."""
    import muscriptor.server as server_module

    monkeypatch.setattr(server_module, "_MAX_TRANSCRIBE_MIDI_DURATION_S", 0.05)
    model = make_model()
    client = TestClient(create_app(model), raise_server_exceptions=False)
    resp = client.post(
        "/transcribe/midi",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
    )
    assert resp.status_code == 413
    model.transcribe_to_midi.assert_not_called()


def _blocking_transcribe_model(first_reached: threading.Event, gate: threading.Event):
    """A mock whose first `transcribe()` call streams one note, then blocks on
    `gate` (signalling via `first_reached` once it's blocked) while still holding
    the lock; later calls stream both notes without blocking. Lets a test force
    two /transcribe requests to overlap deterministically."""
    s0 = NoteStartEvent(pitch=60, start_time=0.0, index=0, instrument="piano")
    e0 = NoteEndEvent(end_time=0.5, start_event=s0)
    call_lock = threading.Lock()
    calls = [0]

    def side_effect(*args, **kwargs):
        with call_lock:
            calls[0] += 1
            first = calls[0] == 1
        yield s0
        if first:
            first_reached.set()
            assert gate.wait(timeout=10), "gate never opened"
        yield e0

    model = create_autospec(TranscriptionModel, instance=True)
    model.transcribe.side_effect = side_effect
    model.events_to_midi_bytes.return_value = FAKE_MIDI
    return model, s0


def _post_transcribe(app, payload, client_id, out, name):
    # A fresh TestClient per thread (httpx.Client isn't for concurrent use); the
    # underlying app — and its lock — is shared, which is what we're exercising.
    client = TestClient(app)
    resp = client.post(
        "/transcribe",
        files={"file": ("silent.wav", payload, "audio/wav")},
        headers={"X-Client-Id": client_id},
    )
    out[name] = _parse_sse(resp.text)


def test_concurrent_different_clients_do_not_preempt(tmp_path):
    """The reported bug: a second window (different client id) must NOT stop the
    transcription already in progress — it waits for the lock instead."""
    first_reached = threading.Event()
    gate = threading.Event()
    model, s0 = _blocking_transcribe_model(first_reached, gate)
    app = create_app(model)
    payload = _wav_bytes(tmp_path)
    out: dict[str, list] = {}

    a = threading.Thread(target=_post_transcribe, args=(app, payload, "tab-A", out, "A"))
    a.start()
    assert first_reached.wait(timeout=5)  # A holds the lock, mid-stream

    b = threading.Thread(target=_post_transcribe, args=(app, payload, "tab-B", out, "B"))
    b.start()
    # Give B time to reach the lock; with broken scoping it would cancel A here.
    time.sleep(0.5)
    gate.set()
    a.join(timeout=10)
    b.join(timeout=10)

    # A ran to completion (ends with the assembled MIDI event) — not preempted.
    assert out["A"][0] == event_to_dict(s0)
    assert out["A"][-1] == {
        "type": "midi",
        "data": base64.b64encode(FAKE_MIDI).decode("ascii"),
    }
    # B, a different client, also completed — after waiting its turn.
    assert out["B"][-1]["type"] == "midi"
    assert model.transcribe.call_count == 2


def test_concurrent_same_client_preempts(tmp_path):
    """A resubmit from the SAME client id still preempts the in-flight run so a
    stale stream stops instead of finishing."""
    first_reached = threading.Event()
    gate = threading.Event()
    model, s0 = _blocking_transcribe_model(first_reached, gate)
    app = create_app(model)
    payload = _wav_bytes(tmp_path)
    out: dict[str, list] = {}

    a = threading.Thread(
        target=_post_transcribe, args=(app, payload, "same-tab", out, "A")
    )
    a.start()
    assert first_reached.wait(timeout=5)

    b = threading.Thread(
        target=_post_transcribe, args=(app, payload, "same-tab", out, "B")
    )
    b.start()
    # Let B reach the lock and signal A's cancel (same id → preempt) before A
    # resumes past the gate.
    time.sleep(0.5)
    gate.set()
    a.join(timeout=10)
    b.join(timeout=10)

    # A was preempted: it streamed its first note but never the trailing MIDI.
    assert out["A"] == [event_to_dict(s0)]
    # B (the resubmit) ran to completion.
    assert out["B"][-1]["type"] == "midi"
    assert model.transcribe.call_count == 2
