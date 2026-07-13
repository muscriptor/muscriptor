"""FastAPI server exposing transcription as an SSE event stream.

POST /transcribe with an audio file (multipart/form-data field `file`; WAV,
or any format soundfile/libsndfile can read — mp3, flac, ogg, m4a, …) returns
`text/event-stream`. Each event's data is a JSON dict tagged by `type`:
`start` / `end` note events (same shape as `muscriptor.main._event_to_dict`),
`progress` chunk anchors (`{completed, total}`), and a final `midi` event
carrying the base64-encoded .mid file.
"""

import asyncio
import base64
import dataclasses
import io
import json
import os
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent
from muscriptor.soundfonts import SF3_URL
from muscriptor.tokenizer.mt3 import MT3_FULL_PLUS_GROUP_NAMES
from muscriptor.transcription_model import TranscriptionModel
from muscriptor.utils.audio import _read_non_wav_file, _read_wav_file
from muscriptor.utils.download import download_if_necessary


def _make_release_once(lock: threading.Lock):
    """Return a callable that releases `lock` at most once.

    Safe to call from multiple cleanup paths (generator finally + response
    background task), possibly from different threads, without risking a
    double-release RuntimeError.
    """
    guard = threading.Lock()
    released = False

    def release():
        nonlocal released
        with guard:
            if released:
                return
            released = True
        lock.release()

    return release


def event_to_dict(ev: NoteStartEvent | NoteEndEvent) -> dict:
    if isinstance(ev, NoteStartEvent):
        return {"type": "start", **dataclasses.asdict(ev)}
    return {
        "type": "end",
        "end_time": ev.end_time,
        "start_event_index": ev.start_event_index,
    }


def create_app(model: TranscriptionModel, web_dir: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="muscriptor")

    transcribe_lock = threading.Lock()
    lock_timeout_s = 60.0
    # Cancel event of the run currently holding the lock (or the last one to
    # have held it). A new /transcribe request sets it so an in-flight run
    # stops at its next event boundary instead of transcribing to completion
    # for a client that has moved on. This must not rely on TCP disconnects:
    # aborts don't always reach us (e.g. port forwards / proxies that keep the
    # upstream connection open after the browser aborts).
    cancel_guard = threading.Lock()
    current_cancel: threading.Event | None = None

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/instruments")
    async def list_instruments():
        return {"instruments": list(MT3_FULL_PLUS_GROUP_NAMES.keys())}

    @app.get("/soundfonts/MuseScore_General.sf3")
    async def soundfont() -> FileResponse:
        """Compressed soundfont for the web UI's in-browser synthesizer.

        Fetched from SF3_URL on first request (in a worker thread, so the
        event loop keeps serving) and cached locally.
        """
        path = await asyncio.to_thread(download_if_necessary, SF3_URL)
        return FileResponse(path, media_type="application/octet-stream")

    @app.post("/transcribe")
    async def transcribe(
        file: Annotated[UploadFile, File()],
        instruments: Annotated[list[str], Form(default_factory=list)],
    ) -> StreamingResponse:
        data = await file.read()
        # PCM WAV goes through the stdlib reader (keeps WAV decoding byte-for-byte
        # identical to the CLI); anything that isn't a readable WAV (mp3, flac,
        # ogg, m4a, …) falls back to soundfile/libsndfile. A genuinely
        # undecodable upload (corrupt/truncated file, or a format libsndfile
        # can't read) is the client's fault, so report it as a 400 rather than
        # letting it surface as a 500.
        try:
            wav, sr = _read_wav_file(io.BytesIO(data))
        except (wave.Error, EOFError):
            try:
                wav, sr = _read_non_wav_file(io.BytesIO(data))
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"could not decode audio file '{file.filename}': {e}",
                ) from e

        unknown = [n for n in instruments if n not in MT3_FULL_PLUS_GROUP_NAMES]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"unknown instrument name(s): {', '.join(unknown)}",
            )

        # Preempt whoever holds the lock, then wait for it. The short acquire
        # timeout re-signals each second so that even a run that started while
        # we were already waiting (another preempting request that beat us to
        # the lock) gets cancelled too — the newest request always wins.
        nonlocal current_cancel
        deadline = time.monotonic() + lock_timeout_s
        while True:
            with cancel_guard:
                if current_cancel is not None:
                    current_cancel.set()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HTTPException(
                    status_code=503,
                    detail="server busy: another transcription is in progress",
                )
            acquired = await asyncio.to_thread(
                transcribe_lock.acquire, True, min(1.0, remaining)
            )
            if acquired:
                break
        cancel = threading.Event()
        with cancel_guard:
            current_cancel = cancel

        # Release exactly once, from whichever path runs first. The generator's
        # finally covers normal completion, errors and mid-stream disconnects;
        # the StreamingResponse background task covers the case where the client
        # disconnects *before* the generator is ever iterated (so its finally
        # would never run) — which otherwise leaks the lock forever.
        release_lock = _make_release_once(transcribe_lock)

        def gen():
            try:
                events: list[NoteStartEvent | NoteEndEvent] = []
                # batch_size=1 so each chunk's notes stream out as soon as it is
                # generated, instead of waiting for a whole batch of chunks.
                # no_eos_is_ok=True so one runaway chunk that never emits EOS only
                # warns (and keeps its notes) instead of aborting the whole stream.
                for ev in model.transcribe(
                    (wav, sr),
                    instruments=instruments or None,
                    batch_size=1,
                    no_eos_is_ok=True,
                ):
                    # A newer request preempted this run — stop generating
                    # (closing the model.transcribe generator) and release the
                    # lock via the finally, at most one chunk after the signal.
                    if cancel.is_set():
                        return
                    if isinstance(ev, ProgressEvent):
                        # Coarse chunk-completion anchor — forward it but keep it
                        # out of the note list the MIDI file is built from.
                        payload = json.dumps(
                            {
                                "type": "progress",
                                "completed": ev.completed,
                                "total": ev.total,
                            }
                        )
                        yield f"data: {payload}\n\n"
                        continue
                    events.append(ev)
                    payload = json.dumps(event_to_dict(ev))
                    yield f"data: {payload}\n\n"
                # All notes streamed — build the MIDI file in memory (reusing the
                # exact `muscriptor transcribe` logic) and send it as a final event
                # with the bytes base64-encoded.
                if cancel.is_set():
                    return
                midi_bytes = model.events_to_midi_bytes(iter(events))
                midi_b64 = base64.b64encode(midi_bytes).decode("ascii")
                payload = json.dumps({"type": "midi", "data": midi_b64})
                yield f"data: {payload}\n\n"
            finally:
                release_lock()

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            background=BackgroundTask(release_lock),
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/auralize")
    async def auralize(
        midi: Annotated[UploadFile, File()],
        audio: Annotated[UploadFile | None, File()] = None,
        mode: Annotated[str, Form()] = "mix",
    ):
        """Render a transcription as WAV.

        mode="mix": stereo, original audio (L) + FluidSynth synthesis (R);
        requires the `audio` upload. mode="synth": mono, just the synthesis.
        """
        from muscriptor.utils.auralization import auralize as do_auralize
        from muscriptor.utils.auralization import synthesize

        if mode not in ("mix", "synth"):
            raise HTTPException(status_code=400, detail=f"unknown mode: {mode!r}")
        if mode == "mix" and audio is None:
            raise HTTPException(
                status_code=400, detail="mode='mix' requires an audio file"
            )

        midi_data = await midi.read()
        tmp_paths: list[str] = []

        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp_midi:
            tmp_midi.write(midi_data)
            midi_tmp = tmp_midi.name
            tmp_paths.append(midi_tmp)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            out_tmp = tmp_out.name
            tmp_paths.append(out_tmp)

        try:
            if mode == "synth":
                synthesize(midi_path=midi_tmp, output_path=out_tmp)
            else:
                audio_data = await audio.read()
                suffix = Path(audio.filename or "audio.wav").suffix.lower() or ".wav"
                with tempfile.NamedTemporaryFile(
                    suffix=suffix, delete=False
                ) as tmp_audio:
                    tmp_audio.write(audio_data)
                    tmp_paths.append(tmp_audio.name)
                do_auralize(
                    midi_path=midi_tmp,
                    original_audio_path=tmp_audio.name,
                    output_path=out_tmp,
                )
            with open(out_tmp, "rb") as f:
                wav_bytes = f.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            for p in tmp_paths:
                if os.path.exists(p):
                    os.unlink(p)

        return Response(content=wav_bytes, media_type="audio/wav")

    if web_dir is not None:
        web_path = Path(web_dir)
        if web_path.is_dir():
            app.mount("/", StaticFiles(directory=web_path, html=True), name="web")

    return app
