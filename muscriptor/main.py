"""CLI for muscriptor: audio → MIDI transcription."""

import dataclasses
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent
from muscriptor.tokenizer.mt3 import (
    MT3_FULL_PLUS_GROUP_NAMES,
    resolve_instrument_names,
)
from muscriptor.transcription_model import TranscriptionModel
from muscriptor.utils.download import ModelDownloadError

app = typer.Typer(add_completion=False, help="muscriptor — audio-to-MIDI transcription")


def _load_model(model_path: str | None, device: str | None) -> TranscriptionModel:
    """load_model with CLI-friendly failure: known download problems (missing
    HuggingFace authentication, …) print a plain message instead of a traceback."""
    try:
        return TranscriptionModel.load_model(weights_path=model_path, device=device)
    except ModelDownloadError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


class OutputFormat(str, Enum):
    midi = "midi"
    json = "json"
    jsonl = "jsonl"


def _event_to_dict(ev: NoteStartEvent | NoteEndEvent) -> dict:
    if isinstance(ev, NoteStartEvent):
        return {"type": "start", **dataclasses.asdict(ev)}
    return {
        "type": "end",
        "end_time": ev.end_time,
        "start_event_index": ev.start_event_index,
    }


@app.command()
def transcribe(
    audio_file: Annotated[
        Path, typer.Argument(help="Input audio file (wav, mp3, flac, …)")
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help=(
                "Output file path. Use '-' to write to stdout (all progress / "
                "timing info is sent to stderr in that case). "
                "Default: <audio_file>.<ext> where ext matches --format."
            ),
        ),
    ] = None,
    format: Annotated[
        OutputFormat,
        typer.Option(
            "--format",
            "-f",
            help=(
                "Output format: midi (default), json (single array of events), "
                "or jsonl (one event per line, streamed as transcription progresses)"
            ),
            case_sensitive=False,
        ),
    ] = OutputFormat.midi,
    notes: Annotated[
        bool, typer.Option("--notes", help="Print decoded events to stdout")
    ] = False,
    sampling: Annotated[
        bool,
        typer.Option(
            "--sampling", help="Use temperature sampling instead of greedy decoding"
        ),
    ] = False,
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature", "-t", help="Sampling temperature (only with --sampling)"
        ),
    ] = 1.0,
    cfg_coef: Annotated[
        float, typer.Option("--cfg-coef", help="Classifier-free guidance coefficient")
    ] = 1.0,  # todo: make it dynamic
    model_path: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help=(
                "Model size ('small', 'medium', 'large'; default: medium), "
                "a local safetensors path, or an hf:// / http(s):// URL"
            ),
        ),
    ] = None,
    device: Annotated[
        str,
        typer.Option(
            "--device", "-d", help="Device: 'auto', 'cpu', 'cuda', 'cuda:0', …"
        ),
    ] = "auto",
    batch_size: Annotated[
        int | None,
        typer.Option(
            "--batch-size",
            "-b",
            help="Batch size for generation (default: 1 on CPU, 4 on GPU)",
        ),
    ] = None,
    strict_eos: Annotated[
        bool,
        typer.Option(
            "--strict-eos",
            help="Raise an error if a chunk fails to emit EOS within the generation budget (default: downgrade to a warning)",
        ),
    ] = False,
    beam_size: Annotated[
        int,
        typer.Option(
            "--beam-size",
            help="Beam search width (1 = greedy/sampling, ≥2 enables beam search)",
        ),
    ] = 1,
    auralize: Annotated[
        Path | None,
        typer.Option(
            "--auralize",
            help=(
                "Write a stereo auralization (L=original audio, R=MIDI synthesis) to "
                "this path. Requires fluidsynth on PATH. Extension determines format: "
                ".wav (default) or .mp3. Only valid with --format midi."
            ),
        ),
    ] = None,
    soundfont: Annotated[
        Path | None,
        typer.Option(
            "--soundfont",
            help=(
                "Path to a .sf2 SoundFont for auralization. Defaults to "
                "MuseScore_General.sf2, downloaded once and cached locally."
            ),
        ),
    ] = None,
    instruments: Annotated[
        str | None,
        typer.Option(
            "--instruments",
            help=(
                "Comma-separated list of expected instrument group names. "
                "Case-insensitive; unambiguous abbreviations are accepted "
                "(e.g. 'timp,cello,dist'). Run 'muscriptor list-instruments' "
                "to see all available names."
            ),
        ),
    ] = None,
    strict_instruments: Annotated[
        bool,
        typer.Option(
            "--strict-instruments",
            help=(
                "Forbid instruments not listed in --instruments from being "
                "decoded at all, instead of only conditioning the model on "
                "the list (requires --instruments)."
            ),
        ),
    ] = False,
    tempo_bpm: Annotated[
        float | None,
        typer.Option(
            "--tempo-bpm",
            min=10.0,
            max=999.0,
            help=(
                "Tempo (in BPM) stamped into the output MIDI file "
                "(default: 120). Note timing stays wall-clock accurate at "
                "any value; set it to the track's real BPM so beats land on "
                "the grid when importing into a DAW. Only valid with "
                "--format midi."
            ),
        ),
    ] = None,
) -> None:
    """Transcribe an audio file to MIDI."""
    instrument_names: list[str] | None = None
    if instruments is not None:
        tokens = [n for n in instruments.split(",") if n.strip()]
        try:
            instrument_names = resolve_instrument_names(tokens)
        except ValueError as e:
            typer.echo(
                f"Error: {e}. "
                "Run 'muscriptor list-instruments' to see available names.",
                err=True,
            )
            raise typer.Exit(1)
        typer.echo(f"Instruments: {', '.join(instrument_names)}", err=True)

    if strict_instruments and not instrument_names:
        typer.echo(
            "Error: --strict-instruments requires --instruments", err=True
        )
        raise typer.Exit(1)

    if not audio_file.exists():
        typer.echo(f"Error: file not found: {audio_file}", err=True)
        raise typer.Exit(1)

    is_stdout = output is not None and str(output) == "-"

    if output is None:
        suffix = {
            OutputFormat.midi: ".mid",
            OutputFormat.json: ".json",
            OutputFormat.jsonl: ".jsonl",
        }[format]
        output = audio_file.with_suffix(suffix)

    _device = None if device == "auto" else device

    # All chatty progress/timing info goes to stderr — stdout is reserved for
    # the actual output when `-o -` is used.
    typer.echo("Loading model…", err=True)
    model = _load_model(model_path, _device)
    import torch

    model._model = model._model.to(torch.float32)

    typer.echo(f"Transcribing {audio_file} …", err=True)

    if auralize is not None and format != OutputFormat.midi:
        typer.echo("Error: --auralize requires --format midi", err=True)
        raise typer.Exit(1)

    if tempo_bpm is not None and format != OutputFormat.midi:
        typer.echo("Error: --tempo-bpm requires --format midi", err=True)
        raise typer.Exit(1)

    kwargs = dict(
        audio=audio_file,
        use_sampling=sampling,
        temperature=temperature,
        cfg_coef=cfg_coef,
        instruments=instrument_names,
        strict_instruments=strict_instruments,
        batch_size=batch_size,
        no_eos_is_ok=not strict_eos,
        beam_size=beam_size,
    )

    if format == OutputFormat.midi:
        if tempo_bpm is not None:
            kwargs["tempo_bpm"] = tempo_bpm
        midi_bytes = model.transcribe_to_midi(**kwargs)
        if is_stdout:
            sys.stdout.buffer.write(midi_bytes)
            sys.stdout.buffer.flush()
        else:
            output.write_bytes(midi_bytes)
            typer.echo(f"Saved MIDI to {output}", err=True)
        if notes:
            typer.echo(
                "Re-run with --format json to inspect the event stream.", err=True
            )
        if auralize is not None and not is_stdout:
            from muscriptor.utils.auralization import auralize as do_auralize

            typer.echo(f"Auralizing → {auralize} …", err=True)
            do_auralize(
                midi_path=output,
                original_audio_path=audio_file,
                output_path=auralize,
                soundfont_path=soundfont,
            )
            typer.echo(f"Saved auralization to {auralize}", err=True)
    elif format == OutputFormat.jsonl:
        # Stream one JSON object per line, flushing after each event so the
        # file (or stdout pipe) can be consumed live.
        if is_stdout:
            sink = sys.stdout
            close_after = False
        else:
            sink = output.open("w")
            close_after = True
        try:
            for e in model.transcribe(**kwargs):
                if isinstance(e, ProgressEvent):
                    continue
                sink.write(json.dumps(_event_to_dict(e)) + "\n")
                sink.flush()
                if notes:
                    typer.echo(str(e), err=True)
        finally:
            if close_after:
                sink.close()
        if not is_stdout:
            typer.echo(f"Saved JSONL to {output}", err=True)
    else:  # json
        events = [
            e
            for e in model.transcribe(**kwargs)
            if not isinstance(e, ProgressEvent)
        ]
        payload = json.dumps([_event_to_dict(e) for e in events], indent=2)
        if is_stdout:
            sys.stdout.write(payload + "\n")
            sys.stdout.flush()
        else:
            output.write_text(payload)
            typer.echo(f"Saved JSON to {output}", err=True)
        if notes:
            for e in events:
                typer.echo(str(e), err=True)


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host", help="Bind address")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port to listen on")] = 8222,
    model_path: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help=(
                "Model size ('small', 'medium', 'large'; default: medium), "
                "a local safetensors path, or an hf:// / http(s):// URL"
            ),
        ),
    ] = None,
    device: Annotated[
        str,
        typer.Option(
            "--device", "-d", help="Device: 'auto', 'cpu', 'cuda', 'cuda:0', …"
        ),
    ] = "auto",
):
    """Run the HTTP transcription server (POST /transcribe → SSE event stream)."""
    import uvicorn

    from muscriptor.server import create_app

    _device = None if device == "auto" else device
    typer.echo("Loading model…")
    model = _load_model(model_path, _device)
    web_dir = Path(__file__).resolve().parent / "web_dist"
    fastapi_app = create_app(model, web_dir=web_dir if web_dir.is_dir() else None)
    uvicorn.run(fastapi_app, host=host, port=port)


@app.command()
def list_instruments():
    """List the instrument group names accepted by --instruments."""
    for name in MT3_FULL_PLUS_GROUP_NAMES:
        typer.echo(name)


def main():
    app()


if __name__ == "__main__":
    main()
