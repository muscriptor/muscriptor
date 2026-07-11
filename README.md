<p align="center">
  <img src="web/logo_muscriptor_final.png" alt="MuScriptor logo" width="300">
</p>

# MuScriptor

MuScriptor is a multi-instrument music transcription model developed by [Kyutai](https://kyutai.org) and [Mirelo](https://www.mirelo.ai).
MuScriptor is the first music transcription model that has been trained on a large scale dataset of 170k songs from classical music to heavy metal.

[Online Demo](https://muscriptor.kyutai.org) | [Paper](https://arxiv.org/abs/2607.08168v1) | [HuggingFace](https://huggingface.co/MuScriptor)

<!-- TODO: record the demo GIF (web UI piano roll), save it as assets/demo.gif,
     then uncomment:
<p align="center">
  <img src="assets/demo.gif" alt="MuScriptor web UI: live piano roll while transcribing" width="700">
</p>
-->

## HuggingFace login (required)

The model weights are hosted on [HuggingFace](https://huggingface.co/MuScriptor)
and gated behind their CC BY-NC 4.0 license, so downloading them — including
on the first `uvx muscriptor` run — requires a (free) HuggingFace account:

1. Accept the model license on the model page, e.g.
   [muscriptor-medium](https://huggingface.co/MuScriptor/muscriptor-medium)
   (access is granted automatically).
2. Authenticate on your machine:

   ```bash
   uvx hf auth login
   ```

   or set a token (create one at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)):

   ```bash
   export HF_TOKEN=hf_...
   ```

The weights are then downloaded on first use and cached locally.

## Try it locally

You can try it locally with the web UI with:

```bash
uvx muscriptor serve
```

or with the CLI:

```bash
uvx muscriptor transcribe
```  

> **Intel Macs:** PyTorch stopped shipping Intel-mac (x86_64) wheels after
> torch 2.2.2, which supports Python ≤ 3.12, so you must tell uvx to use
> Python 3.12:
>
> ```bash
> uvx --python 3.12 muscriptor serve
> ```
>
> If you install with pip/uv instead, use Python 3.10–3.12.


## Installation

with uv (recommended):

```bash
uv add muscriptor
```

```bash
pip install muscriptor
```

## Models

Three variants are published under the [MuScriptor](https://huggingface.co/MuScriptor)
HuggingFace organization. Everywhere a model is selected (`load_model()`, the
CLI's `--model`, `serve --model`) you can pass the bare size keyword and the
weights are downloaded and cached automatically. The architecture is a transformer decoder only. Here are the detailed model sizes:

| Variant | Parameters | Layers | Dim | HuggingFace repo |
|---|---|---|---|---|
| `small` | 103M | 14 | 768 | [muscriptor-small](https://huggingface.co/MuScriptor/muscriptor-small) |
| `medium` (default) | 307M | 24 | 1024 | [muscriptor-medium](https://huggingface.co/MuScriptor/muscriptor-medium) |
| `large` | 1.4B | 48 | 1536 | [muscriptor-large](https://huggingface.co/MuScriptor/muscriptor-large) |

`small` is the practical choice on CPU-only machines, `medium` is the default
speed/accuracy trade-off, and `large` is the most accurate but really wants a
GPU. 
## Usage

```python
from pathlib import Path
from muscriptor import TranscriptionModel

# Downloads the default "medium" variant from HuggingFace (cached under
# ~/.cache/muscriptor/). Also accepts "small"/"large", a local safetensors
# path, or an hf:// / http(s):// URL.
model = TranscriptionModel.load_model()

# Stream events as they're transcribed. Optionally tell the model which
# instruments to expect — run `muscriptor list-instruments` for the names.
for event in model.transcribe("audio.wav", instruments=["acoustic_piano", "drums"]):
    print(event)

# Or get a MIDI file directly
midi_bytes = model.transcribe_to_midi("audio.wav")
Path("out.mid").write_bytes(midi_bytes)
```

```python
from dataclasses import dataclass
from typing import Generator


@dataclass
class NoteStartEvent:
    pitch: int
    # Start of the note in seconds, from the beginning of the audio.
    start_time: float
    # A unique index for this note, used to match the corresponding
    # NoteEndEvent.
    index: int
    instrument: str


@dataclass
class NoteEndEvent:
    end_time: float
    # The NoteStartEvent this end matches. Convenient for consumers —
    # when serializing (e.g. to JSON) drop this field and rely on
    # `start_event_index` to refer back to the start by id.
    start_event: NoteStartEvent

    @property
    def start_event_index(self) -> int:
        return self.start_event.index


@dataclass
class ProgressEvent:
    # A coarse progress anchor woven into the event stream: `completed` of
    # `total` 5-second chunks have been transcribed. One is emitted up front
    # with completed == 0 (so consumers learn `total`), then one per finished
    # chunk. Advisory only — consumers that just build notes can ignore them.
    completed: int
    total: int


class TranscriptionModel:
    ...
    def transcribe(
            self,
            audio: str | Path | tuple[torch.Tensor, int],
            use_sampling: bool = False,
            temperature: float = 1.0,
            cfg_coef: float = 1.0,
            instruments: list[str] | None = None,
            batch_size: int | None = None,
            no_eos_is_ok: bool = True,
            beam_size: int = 1,
        ) -> Generator[NoteStartEvent | NoteEndEvent | ProgressEvent, None, None]:
        """Transcribe audio into a stream of note events.

        Args:
            audio: Path to an audio file, or a tuple `(tensor, sample_rate)`
                with a float tensor of shape [T] or [1, T] at `sample_rate`
                Hz. The tuple form is useful when the audio is already
                loaded in memory.
            use_sampling: Use temperature sampling instead of greedy decoding.
            temperature: Sampling temperature (only used when use_sampling=True).
            cfg_coef: Classifier-free guidance coefficient. Keep to 1 for the released models (they are post-RL)
            instruments: Optional list of instrument group names to
                condition the model on (exact names, e.g.
                ["acoustic_piano", "drums"]). Run `muscriptor
                list-instruments` (or GET /instruments on the server)
                for the full list of valid names.
            batch_size: Number of 5-second chunks processed per forward
                pass. `None` (default) picks a value based on the device:
                1 on CPU, 4 on GPU. Use `batch_size=1` for the lowest
                streaming latency — larger batches process several chunks
                together, so events belonging to later chunks of a batch
                won't arrive until the whole batch finishes. Within a
                batch, events are always yielded in temporal order; all
                events from chunk N are emitted before any event from
                chunk N+1.
            no_eos_is_ok: If True, a chunk that doesn't emit EOS within
                the generation budget produces a warning instead of raising.
            beam_size: Beam search width. 1 (default) uses greedy decoding
                (or sampling, with use_sampling=True); >= 2 enables beam
                search, which is slower but can be more accurate.

        Returns:
            Generator of NoteStartEvent, NoteEndEvent and ProgressEvent
            objects. Every
            NoteStartEvent is guaranteed to be followed by exactly one
            matching NoteEndEvent later in the stream (with the same
            `index`). Drum hits appear as a NoteStartEvent immediately
            followed by its matching NoteEndEvent at the same start time
            plus a tiny duration. Note: this tokenizer does not preserve
            velocity (loudness) — only onset/offset timing, pitch, and
            instrument are recovered.
        """

    def transcribe_to_midi(
            self,
            audio: str | Path | tuple[torch.Tensor, int],
            use_sampling: bool = False,
            temperature: float = 1.0,
            cfg_coef: float = 1.0,
            instruments: list[str] | None = None,
            batch_size: int | None = None,
            no_eos_is_ok: bool = True,
            beam_size: int = 1,
        ) -> bytes:
        """Same as `transcribe`, but returns a MIDI file as bytes instead
        of a generator of events. Useful when you want to save the MIDI
        to disk or send it over a network without going through the
        event stream.
        """
```


## CLI

```bash
# Transcribe to MIDI (defaults to <audio_file>.mid next to the input)
muscriptor transcribe audio.wav -o out.mid

# Pick a model variant: small / medium / large (default: medium),
# a local safetensors path, or an hf:// / http(s):// URL
muscriptor transcribe audio.wav --model large

# Tell the model which instruments to expect (comma-separated names;
# run `muscriptor list-instruments` for the full list). Case-insensitive,
# and unambiguous abbreviations work: 'timp,dist' = timpani + distorted
# electric guitar
muscriptor transcribe audio.wav --instruments acoustic_piano,drums

# Get the event stream instead of MIDI: json (single array) or
# jsonl (one event per line, streamed while transcribing); -o - = stdout
muscriptor transcribe audio.wav --format jsonl -o -

# Decoding options: temperature sampling, or beam search (slower, can be
# more accurate)
muscriptor transcribe audio.wav --sampling -t 0.8
muscriptor transcribe audio.wav --beam-size 4

# Render a stereo check-mix of the result (left channel = original audio,
# right channel = synthesized MIDI; requires fluidsynth on PATH)
muscriptor transcribe audio.wav -o out.mid --auralize check.wav
```

See `muscriptor transcribe --help` for the full list of options.

## Web UI

A browser client is included under `web/`. The FastAPI server serves both
the UI and a POST `/transcribe` endpoint that streams `NoteStart`/`NoteEnd`
events back as Server-Sent Events. The UI accepts an audio file (WAV, or any
format soundfile/libsndfile can read — mp3, flac, ogg, m4a, …) via drag-and-drop,
renders a live piano roll while events arrive, auto-plays once enough notes
are available, and crossfades between the original WAV and the synthesized
MIDI playback.

### One-time setup

```bash
uv sync
cd web && pnpm install && pnpm run build && cd ..
```

`pnpm run build` is required once — it outputs to `muscriptor/web_dist/`,
which the FastAPI server auto-mounts if it exists (and which ships inside
the PyPI wheel, so `uvx muscriptor serve` works without a checkout).

The soundfonts are not bundled: the server fetches
`MuseScore_General.sf2` (215 MB, used by `/auralize`) and
`MuseScore_General.sf3` (38 MB, the compressed build the UI plays) from
[MuScriptor/assets](https://huggingface.co/MuScriptor/assets) on first use
and caches them locally (see `muscriptor/soundfonts.py`).

### Run

```bash
uv run muscriptor serve \
    --model medium \
    --device cuda \
    --host 0.0.0.0 \
    --port 8222
```

`--model` accepts a size keyword (`small`, `medium`, `large`) that downloads
the matching variant from HuggingFace (cached under `~/.cache/muscriptor/`),
a local safetensors path, or an `hf://` / `http(s)://` URL. It defaults to
`medium` when omitted.

Then open <http://127.0.0.1:8222/> (or the LAN address of the host) and drop
a WAV onto the page.

- Drop `--device cuda` if running CPU-only.
- `--host 0.0.0.0` makes it reachable on the LAN; the default `127.0.0.1`
  is local-only.
- Playback runs a full SoundFont synthesizer ([SpessaSynth](https://github.com/spessasus/spessasynth_lib))
  in the browser, fed with `MuseScore_General.sf3` — the same soundfont the
  `/auralize` endpoint uses, served by the app itself from `/soundfonts/`
  (cached server-side), no third-party CDN.

## License

The code in this repository is released under the [MIT license](LICENSE).

The model weights, published on
[HuggingFace](https://huggingface.co/MuScriptor), are released under the
[CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/)
(non-commercial use).

The MuseScore General SoundFont downloaded for playback / auralization is
distributed under its own (MIT) license.

## Citation

```bibtex
@misc{rouard2026muscriptoropenmodelmultiinstrument,
      title={MuScriptor: An Open Model for Multi-Instrument Music Transcription}, 
      author={Simon Rouard and Michael Krause and Axel Roebel and Carl-Johann Simon-Gabriel and Alexandre Défossez},
      year={2026},
      eprint={2607.08168},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2607.08168}, 
}
```
