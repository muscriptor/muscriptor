<p align="center">
  <img src="web/logo_muscriptor_final.png" alt="MuScriptor logo" width="300">
</p>

# MuScriptor

MuScriptor is a multi-instrument music transcription model developed by [Kyutai](https://kyutai.org) and [Mirelo](https://www.mirelo.ai).
It turns a recording into MIDI and into sheet music.
It's the most accurate open-source transcription model.
You can use the model [here](https://muscriptor.kyutai.org) or self-host it using this repository.


[Use it](https://muscriptor.kyutai.org) | [Paper](https://arxiv.org/abs/2607.08168v1) | [HuggingFace](https://huggingface.co/MuScriptor)

<!-- TODO: record the demo GIF (web UI piano roll), save it as assets/demo.gif,
     then uncomment:
<p align="center">
  <img src="assets/demo.gif" alt="MuScriptor web UI: live piano roll while transcribing" width="700">
</p>
-->

## HuggingFace login (required)

To use MuScriptor locally, you first need to log into [HuggingFace](https://huggingface.co/MuScriptor)
and accept the CC BY-NC 4.0 license.

1. Accept the model license on the model page for the [small](https://huggingface.co/MuScriptor/muscriptor-small),
   [medium](https://huggingface.co/MuScriptor/muscriptor-medium) or [large](https://huggingface.co/MuScriptor/muscriptor-large) model
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

The weights are then automatically downloaded on first use and cached locally.

## Try it locally

After Hugging Face authentication, you can use MuScriptor with `uvx` without having to clone this repo.

Some platforms need an extra `uvx` flag, on every `uvx muscriptor` command:

| Platform | Command |
|---|---|
| Linux, macOS with Apple Silicon | `uvx muscriptor serve` |
| Windows (to use the GPU) | `uvx --torch-backend=cu128 muscriptor serve` |
| macOS with Intel | `uvx --python 3.12 muscriptor serve` |

On Windows the default PyTorch backend is `cpu`, so the GPU needs
`--torch-backend=cu128`. On Intel Macs, PyTorch stopped shipping x86_64 wheels
after torch 2.2.2, which supports Python ≤ 3.12, so the Python version has to
be pinned (if you install with pip/uv instead, use Python 3.10–3.12).

## Web UI

You can host the web UI locally with:

```bash
uvx muscriptor serve
```

This gives you the same UI as hosted on https://muscriptor.kyutai.org/, just with a different look.

The sheet music download needs **MuseScore 4 or newer** installed separately (see
[Sheet music](#sheet-music) below). Without it, everything except that download still works.

## Command-line interface (CLI)

```bash
uvx muscriptor transcribe path/to/audio_file.wav
```

See `--help` for all the options.

### Sheet music

Using the CLI with `--format sheets` engraves the transcription as readable notation instead of
writing a single MIDI file.

```bash
muscriptor transcribe audio.wav --format sheets --output score/
```

The output structure looks like this:

```
score/
├── score.mid                       the transcription, as quantized MIDI
├── score.musicxml                  the engraved score, as MusicXML
├── full_score.pdf                  every instrument on one system
├── 01_electric_guitar.pdf          one PDF per instrument …
├── 01_electric_guitar_tab.pdf      … and a tablature PDF for fretted ones
├── 02_electric_bass.pdf
├── 02_electric_bass_tab.pdf
└── 03_drum_kit.pdf
```

This needs **MuseScore 4 or newer** installed separately. Downloads for every
platform are at [musescore.org/en/download](https://musescore.org/en/download).
Set `$MUSCRIPTOR_MUSESCORE` if it lives somewhere unusual.

It works best if there is a steady tempo (i.e. playing with a metronome), because
that allows us to quantize the notes (snap them to a grid) for a cleaner transcription.
Rubato recordings will work significantly worse.

## Using from Python

MuScriptor is also on PyPI, so you can install it with with uv (recommended) or with pip:

```bash
uv add muscriptor
```

```bash
pip install muscriptor
```

Ask your coding agent to show you around the codebase.

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
GPU. On Apple Silicon the model runs on Metal (MPS) automatically.

## Developing

To set up for development, get [uv](https://docs.astral.sh/uv/getting-started/installation/),
clone this repo and run:
```bash
uv sync
```

For the web UI, you also need [pnpm](https://pnpm.io/installation) and Node
(can be installed [via pnpm](https://pnpm.io/cli/runtime)).
Then run:

```bash
cd web
pnpm install
pnpm run build
```

If you're not editing the frontend, you only need to do this once.
If you are, run `pnpm dev` instead for a hot-reloading dev server.
Start the backend alongside it with it using `uv run muscriptor serve --port 8222`
and then open the frontend on http://localhost:5173/.

### Run

After this setup, you can run Muscriptor from your local repository using
`uv` (note - not `uvx` like before):

```bash
uv run muscriptor serve
# or 
uv run muscriptor transcribe path/to/audio_file.wav
```

Again, see `--help` for more options.

## License

The code in this repository is released under the [MIT license](LICENSE).

The model weights, published on
[HuggingFace](https://huggingface.co/MuScriptor), are released under the
[CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/)
(non-commercial use).

The MuseScore General SoundFont downloaded for playback is
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
