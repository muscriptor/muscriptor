---
license: cc-by-nc-4.0
library_name: muscriptor
extra_gated_prompt: "MuScriptor is the result of a research collaboration between Mirelo and Kyutai whose purpose is to transcribe audio to MIDI/music sheet. It is provided primarily for research purposes under the CC BY-NC 4.0 licence supplemented by the below specific conditions of use.\nSpecific conditions of use: MuScriptor and any generated content by MuScriptor are provided as is without any warranty of any kind, including but not limited to any warranty of non-infringement. Use of MuScriptor and its output must comply with all applicable laws and must not result in, involve, or facilitate any illegal or unauthorized activity. Prohibited uses include, without limitation, inputting music files and transcribing them to MIDI/music sheet without having all the necessary rights, including intellectual property rights, under applicable laws. Accordingly, users of MuScriptor undertake and warrant to have all the necessary rights, including intellectual property rights, in connection with their use of MuScriptor and its output. We disclaim all liability for any non-compliant use and users of MuScriptor shall indemnify, defend, and hold harmless Mirelo and Kyutai from and against any and all claims, damages, losses, liabilities, and expenses (including reasonable attorneys' fees) incurred by Mirelo and/or Kyutai arising out of or resulting from their failure to comply with the terms of the CC BY-NC 4.0 licence and/or these specific conditions of use."
extra_gated_fields:
  Company or university if applicable: text
  I am a:
    type: select
    options: 
      - Musician
      - AI Researcher
      - Other
tags:
- music
- music-transcription
- automatic-music-transcription
- amt
- audio-to-midi
- midi
- music-information-retrieval
- transformer
- pytorch
---

# MuScriptor — large (≈1.3B)

**MuScriptor** is an open-weight model for **general-purpose, multi-instrument automatic music transcription (AMT)**: it converts a music recording (any genre, multiple simultaneous instruments) into a stream of notes played. This repository hosts the **large** variant (≈1.3B parameters) — the **flagship, best-quality checkpoint**.

For a smaller footprint use [`muscriptor-medium`](https://huggingface.co/MuScriptor/muscriptor-medium) (≈300M, good trade-off) or [`muscriptor-small`](https://huggingface.co/MuScriptor/muscriptor-small) (≈100M, fastest).

- Developed by [Mirelo](https://www.mirelo.ai/) x [kyutai](https://kyutai.org/)
- 📄 Paper: *MuScriptor: An Open Model for Multi-Instrument Music Transcription* — Rouard, Krause, Roebel, Simon-Gabriel, Défossez (2026). _<!-- TODO: add arXiv link once public; it will auto-cross-link on the Hub -->_
- 💻 Code: <https://github.com/muscriptor/muscriptor>
- 🔊 Audio samples: <https://muscriptor.github.io>

## Table of contents

- [Quickstart](#quickstart)
- [Model description](#model-description)
- [Model variants](#model-variants)
- [Intended uses & limitations](#intended-uses--limitations)
- [Instrument conditioning](#instrument-conditioning)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)

## Quickstart

Install the `muscriptor` package (it uses `huggingface_hub` to fetch weights automatically):

```bash
pip install git+https://github.com/muscriptor/muscriptor.git
# TODO (PyPI release forthcoming: pip install muscriptor)
```

### Python

```python
from pathlib import Path
from muscriptor import TranscriptionModel

# "large" resolves to hf://MuScriptor/muscriptor-large and downloads on first use.
model = TranscriptionModel.load_model("large")

# Get a MIDI file directly:
Path("out.mid").write_bytes(model.transcribe_to_midi("audio.wav"))

# Or stream note events as they are transcribed:
for event in model.transcribe("audio.wav"):
    print(event)  # NoteStartEvent / NoteEndEvent / ProgressEvent
```

`load_model` accepts a size keyword (`"small"`/`"medium"`/`"large"`), a local `.safetensors` path, or an `hf://` / `https://` URL. Weights loaded by size keyword (or any `hf://` URL) are cached in the standard Hugging Face cache (`~/.cache/huggingface/hub`, configurable via `HF_HOME`); weights fetched from a plain `http(s)://` URL are cached under `~/.cache/muscriptor/`. Input audio can be WAV or any format `libsndfile` reads (mp3, flac, ogg, m4a, …); it is resampled to 16 kHz mono internally.

### CLI

```bash
muscriptor transcribe --model large audio.wav -o out.mid
```

## Model description

MuScriptor performs transcription by **autoregressively predicting a MIDI-like token sequence** given the mel-spectrogram of a short audio segment, following the sequence-to-sequence AMT paradigm (cf. MT3). It deliberately avoids complex architectural tweaks in favor of a simple, decoder-only Transformer.

- **Architecture:** decoder-only Transformer (this variant: `dim=1536`, `num_heads=24`, `num_layers=48`).
- **Input:** raw waveform (16 kHz, mono) of a 5-second segment → mel-spectrogram (STFT `n_fft=2048`, hop 160 → 100 Hz frame rate, 512 mel bins). The spectrogram is projected to the model dimension and used as a prefix condition.
- **Output tokenization:** MT3-like note events; the 128 MIDI programs are mapped to **36 instrument subgroups** using the `MT3_FULL_PLUS` taxonomy. Decoding is greedy (argmax) by default, with optional classifier-free guidance (CFG).
- **Inference:** audio is processed in 5-second chunks; note events are emitted in temporal order. Optional **instrument conditioning** stabilizes predictions across chunk boundaries and lets you restrict/customize the transcription (see below).

**Note on the representation:** the tokenizer recovers onset/offset timing, pitch, and instrument, but **not velocity**. It also cannot represent two notes of the same pitch and instrument sounding at the same time. Drums are onset-only.

## Model variants

| Repo | Params | `dim` | heads | layers | Notes |
|---|---|---|---|---|---|
| [`muscriptor-small`](https://huggingface.co/MuScriptor/muscriptor-small) | ≈100M | 768 | 12 | 14 | smallest / fastest |
| [`muscriptor-medium`](https://huggingface.co/MuScriptor/muscriptor-medium) | ≈300M | 1024 | 16 | 24 | good trade-off |
| [`muscriptor-large`](https://huggingface.co/MuScriptor/muscriptor-large) | ≈1.3B | 1536 | 24 | 48 | **this model** · best quality |

All variants share the same input pipeline, tokenizer, and training recipe; they differ only in latent dimension, attention heads, and depth.

## Intended uses & limitations

**Intended uses**
- General-purpose transcription of real, multi-instrument music across genres (classical → heavy metal) into MIDI.
- A building block for music information retrieval (chord/key recognition), musicological analysis, generative-modeling data pipelines, and tools for musicians.

**Out of scope / use with care**
- Not a substitute for a hand-annotated score; expect errors, especially on dense mixes, unusual timbres, and heavily processed audio.
- Velocity/dynamics are **not** produced (see note above).
- Onset/offset precision is lower for some styles (e.g. choral music), and exact offsets are inherently harder than onsets.

**Limitations & biases**
- Training data skews toward pop and Western classical music, and the instrument distribution is long-tailed (piano/guitar/bass/drums are most frequent). Rare instruments and underrepresented genres may be transcribed less reliably.
- The fixed `MT3_FULL_PLUS` 36-group instrument taxonomy limits instrument granularity.
- Simultaneous same-pitch/same-instrument notes cannot be represented by the tokenizer.

## Instrument conditioning

The model can be told which instrument groups are present in the track. Supplying the correct set improves quantitative scores and produces more coherent instrument assignments across segments.

```python
from muscriptor.tokenizer.mt3 import MT3_FULL_PLUS_GROUP_NAMES

# `instrument_group` is a space-separated string of MT3_FULL_PLUS group IDs.
# Convert readable group names to IDs:
names = ["acoustic_piano", "acoustic_guitar", "acoustic_bass"]
instrument_group = " ".join(str(MT3_FULL_PLUS_GROUP_NAMES[n]) for n in names)  # -> "0 4 7"

# Only expect piano, acoustic guitar and bass in this track:
model.transcribe_to_midi("audio.wav", instrument_group=instrument_group)
```

```bash
muscriptor transcribe --model large --instruments "acoustic_piano,acoustic_guitar,acoustic_bass" audio.wav -o out.mid
muscriptor list-instruments   # show all available group names
```

## Evaluation

Metrics are instrument-agnostic F1 scores computed with [`mir_eval`](https://github.com/craffel/mir_eval).

### Headline results on `D_Test`

`D_Test` is the authors' held-out test set of 372 multi-instrument tracks. Results below are for this 1.3B model with the full training pipeline (`D_Synth` + `D_Real` + `D_RL`), CFG = 2:

| Model | Onset F1 | Frame F1 | Offset F1 | Drums F1 | Multi F1 |
|---|---|---|---|---|---|
| YourMT3+ (baseline) | 32.5 | 45.5 | 17.8 | 41.4 | 21.9 |
| **MuScriptor 1.3B** | **60.4** | **72.4** | **48.6** | **49.6** | **47.8** |

### Model-size comparison

F1 ↑ on `D_Test` from the paper's scaling study (models trained on `D_Real` only, CFG = 2). Note these ablation numbers omit synthetic pre-training and RL, so they are **lower** than the full-pipeline results above:

| Variant | Params | Onset | Frame | Offset | Drums | Multi |
|---|---|---|---|---|---|---|
| `muscriptor-small` | 100M | 51.2 | 67.2 | 38.7 | 41.5 | 38.2 |
| `muscriptor-medium` | 300M | 52.4 | 68.0 | 40.3 | 42.0 | 39.7 |
| **`muscriptor-large`** | **1.3B** | **53.2** | **68.7** | **41.0** | **42.5** | **40.5** |


## Citation

```bibtex
@inproceedings{muscriptor2026,
  title     = {MuScriptor: An Open Model for Multi-Instrument Music Transcription},
  author    = {Rouard, Simon and Krause, Michael and Roebel, Axel and
               Simon-Gabriel, Carl-Johann and D{\'e}fossez, Alexandre},
  year      = {2026},
  note      = {Kyutai, Mirelo AI, IRCAM}
}
```

<!-- TODO: replace with the final published citation (venue / arXiv id) once available. -->

## License

Code released under the [MIT License](https://github.com/muscriptor/muscriptor/blob/main/LICENSE). Weights released under CC-BY-NC.
