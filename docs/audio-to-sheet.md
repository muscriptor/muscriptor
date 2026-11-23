# Audio → sheet music (`muscriptor audio-to-sheet`)

Turns a recording into a score a musician can read: separate parts per
instrument, readable rhythms, chord symbols, key and time signatures, tempo,
rehearsal marks — as MusicXML (open/edit in MuseScore Studio) and PDF.

```bash
uv pip install -e ".[sheet]"          # or: pip install "muscriptor[sheet]"
muscriptor download-models            # optional: fetch every model up front
muscriptor audio-to-sheet "song.mp3"  # → output/song/score.musicxml, score.pdf, …
```

## Pipeline

Transcription (what sounded) and notation (how to write it) are separate
layers, joined by an intermediate representation (the Notation IR):

```
audio ─┬─ separation      Demucs htdemucs_6s → drums, bass, vocals, guitar, piano, other
       ├─ transcription   MuScriptor on the mix (all instruments, instrument labels)
       │                  + MuScriptor on the vocals/bass stems, instrument-constrained;
       │                    stem notes replace that family's notes from the mix
       ├─ beats           beat_this on the mix (tempo map follows the performance)
       └─ audio chords    lv-chordia on the mix (large vocabulary, root/quality/bass)
raw performance notes ── raw.mid
  → cleanup        duplicates, split fragments, ghost/harmonic notes, octave errors,
                   range, monophony (bass/voice/horns), polyphony limits,
                   label noise (one bass part, one voice), sparse parts merged/dropped
  → meter          detected meter, or inferred from chord changes / bass / kick
                   when the tracker's downbeats disagree; leading empty bars trimmed
  → quantization   per-beat grid (1, 2, 4 | 3, 6) chosen by Viterbi: timing error
                   vs. readability, binary↔triplet switches penalised; swing detected
                   and written as straight eighths + "Swing"
  → sustain        notes held over short rests (let ring / pedal)
  → part compilers grand staff with cost-based hand split (keyboards, string
                   sections), ≤ 2 voices per staff, clefs from the instrument and the
                   range actually played, guitar/bass written an octave up,
                   drums: GM map → percussion staff, hands up / feet down
  → harmony        chord per beat from audio + notes + bass (Viterbi, changes cheap on
                   bar lines), slash chords only for a dominant bass note, key
                   (Krumhansl profiles + tonic/cadence evidence), local key changes
  → structure      bar-level self-similarity → sections → rehearsal letters
Notation IR ── notation.json
  → engraving rules  bar lines, beat-visible ties, dotted values, eighth–quarter–eighth
                     syncopation, triplet brackets per beat, key-aware spelling,
                     accidentals per bar, whole-bar rests
  → MusicXML 4.0     deterministic writer ── score.musicxml, parts/*.musicxml
  → MuseScore 4 CLI  -o per file ── score.mscz, score.pdf, parts/*.pdf
```

Code: `muscriptor/sheet/` — one module per stage (`separation`, `transcription`,
`cleanup`, `rhythm`, `voices`, `harmony`, `structure`, `notation`, `ir`,
`musicxml`, `render`, `lint`, `pipeline`, `config`).

### Why these components

| Stage | Choice | Why |
|---|---|---|
| Transcription | MuScriptor (existing) | Strongest open multi-instrument model; its instrument labels are the instrument detection. |
| Separation | Demucs `htdemucs_6s` | MIT code + weights, 6 stems incl. guitar/piano, installs alongside this project. BS-RoFormer via `audio-separator` (the research's first pick) pins an older `rotary-embedding-torch` than `beat_this` needs. |
| Beats | beat_this (existing) | No DBN, follows tempo changes. |
| Audio chords | lv-chordia (ISMIR 2019) | MIT, weights bundled, CPU-fast, outputs root/quality/bass (slash chords). ChordFormer/BTC need separate weight hosting. |
| Chord fusion, key, quantization, voices, structure | Own algorithms | No packaged open tool does these for band scores (research: "build ourselves"). |
| Piano PM2S (DuET) | Not used | No installable release; the cost-based hand split + voice compiler is the fallback. |
| Structure (All-In-One) | Not used | Needs NATTEN (no Windows/recent-torch builds); a self-similarity novelty stand-in is used. |
| MusicXML | Own writer | Deterministic, no music21 dependency; validated by MuseScore import and by per-measure duration checks. |

## Output

```
output/<song>/
├── stems/                 drums.wav bass.wav vocals.wav guitar.wav piano.wav other.wav
├── transcription/
│   ├── mix.json           raw notes from the mix
│   ├── vocals.json …      raw notes from re-transcribed stems
│   ├── beats.json         tracked beats / tempo / meter
│   ├── audio_chords.json  lv-chordia segments
│   └── cleaned.json       notes after cleanup
├── raw.mid                merged raw transcription (performance timing + tempo map)
├── reconstructed.mid      cleaned, quantized, voice-separated
├── chords.json            key, key changes, chord symbols (bar, beat, seconds)
├── structure.json         sections
├── notation.json          Notation IR
├── score.musicxml         ← open in MuseScore Studio
├── score.mscz / score.pdf
├── parts/NN_<part>.musicxml / .pdf
├── report.json            per-stage status/timing, note counts, readability metrics
└── pipeline.log
```

Rerunning on the same output directory reuses stems, transcriptions, beats and
audio chords (`--no-resume` recomputes), so reconstruction settings can be
tuned in seconds.

## Configuration

`--config file.toml` (see [`sheet.example.toml`](../sheet.example.toml)), then
environment variables, then CLI flags.

| CLI | Config | Env | |
|---|---|---|---|
| `--quality` | `quality` | | `draft` (small model, no separation), `standard`, `high` (large model, all stems) |
| `--device` | `device` | `MUSCRIPTOR_DEVICE` | `auto`, `cpu`, `cuda`, `mps` |
| `--model` | `transcription.model` | `MUSCRIPTOR_MODEL` | `small`/`medium`/`large`/path/URL |
| `--separation` | `separation.method` | `MUSCRIPTOR_SEPARATION` | `demucs`/`none` |
| `--stems` | `transcription.stems` | | e.g. `vocals,bass,guitar` |
| `--musescore` | `render.musescore` | `MUSCRIPTOR_MUSESCORE` | MuseScore 4 executable |
| `-o` | `output_dir` | `MUSCRIPTOR_OUTPUT_DIR` | |
| `--beats-per-bar`, `--beat-scale`, `--bpm`, `--swing` | `rhythm.*` | | fix meter/tempo detection |
| `--no-audio-chords`, `--no-pdf`, `--title`, `--no-resume` | | | |

Model caches: MuScriptor weights and beat_this in the HuggingFace cache
(`HF_HOME`; gated — log in first, see the main README), Demucs in the torch.hub
cache (`TORCH_HOME`), lv-chordia inside its package. Nothing is downloaded
again once cached; `muscriptor download-models` fetches everything explicitly,
and `HF_HUB_OFFLINE=1` forbids downloads.

MuseScore is found via `--musescore`, `$MUSCRIPTOR_MUSESCORE`, `PATH`, and the
default install locations (Windows `Program Files\MuseScore 4`, macOS
`/Applications`, Linux AppImage in `~`). Without it, everything up to
`score.musicxml` is still written.

## Hardware

Measured on an RTX 5060 Ti 16 GB, 3:33 song, `standard`, nothing cached:
Demucs 7 s, MuScriptor medium 114 s for the mix + vocals + bass stems, beats
3.5 s, lv-chordia 4 s, all reconstruction stages < 1 s, MuseScore 22 s
(full score + 7 part PDFs) — about 2.5 minutes in total; a re-run with
cached transcriptions takes ~25 s, nearly all of it MuseScore. Peak VRAM is
the transcription model (medium fits easily in 8 GB; large wants ~12 GB+).
CPU-only works with `--quality draft` (small model, no separation) but is slow.

## Readability metrics

`report.json → readability` counts, per part: notes, rests, short rests
(< eighth), 32nd-or-shorter notes, ties, tuplet notes, accidentals, empty bars,
two-voice bars, events per bar — the "linter" view of a score. Use it to
compare settings on the same song.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Bar lines off by a beat or two | `--first-downbeat <seconds>` (a bar line in the recording) |
| Everything in 16ths/32nds, tempo looks double/half | `--beat-scale 0.5` / `--beat-scale 2` |
| Wrong meter (3/4 vs 4/4) | `--beats-per-bar 3` |
| Straight song written with "Swing" (or vice versa) | `--swing off` / `--swing on` |
| `MuseScore was not found` | install MuseScore 4, or `--musescore <path>` / `$MUSCRIPTOR_MUSESCORE` |
| `Source separation needs Demucs` | `uv pip install -e ".[sheet]"` or `--separation none` |
| No chords from audio | install the `sheet` extra (lv-chordia); on Python 3.13 it also needs `audioop-lts` |
| HuggingFace 401/403 | accept the model license and log in (main README) |

## Known limitations

- Transcription errors pass through: wrong pitches or instruments from the
  model can be cleaned up but not corrected. Instrument labels are
  MuScriptor's; parts it does not detect cannot appear.
- Pitch spelling and enharmonics follow the key; chromatic passages may be
  spelled sub-optimally. Transposing instruments are written at concert pitch.
- No guitar TAB in this pipeline (the existing `--format sheets` still makes TAB
  from MuseScore's MIDI import); no lyrics, dynamics, articulations, ornaments,
  repeats or pedal markings.
- Key mode between relative major/minor is decided from chord evidence and can
  go either way on modal pop; the key signature is the same.
- Sections are labelled with letters (A, B, …), not verse/chorus.
- Triplets are only written within a beat (no quarter-note triplets).
- Recordings without a clear beat (rubato, free time) still produce a dense score.
