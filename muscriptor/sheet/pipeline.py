"""The audio-to-sheet pipeline: stages, artifacts, report.

Output directory layout (every stage leaves its result behind, so a bad score
can be traced to the stage that went wrong):

    <out>/
    ├── stems/                 separated stems (drums, bass, vocals, guitar, piano, other)
    ├── transcription/
    │   ├── mix.json           raw MuScriptor notes from the full mix
    │   ├── <stem>.json        raw notes from re-transcribed stems
    │   ├── beats.json         tracked beats, tempo, meter
    │   ├── audio_chords.json  lv-chordia output on the mix
    │   └── cleaned.json       notes after cleanup
    ├── raw.mid                merged raw transcription (performance timing, tempo map)
    ├── reconstructed.mid      quantized, cleaned, voice-separated MIDI
    ├── chords.json            chord symbols (time, bar, beat, label) and key(s)
    ├── structure.json         sections
    ├── notation.json          the Notation IR
    ├── score.musicxml         the score (open this in MuseScore)
    ├── score.mscz             MuseScore project (proves MuseScore imported it)
    ├── score.pdf              engraved full score
    ├── parts/NN_<part>.musicxml / .pdf
    ├── report.json            stage timings, statistics, readability metrics
    └── pipeline.log

With `resume` (the default), expensive artifacts already present (stems,
transcriptions, beats, audio chords) are reused, so the reconstruction stages
can be re-run and tuned in seconds.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from muscriptor.sheet import (
    cleanup,
    harmony,
    notation,
    rhythm,
    sections,
    structure,
    voices,
)
from muscriptor.sheet.config import PipelineConfig, config_to_dict
from muscriptor.sheet.instruments import STEM_GROUPS, spec_for
from muscriptor.sheet.ir import (
    PerformanceNote,
    QuantizedNote,
    Score,
    load_performance,
    save_performance,
    score_to_json,
)
from muscriptor.sheet.lint import lint_score
from muscriptor.sheet.musicxml import single_part, write_musicxml

logger = logging.getLogger("muscriptor.sheet")


class PipelineError(RuntimeError):
    pass


# Bump when stem transcription changes in a way that makes cached stem
# transcriptions stale (transcription.transcribe_stem).
STEM_TRANSCRIPTION = "detect+condition/1"


@dataclass
class PipelineResult:
    output_dir: Path
    artifacts: dict[str, str] = field(default_factory=dict)
    report: dict = field(default_factory=dict)
    score: Score | None = None


class _Stages:
    """Times each stage and records its status in the report."""

    def __init__(self):
        self.records: list[dict] = []

    @contextlib.contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        rec = {"stage": name, "status": "running"}
        self.records.append(rec)
        logger.info("── %s", name)
        try:
            yield rec
        except Exception as e:
            rec["status"] = "failed"
            rec["error"] = f"{type(e).__name__}: {e}"
            raise
        else:
            if rec["status"] == "running":
                rec["status"] = "ok"
        finally:
            rec["seconds"] = round(time.perf_counter() - t0, 2)


# ---------------------------------------------------------------------------
# Beat grid caching


def _grid_to_json(grid) -> dict | None:
    if grid is None:
        return None
    return {
        "bpm": grid.bpm,
        "beats_per_bar": grid.beats_per_bar,
        "first_downbeat": grid.first_downbeat,
        "steady": grid.steady,
        "beats": [float(b) for b in grid.beats] if grid.beats is not None else None,
    }


def _grid_from_json(data: dict | None):
    from muscriptor.utils.beats import BeatGrid

    if data is None:
        return None
    return BeatGrid(
        bpm=data["bpm"],
        beats_per_bar=data["beats_per_bar"],
        first_downbeat=data["first_downbeat"],
        beats=np.asarray(data["beats"]) if data["beats"] is not None else None,
        steady=data["steady"],
    )


# ---------------------------------------------------------------------------
# MIDI writers


def write_raw_midi(notes: list[PerformanceNote], grid, path: Path) -> None:
    """The merged raw transcription, performance timing, through the existing MIDI writer."""
    from muscriptor.tokenizer.notes import DRUM_PROGRAM, Note
    from muscriptor.utils.midi import notes_to_midi

    tn, names = [], {}
    for n in notes:
        spec = spec_for(n.instrument)
        program = DRUM_PROGRAM if n.is_drum else spec.gm_program
        names[program] = spec.name
        tn.append(
            Note(
                is_drum=n.is_drum,
                program=program,
                onset=n.onset,
                offset=max(n.offset, n.onset + 0.01),
                pitch=n.pitch,
            )
        )
    midi = notes_to_midi(tn, program_names=names, beat_grid=grid)
    midi.save(str(path))


def write_reconstructed_midi(
    parts: list[tuple[str, int, bool, list[QuantizedNote]]],
    meter: rhythm.MeterMap,
    path: Path,
    ticks_per_beat: int = 480,
) -> None:
    """Quantized parts as a type-1 MIDI file, one track per part, with the tempo map."""
    from mido import Message, MetaMessage, MidiFile, MidiTrack

    mid = MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    meta = MidiTrack()
    meta.append(
        MetaMessage(
            "time_signature", numerator=meter.beats_per_bar, denominator=4, time=0
        )
    )
    last = 0
    for tick, tempo in meter.timeline.tempo_changes(ticks_per_beat):
        meta.append(MetaMessage("set_tempo", tempo=tempo, time=tick - last))
        last = tick
    mid.tracks.append(meta)
    channel = 0
    for name, program, is_drum, notes in parts:
        tr = MidiTrack()
        tr.append(MetaMessage("track_name", name=name, time=0))
        if is_drum:
            ch = 9
        else:
            ch = channel if channel < 9 else channel + 1
            channel += 1
            ch = min(ch, 15)
            tr.append(Message("program_change", program=program, channel=ch, time=0))
        events = []
        for n in notes:
            on = int(n.onset * ticks_per_beat)
            off = (
                int(n.offset * ticks_per_beat)
                if not is_drum
                else on + ticks_per_beat // 8
            )
            events.append((on, 1, n.pitch, n.velocity))
            events.append((max(off, on + 1), 0, n.pitch, 0))
        events.sort(key=lambda e: (e[0], e[1]))
        t = 0
        for tick, is_on, pitch, vel in events:
            msg = "note_on" if is_on else "note_off"
            tr.append(Message(msg, note=pitch, velocity=vel, channel=ch, time=tick - t))
            t = tick
        mid.tracks.append(tr)
    mid.save(str(path))


# ---------------------------------------------------------------------------


def _tempo_marks(meter: rhythm.MeterMap, n_measures: int) -> dict[int, float]:
    """Metronome marks: one at the start, more only for lasting tempo changes."""
    tempos = np.array([meter.measure_tempo(m) for m in range(n_measures)])
    if meter.steady or n_measures < 8:
        return {0: float(round(meter.bpm))}
    marks = {0: float(round(np.median(tempos[: min(8, n_measures)])))}
    current = marks[0]
    m = 4
    while m < n_measures - 3:
        window = float(np.mean(tempos[m : m + 4]))
        if abs(window - current) / current > 0.07:
            current = float(round(window))
            marks[m] = current
            m += 8
        else:
            m += 1
    return marks


def _fill_gap_beats(spec, cfg: PipelineConfig) -> Fraction:
    if spec.role == "drums":
        return Fraction(0)
    if spec.role in ("keyboard", "guitar"):
        return Fraction(cfg.rhythm.fill_gaps_beats_keyboard).limit_denominator(8)
    return Fraction(cfg.rhythm.fill_gaps_beats).limit_denominator(8)


def _bar_chord_pcs(segments, n_measures: int, bpb: int) -> list[set[int]]:
    out: list[set[int]] = [set() for _ in range(n_measures)]
    for s in segments:
        for b in range(s.start_beat, s.end_beat):
            m = b // bpb
            if 0 <= m < n_measures:
                out[m] |= set(s.chord.pcs())
    return out


def _setup_logging(out_dir: Path) -> logging.Handler:
    handler = logging.FileHandler(out_dir / "pipeline.log", mode="w", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.addHandler(handler)
    if root.level > logging.INFO or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)
    return handler


def run_pipeline(
    audio: Path,
    cfg: PipelineConfig,
    out_dir: Path,
    model=None,
) -> PipelineResult:
    """Run every stage on `audio`, writing artifacts into `out_dir`."""
    audio = Path(audio)
    if not audio.is_file():
        raise PipelineError(f"audio file not found: {audio}")
    from muscriptor.sheet.config import validate_config

    try:
        validate_config(cfg)
    except ValueError as e:
        raise PipelineError(str(e)) from e
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "transcription").mkdir(exist_ok=True)
    handler = _setup_logging(out_dir)
    stages = _Stages()
    result = PipelineResult(out_dir)
    art = result.artifacts
    title = cfg.title or audio.stem
    report: dict = {
        "audio": str(audio),
        "title": title,
        "config": config_to_dict(cfg),
        "stages": stages.records,
    }
    result.report = report

    model_holder = {"model": model}

    def get_model():
        if model_holder["model"] is None:
            from muscriptor.transcription_model import TranscriptionModel

            logger.info(
                "loading MuScriptor (%s) on %s",
                cfg.transcription.model or "medium",
                cfg.device,
            )
            model_holder["model"] = TranscriptionModel.load_model(
                weights_path=cfg.transcription.model,
                device=cfg.torch_device,
                dtype=cfg.transcription.dtype,
            )
        return model_holder["model"]

    try:
        # 1. Audio ----------------------------------------------------------
        with stages.stage("audio") as rec:
            import soundfile as sf

            try:
                info = sf.info(str(audio))
            except Exception as e:
                raise PipelineError(f"cannot read audio {audio}: {e}") from e
            rec.update(
                duration=round(info.duration, 2),
                sample_rate=info.samplerate,
                channels=info.channels,
            )
            report["duration"] = info.duration

        # 2. Separation -----------------------------------------------------
        from muscriptor.sheet.config import resolve_separation_method, resolve_stems
        from muscriptor.sheet.separation import (
            cached_stems,
            ingest_audio,
            make_separator,
            separator_signature,
            write_manifest,
        )

        stems_dir = out_dir / "stems"
        stems = []
        sig: dict | None = None
        with stages.stage("separation") as rec:
            sep_cfg = dataclasses.replace(
                cfg.separation, method=resolve_separation_method(cfg)
            )
            rec["method"] = sep_cfg.method
            if sep_cfg.method == "none":
                rec["status"] = "skipped"
            else:
                # Decode once: separators and stem levels all read this file.
                mix_wav = ingest_audio(audio, out_dir / "input.wav")
                sig = separator_signature(sep_cfg)
                rec["separator"] = sig
                cached = cached_stems(stems_dir, mix_wav, sig) if cfg.resume else None
                if cached:
                    stems = cached
                    rec["status"] = "cached"
                else:
                    sep = make_separator(sep_cfg, cfg.torch_device)
                    if stems_dir.is_dir():
                        for old in stems_dir.glob("*.wav"):
                            old.unlink()
                    stems = sep.separate(mix_wav, stems_dir)
                    write_manifest(stems_dir, sig)
            rec["stems"] = {s.name: round(s.level_db, 1) for s in stems}
            rec["active"] = {
                s.name: round(s.activity.active_ratio(sep_cfg.gate_db), 2)
                for s in stems
                if s.activity is not None
            }
            if stems:
                art["stems"] = str(stems_dir)

        # 3. Transcription --------------------------------------------------
        from muscriptor.sheet.transcription import (
            gate_notes,
            merge_stem_notes,
            transcribe_audio,
            transcribe_stem,
        )

        tdir = out_dir / "transcription"
        with stages.stage("transcription") as rec:
            # What produced the cached transcriptions: a stem transcription is
            # only reused if it came from these very stems, this model and
            # this stem-transcription method (older runs have no key).
            key_path = tdir / "cache.json"
            key = {
                "model": cfg.transcription.model or "medium",
                "beam_size": cfg.transcription.beam_size,
                "separator": sig,
                "stem_mode": STEM_TRANSCRIPTION,
            }
            old_key = (
                json.loads(key_path.read_text(encoding="utf-8"))
                if key_path.is_file()
                else {}
            )
            mix_ok = (
                old_key.get("model", key["model"]) == key["model"]
                and old_key.get("beam_size", key["beam_size"]) == key["beam_size"]
            )
            stems_ok = old_key == key
            if (
                cfg.resume
                and not stems_ok
                and any((tdir / f"{n}.json").is_file() for n in STEM_GROUPS)
            ):
                logger.info(
                    "transcription: cached stem transcriptions are from another "
                    "separator/model/method, re-transcribing the stems"
                )
            mix_path = tdir / "mix.json"
            if cfg.resume and mix_ok and mix_path.is_file():
                mix_notes = load_performance(mix_path)
                rec["status"] = "cached"
            else:
                mix_notes = transcribe_audio(
                    get_model(), audio, "mix", beam_size=cfg.transcription.beam_size
                )
                save_performance(mix_notes, mix_path)
            art["transcription_mix"] = str(mix_path)
            stem_notes: dict[str, list[PerformanceNote]] = {}
            by_name = {s.name: s for s in stems}
            wanted = resolve_stems(cfg.transcription.stems, list(by_name))
            rec["stems"] = wanted
            detected: dict[str, list[str]] = {}
            absent: set[str] = set()
            for name in wanted:
                stem = by_name.get(name)
                if stem is None or name not in STEM_GROUPS:
                    continue
                if stem.level_db < cfg.separation.silence_db:
                    # The separator has a stem for this family and found it
                    # silent: whatever the mix transcription labelled so is
                    # another instrument (e.g. strings read as guitar).
                    absent.add(name)
                    logger.info(
                        "transcription: %s stem is silent (%.1f dB): dropping "
                        "that family from the mix transcription",
                        name,
                        stem.level_db,
                    )
                    continue
                path = tdir / f"{name}.json"
                if cfg.resume and stems_ok and path.is_file():
                    stem_notes[name] = load_performance(path)
                else:
                    rec["status"] = "running"  # not all of it came from the cache
                    stem_notes[name], detected[name] = transcribe_stem(
                        get_model(),
                        stem.path,
                        name,
                        list(STEM_GROUPS[name]),
                        beam_size=cfg.transcription.beam_size,
                    )
                    save_performance(stem_notes[name], path)
                art[f"transcription_{name}"] = str(path)
                if name in detected:
                    rec.setdefault("detected", {})[name] = detected[name]
                # Gate after caching, so the threshold can be tuned on resume.
                stem_notes[name] = gate_notes(
                    stem_notes[name], stem.activity, cfg.separation.gate_db, name
                )
            key_path.write_text(json.dumps(key, indent=1), encoding="utf-8")
            raw_notes = merge_stem_notes(
                mix_notes, stem_notes, cfg.transcription.min_stem_notes, absent
            )
            if absent:
                rec["absent"] = sorted(absent)
            rec["notes"] = len(raw_notes)
            rec["by_instrument"] = _count(raw_notes)
            if not raw_notes:
                raise PipelineError("the transcription produced no notes")

        # 4. Beats ----------------------------------------------------------
        with stages.stage("beats") as rec:
            beats_path = tdir / "beats.json"
            if cfg.resume and beats_path.is_file():
                grid = _grid_from_json(json.loads(beats_path.read_text()))
                rec["status"] = "cached"
            else:
                grid = get_model().detect_beat_grid_for(audio, "best-effort")
                beats_path.write_text(json.dumps(_grid_to_json(grid)))
            art["beats"] = str(beats_path)
            if grid is not None:
                rec.update(
                    bpm=round(grid.bpm, 2),
                    beats_per_bar=grid.beats_per_bar,
                    steady=grid.steady,
                    n_beats=len(grid.beats) if grid.beats is not None else 0,
                )
            else:
                rec["status"] = "fallback"

        with stages.stage("raw_midi"):
            write_raw_midi(raw_notes, grid, out_dir / "raw.mid")
            art["raw_midi"] = str(out_dir / "raw.mid")

        # 5. Audio chords ---------------------------------------------------
        audio_chords = None
        with stages.stage("audio_chords") as rec:
            ac_path = tdir / "audio_chords.json"
            if not cfg.harmony.audio_chords:
                rec["status"] = "skipped"
            elif cfg.resume and ac_path.is_file():
                audio_chords = [tuple(x) for x in json.loads(ac_path.read_text())]
                rec["status"] = "cached"
            else:
                try:
                    audio_chords = harmony.run_audio_chords(
                        str(audio), cfg.harmony.chord_dict
                    )
                    ac_path.write_text(json.dumps(audio_chords, indent=0))
                except ImportError:
                    logger.warning(
                        "lv-chordia not installed: chords from the notes alone"
                    )
                    rec["status"] = "unavailable"
            if audio_chords is not None:
                rec["segments"] = len(audio_chords)
                art["audio_chords"] = str(ac_path)

        # 6. Cleanup --------------------------------------------------------
        with stages.stage("cleanup") as rec:
            opts = cleanup.CleanupOptions(
                min_part_notes=cfg.cleanup.min_part_notes,
                min_part_notes_alone=cfg.cleanup.min_part_notes,
                min_duration=cfg.cleanup.min_duration,
                min_duration_sustained=cfg.cleanup.min_duration_sustained,
            )
            clean = cleanup.clean_notes(raw_notes, opts)
            if cfg.cleanup.split_strings:
                secs = (
                    (sections.CELLO, sections.VIOLA, sections.VIOLIN)
                    if cfg.cleanup.string_parts >= 3
                    else (sections.CELLO, sections.VIOLIN)
                )
                sopts = sections.SectionSplitOptions(
                    sections=secs, articulation=cfg.cleanup.strings_articulation
                )
                sources = {s.name: s.path for s in stems}
                sources["mix"] = audio
                clean = sections.split_string_sections(clean, sopts, sources)
            save_performance(clean, tdir / "cleaned.json")
            art["cleaned"] = str(tdir / "cleaned.json")
            rec["notes"] = len(clean)
            rec["by_instrument"] = _count(clean)

        # 7. Meter, swing, quantization, voices ----------------------------
        with stages.stage("rhythm") as rec:
            onsets = [n.onset for n in clean]
            beats_per_bar, first_downbeat = (
                cfg.rhythm.beats_per_bar,
                cfg.rhythm.first_downbeat,
            )
            if (
                grid is not None
                and grid.beats is not None
                and grid.beats_per_bar is None
                and beats_per_bar is None
                and first_downbeat is None
                and cfg.rhythm.beat_scale == 1
                and cfg.rhythm.bpm is None
            ):
                # The tracker's downbeats disagree: infer the bar lines from
                # where the music changes (chords, bass, kick drum).
                events = _downbeat_evidence(clean, audio_chords)
                guess = rhythm.infer_meter_phase(grid.beats, events)
                if guess is not None:
                    beats_per_bar, first_downbeat = (
                        guess.beats_per_bar,
                        float(grid.beats[guess.phase]),
                    )
                    rec["meter_inferred"] = {
                        "beats_per_bar": guess.beats_per_bar,
                        "first_downbeat": round(first_downbeat, 3),
                        "score": round(guess.score, 3),
                        "runner_up": round(guess.runner_up, 3),
                    }
                    logger.info(
                        "meter: inferred %d/4 from chord changes/bass/kick, first downbeat %.2fs (score %.2f vs %.2f)",
                        guess.beats_per_bar,
                        first_downbeat,
                        guess.score,
                        guess.runner_up,
                    )
            meter = rhythm.build_meter_map(
                grid,
                onsets,
                beats_per_bar=beats_per_bar,
                beat_scale=cfg.rhythm.beat_scale,
                first_downbeat=first_downbeat,
                bpm=cfg.rhythm.bpm,
            )
            bpb = meter.beats_per_bar
            positions = np.asarray(meter.beats_at(onsets), dtype=float)
            # Drop the empty bars before the music starts.
            lead = math.floor((positions.min() + 0.12) / bpb) * bpb
            if lead > 0:
                meter = meter.shifted(-lead)
                positions = positions - lead
            if cfg.rhythm.swing == "auto":
                sw = rhythm.detect_swing(positions)
                swing = sw.swing
                rec["swing_ratio"] = round(sw.ratio, 3)
            else:
                swing = cfg.rhythm.swing == "on"
            # The score ends with the bar holding the last note (a quantization
            # step of slack, so a note ending just past a bar line adds no bar).
            last = max(
                float(np.max(meter.beats_at([n.offset for n in clean]))),
                float(positions.max()) + 0.25,
            )
            n_measures = max(1, math.ceil((last - 0.15) / bpb))
            n_beats = n_measures * bpb
            rec.update(
                beats_per_bar=bpb,
                bpm=round(meter.bpm, 2),
                steady=meter.steady,
                swing=swing,
                measures=n_measures,
                leading_beats_trimmed=lead,
            )

            qopts = rhythm.QuantizeOptions(
                allow_triplets=cfg.rhythm.allow_triplets, finest=cfg.rhythm.finest
            )
            by_inst: dict[str, list[PerformanceNote]] = defaultdict(list)
            for n in clean:
                by_inst[n.instrument].append(n)
            part_inputs: list[notation.PartInput] = []
            midi_parts = []
            grid_stats: dict[str, dict] = {}
            for inst in sorted(by_inst, key=lambda i: (spec_for(i).order, i)):
                spec = spec_for(inst)
                # Singers and horns place notes more loosely than a keyboard
                # player: a wider tolerance keeps that from reading as triplets.
                loose = spec.role in (
                    "voice",
                    "winds",
                    "brass",
                    "strings",
                    "string_section",
                )
                part_opts = (
                    dataclasses.replace(qopts, sigma=0.1, family_switch_penalty=4.0)
                    if loose
                    else qopts
                )
                q = rhythm.quantize_part(
                    by_inst[inst], meter, n_beats, swing, part_opts
                )
                q = [n for n in q if n.onset < n_beats]
                if not q:
                    continue
                q = rhythm.fill_gaps(q, _fill_gap_beats(spec, cfg))
                layout, laid = voices.layout_part(spec, q)
                part_inputs.append(notation.PartInput(spec, layout, laid))
                midi_parts.append(
                    (spec.name, spec.gm_program, spec.role == "drums", laid)
                )
                grid_stats[inst] = dict(sorted(_count_grids(q).items()))
            for i, p in enumerate(part_inputs, start=1):
                p.part_id = f"P{i}"
            rec["grids"] = grid_stats
            write_reconstructed_midi(midi_parts, meter, out_dir / "reconstructed.mid")
            art["reconstructed_midi"] = str(out_dir / "reconstructed.mid")

        # 8. Harmony: chords and key ---------------------------------------
        with stages.stage("harmony") as rec:
            beat_times = np.asarray(
                meter.seconds_at(np.arange(n_beats + 1, dtype=float)), dtype=float
            )
            hopts = harmony.HarmonyOptions(min_chord_beats=cfg.harmony.min_chord_beats)
            segments = harmony.recognize_chords(
                clean, beat_times, bpb, audio_chords, hopts
            )
            global_key = harmony.estimate_key(clean, [s.chord for s in segments])
            bar_pcps = np.array(
                [
                    harmony.pitch_class_profile(
                        clean, beat_times[m * bpb], beat_times[(m + 1) * bpb]
                    )
                    for m in range(n_measures)
                ]
            )
            local = (
                harmony.estimate_local_keys(bar_pcps, global_key)
                if cfg.harmony.local_keys
                else [global_key] * n_measures
            )
            keys: dict[int, tuple[int, str]] = {}
            prev = None
            for m, k in enumerate(local):
                if prev is None or k.fifths != prev.fifths:
                    keys[m] = (k.fifths, k.mode)
                prev = k
            chords_by_measure: dict[int, list] = defaultdict(list)
            chord_rows = []
            for s in segments:
                if s.chord.is_none:
                    continue
                m, beat = divmod(s.start_beat, bpb)
                if m >= n_measures:
                    continue
                sym = harmony.chord_symbol(s.chord, local[m], Fraction(beat))
                chords_by_measure[m].append(sym)
                chord_rows.append(
                    {
                        "measure": m + 1,
                        "beat": beat + 1,
                        "start": round(s.start, 3),
                        "end": round(s.end, 3),
                        "beats": s.end_beat - s.start_beat,
                        "symbol": sym.label,
                        "chord": str(s.chord),
                    }
                )
            chords_doc = {
                "key": str(global_key),
                "key_fifths": global_key.fifths,
                "key_changes": [
                    {"measure": m + 1, "fifths": f, "mode": md}
                    for m, (f, md) in sorted(keys.items())
                ],
                "sources": [
                    "lv-chordia (audio)" if audio_chords else None,
                    "notes",
                    "bass line",
                ],
                "chords": chord_rows,
            }
            (out_dir / "chords.json").write_text(
                json.dumps(chords_doc, indent=1, ensure_ascii=False), encoding="utf-8"
            )
            art["chords"] = str(out_dir / "chords.json")
            rec.update(
                key=str(global_key), chords=len(chord_rows), key_changes=len(keys) - 1
            )

        # 9. Structure ------------------------------------------------------
        sections_map: dict[int, str] = {}
        with stages.stage("structure") as rec:
            if cfg.harmony.sections:
                bar_starts = beat_times[::bpb][: n_measures + 1]
                first_bar = int(math.floor(positions.min() / bpb))
                secs = structure.detect_sections(
                    clean,
                    bar_starts,
                    _bar_chord_pcs(segments, n_measures, bpb),
                    first_bar,
                )
                sections_map = {s.start_bar: s.label for s in secs}
                (out_dir / "structure.json").write_text(
                    json.dumps(
                        [
                            {
                                "start_measure": s.start_bar + 1,
                                "end_measure": s.end_bar,
                                "label": s.label,
                                "start": round(float(bar_starts[s.start_bar]), 2),
                            }
                            for s in secs
                        ],
                        indent=1,
                    )
                )
                art["structure"] = str(out_dir / "structure.json")
                rec["sections"] = " ".join(f"{s.label}{s.start_bar + 1}" for s in secs)
            else:
                rec["status"] = "skipped"

        # 10. Notation IR ---------------------------------------------------
        with stages.stage("notation") as rec:
            # Chord symbols live on the top pitched part (lead-sheet style) and
            # on the chord instruments, whose part files a player reads alone.
            top_part = set()
            pitched = [p for p in part_inputs if p.spec.role != "drums"]
            if pitched:
                top_part.add(pitched[0].part_id)
            chord_part_ids = top_part | {
                p.part_id for p in part_inputs if p.spec.role in ("guitar", "keyboard")
            }
            ctx = notation.ScoreContext(
                beats_per_bar=bpb,
                n_measures=n_measures,
                title=title,
                keys=keys,
                tempos=_tempo_marks(meter, n_measures),
                chords=dict(chords_by_measure),
                sections=sections_map,
                chord_parts=chord_part_ids,
                words={0: "Swing"} if swing else {},
                measure_seconds=[
                    float(meter.seconds_at(m * bpb)) for m in range(n_measures)
                ],
                metadata={
                    "key": str(global_key),
                    "bpm": round(meter.bpm, 1),
                    "swing": swing,
                    "source_audio": audio.name,
                },
            )
            score = notation.build_score(part_inputs, ctx)
            score.source = f"Transcribed from {audio.name} by MuScriptor"
            result.score = score
            (out_dir / "notation.json").write_text(
                score_to_json(score), encoding="utf-8"
            )
            art["notation"] = str(out_dir / "notation.json")
            rec["parts"] = [f"{p.id}:{p.name}" for p in score.parts]

        # 11. MusicXML ------------------------------------------------------
        with stages.stage("musicxml") as rec:
            xml_path = write_musicxml(
                score, out_dir / "score.musicxml", chord_part_ids=top_part
            )
            art["musicxml"] = str(xml_path)
            parts_dir = out_dir / "parts"
            parts_dir.mkdir(exist_ok=True)
            # Parts of a previous run (other instruments, other numbering)
            # would otherwise sit next to this run's.
            for old in parts_dir.glob("[0-9][0-9]_*"):
                if old.suffix in (".musicxml", ".pdf", ".mscz"):
                    old.unlink()
            part_files = []
            for i, p in enumerate(score.parts, start=1):
                slug = "".join(
                    ch if ch.isalnum() else "_" for ch in p.name.lower()
                ).strip("_")
                path = write_musicxml(
                    single_part(score, p.id), parts_dir / f"{i:02d}_{slug}.musicxml"
                )
                part_files.append(path)
            rec["bytes"] = xml_path.stat().st_size
            report["readability"] = lint_score(score)

        # 12. MuseScore -----------------------------------------------------
        with stages.stage("musescore") as rec:
            from muscriptor.sheet.render import (
                MuseScoreNotFoundError,
                convert,
                resolve_musescore,
            )

            if not cfg.render.pdf:
                rec["status"] = "skipped"
            else:
                try:
                    binary = resolve_musescore(cfg.render.musescore)
                except MuseScoreNotFoundError as e:
                    logger.warning("MuseScore not found, no PDF: %s", e)
                    rec["status"] = "unavailable"
                    rec["error"] = str(e)
                else:
                    rec["musescore"] = binary
                    convert(binary, xml_path, out_dir / "score.mscz")
                    art["mscz"] = str(out_dir / "score.mscz")
                    convert(binary, xml_path, out_dir / "score.pdf")
                    art["pdf"] = str(out_dir / "score.pdf")
                    if cfg.render.part_pdfs:
                        for path in part_files:
                            convert(binary, path, path.with_suffix(".pdf"))
                        art["parts"] = str(parts_dir)
    finally:
        report["artifacts"] = art
        (out_dir / "report.json").write_text(
            json.dumps(report, indent=1, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        logging.getLogger().removeHandler(handler)
        handler.close()
    return result


def _downbeat_evidence(
    notes: list[PerformanceNote], audio_chords: list[tuple] | None
) -> list[tuple[float, float]]:
    """(time, weight) of events that tend to fall on downbeats."""
    events: list[tuple[float, float]] = []
    if audio_chords:
        prev = None
        for start, _end, label in audio_chords:
            if label not in ("N", "X") and label != prev and prev is not None:
                events.append((float(start), 1.0))
            prev = label
    for n in notes:
        role = spec_for(n.instrument).role
        if role == "bass":
            events.append((n.onset, 0.3))
        elif role == "drums" and n.pitch in (35, 36):
            events.append((n.onset, 0.25))
        elif role == "drums" and n.pitch in (
            49,
            57,
        ):  # crash cymbals mark phrase starts
            events.append((n.onset, 0.5))
    return events


def _count(notes: list[PerformanceNote]) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for n in notes:
        c[n.instrument] += 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


def _count_grids(notes: list[QuantizedNote]) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for n in notes:
        c[f"1/{n.grid}"] += 1
    return c


__all__ = ["run_pipeline", "PipelineResult", "PipelineError"]
