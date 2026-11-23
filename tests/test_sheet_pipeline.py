"""End-to-end test of the audio-to-sheet orchestrator with a stand-in model.

The real transcription model needs gated weights and minutes of GPU time; a
fake that emits a known song exercises every stage after it — cleanup,
meter, quantization, voices, chords, key, structure, Notation IR, MusicXML,
and (when MuseScore is installed) the PDF render.
"""

import json
import wave
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from muscriptor.events import NoteEndEvent, NoteStartEvent
from muscriptor.sheet.config import PipelineConfig
from muscriptor.sheet.pipeline import run_pipeline
from muscriptor.utils.beats import BeatGrid
from muscriptor.utils.sheets import sheets_are_available

BPM = 100
BEAT = 60 / BPM
# I – vi – IV – V in C, one chord per bar, 16 bars.
PROGRESSION = [
    (48, (60, 64, 67)),
    (45, (57, 60, 64)),
    (41, (57, 60, 65)),
    (43, (55, 59, 62)),
] * 4
MELODY = [72, 74, 76, 77, 79, 77, 76, 74]


class FakeModel:
    def __init__(self):
        self.calls = []

    def _song(self):
        notes = []
        for bar, (bass, chord) in enumerate(PROGRESSION):
            t0 = bar * 4 * BEAT + 0.5
            notes.append((bass - 12, t0, t0 + 4 * BEAT - 0.05, "electric_bass"))
            for beat in range(4):
                for p in chord:
                    notes.append(
                        (
                            p,
                            t0 + beat * BEAT + 0.003 * p % 0.01,
                            t0 + (beat + 0.9) * BEAT,
                            "acoustic_piano",
                        )
                    )
                notes.append(
                    (
                        36 if beat % 2 == 0 else 38,
                        t0 + beat * BEAT,
                        t0 + beat * BEAT + 0.05,
                        "drums",
                    )
                )
                notes.append(
                    (
                        42,
                        t0 + (beat + 0.5) * BEAT,
                        t0 + (beat + 0.5) * BEAT + 0.05,
                        "drums",
                    )
                )
            for k in range(2):
                p = MELODY[(bar * 2 + k) % len(MELODY)]
                notes.append((p, t0 + 2 * k * BEAT, t0 + (2 * k + 1.8) * BEAT, "voice"))
        return sorted(notes, key=lambda n: n[1])

    def transcribe(self, audio, instruments=None, beam_size=1, **kwargs):
        self.calls.append((str(audio), instruments))
        for i, (pitch, on, off, inst) in enumerate(self._song()):
            start = NoteStartEvent(pitch=pitch, start_time=on, index=i, instrument=inst)
            yield start
            yield NoteEndEvent(end_time=off, start_event=start)

    def detect_beat_grid_for(self, audio, mode="best-effort"):
        beats = 0.5 + np.arange(16 * 4 + 1) * BEAT
        return BeatGrid(
            bpm=BPM, beats_per_bar=4, first_downbeat=0.5, beats=beats, steady=True
        )


@pytest.fixture
def audio_file(tmp_path):
    path = tmp_path / "song.wav"
    sr = 16000
    n = int(sr * (0.5 + 16 * 4 * BEAT + 1))
    t = np.arange(n) / sr
    data = (0.1 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())
    return path


def _config(pdf: bool) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.separation.method = "none"
    cfg.transcription.stems = []
    cfg.harmony.audio_chords = False
    cfg.render.pdf = pdf
    cfg.render.part_pdfs = False
    cfg.resume = False
    return cfg


def test_pipeline_writes_every_artifact(tmp_path, audio_file):
    out = tmp_path / "out"
    result = run_pipeline(audio_file, _config(pdf=False), out, model=FakeModel())

    for name in (
        "raw.mid",
        "reconstructed.mid",
        "chords.json",
        "notation.json",
        "score.musicxml",
        "report.json",
        "pipeline.log",
    ):
        assert (out / name).is_file(), name
    assert (out / "transcription" / "mix.json").is_file()
    assert (out / "transcription" / "cleaned.json").is_file()

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    statuses = {s["stage"]: s["status"] for s in report["stages"]}
    assert statuses["musicxml"] == "ok"

    # Parts: voice on top, then piano, bass, drums — score order.
    names = [p.name for p in result.score.parts]
    assert names == ["Voice", "Piano", "Electric Bass", "Drum Set"]

    # The progression comes back as chord symbols, one per bar.
    chords = json.loads((out / "chords.json").read_text(encoding="utf-8"))
    symbols = [c["symbol"] for c in chords["chords"]]
    assert symbols[:4] == ["C", "Am", "F", "G"]
    assert chords["key"] == "C major"

    root = ET.parse(out / "score.musicxml").getroot()
    measures = root.find("part").findall("measure")
    assert len(measures) == 16
    assert root.find("part/measure/attributes/time/beats").text == "4"
    assert root.find(
        "part/measure/direction/direction-type/metronome/per-minute"
    ).text == str(BPM)


def test_pipeline_resume_skips_transcription(tmp_path, audio_file):
    out = tmp_path / "out"
    cfg = _config(pdf=False)
    model = FakeModel()
    run_pipeline(audio_file, cfg, out, model=model)
    cfg.resume = True
    model2 = FakeModel()
    run_pipeline(audio_file, cfg, out, model=model2)
    assert model.calls and not model2.calls


@pytest.mark.skipif(not sheets_are_available(), reason="MuseScore 4 not installed")
def test_pipeline_renders_pdf_with_musescore(tmp_path, audio_file):
    out = tmp_path / "out"
    result = run_pipeline(audio_file, _config(pdf=True), out, model=FakeModel())
    assert (out / "score.pdf").stat().st_size > 1000
    assert (out / "score.mscz").is_file()
    assert result.artifacts["pdf"].endswith("score.pdf")
