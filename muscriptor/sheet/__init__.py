"""Audio → readable sheet music.

The transcription model (:mod:`muscriptor.transcription_model`) answers "which
notes sounded, when, on what instrument". This package answers the separate
question of how a musician should *read* them:

    audio
      ├─ separation      (stems, optional)          separation.py
      ├─ transcription   (MuScriptor, mix + stems)  transcription.py
      ├─ beats / meter   (beat_this)                rhythm.py
      └─ audio chords    (lv-chordia)               harmony.py
    raw performance notes  ─────────────────────── ir.PerformanceNote
      → cleanup          (ghosts, duplicates, octave errors, monophony)  cleanup.py
      → quantization     (per-beat adaptive grid, swing)                 rhythm.py
      → part compilers   (staves, voices, drum mapping)                  voices.py
      → harmony          (key, chord symbols)                            harmony.py
      → structure        (sections → rehearsal marks)                    structure.py
    Notation IR  ────────────────────────────────── ir.Score
      → MusicXML                                                         musicxml.py
      → MuseScore → PDF                                                  render.py

`pipeline.run_pipeline` wires the stages together and writes every
intermediate artifact so each stage can be inspected on its own.
"""
