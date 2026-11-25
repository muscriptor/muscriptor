"""Run `python-audio-separator` in its own Python environment.

This file is executed as a script by the interpreter configured as
``separation.python`` (``$MUSCRIPTOR_SEPARATOR_PYTHON``), never imported by
MuScriptor: audio-separator pins dependencies (``rotary-embedding-torch``)
that cannot share an environment with ``beat_this``. It only imports the
standard library and ``audio_separator``.

Usage: python -I _separator_driver.py <request.json | '<json request>'>
Request: {"input": path, "output_dir": path, "model": filename,
          "model_dir": path | null, "batch_size": int, "overlap": int,
          "segment_size": int, "autocast": bool}
Prints one line ``MUSCRIPTOR_STEMS <json>`` mapping lower-case stem names
(as the model names them: "vocals", "piano", "other", ...) to WAV paths.
"""

import json
import re
import sys
from pathlib import Path


def main() -> None:
    arg = sys.argv[1]
    if not arg.lstrip().startswith("{"):
        arg = Path(arg).read_text(encoding="utf-8")
    req = json.loads(arg)
    from audio_separator.separator import Separator

    out_dir = Path(req["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "output_dir": str(out_dir),
        "output_format": "WAV",
        "sample_rate": 44100,
        "use_autocast": bool(req.get("autocast", False)),
        "mdxc_params": {
            "segment_size": int(req.get("segment_size", 256)),
            "override_model_segment_size": False,
            "batch_size": int(req.get("batch_size", 1)),
            "overlap": int(req.get("overlap", 8)),
            "pitch_shift": 0,
        },
    }
    if req.get("model_dir"):
        kwargs["model_file_dir"] = req["model_dir"]
    sep = Separator(**kwargs)
    sep.load_model(model_filename=req["model"])
    files = sep.separate(req["input"])
    stems = {}
    for f in files:
        path = Path(f)
        if not path.is_absolute():
            path = out_dir / path
        # "<input>_(Piano)_<model>.wav"
        m = re.search(r"_\(([^)]+)\)", path.name)
        name = (m.group(1) if m else path.stem).strip().lower().replace(" ", "_")
        stems[name] = str(path)
    print("MUSCRIPTOR_STEMS " + json.dumps(stems), flush=True)


if __name__ == "__main__":
    main()
