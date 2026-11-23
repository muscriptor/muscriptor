"""MusicXML → PDF (and a MuseScore project) with the MuseScore 4 CLI.

Per the research notes, MuseScore's batch-job (`-j`) and part-export (`-P`)
flags have had export bugs across 4.x releases, so every file is converted
with a plain `-o` call: the full score, and each per-instrument MusicXML file
the pipeline wrote itself. MuseScore exits 0 on several failures, so success
is judged by the output file existing and being non-empty.

The executable comes from the config, else ``$MUSCRIPTOR_MUSESCORE``, else
PATH and the platforms' default install locations
(:func:`muscriptor.utils.sheets.find_musescore`).
"""

from __future__ import annotations

import logging
from pathlib import Path

from muscriptor.utils.sheets import (
    MuseScoreError,
    MuseScoreNotFoundError,
    _run,
    find_musescore,
    musescore_version,
)

logger = logging.getLogger(__name__)

__all__ = ["MuseScoreError", "MuseScoreNotFoundError", "resolve_musescore", "convert"]


def resolve_musescore(configured: str | None) -> str:
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise MuseScoreNotFoundError(
                f"configured MuseScore {configured!r} is not a file"
            )
        version = musescore_version(str(path))
        if version is None or version[0] < 4:
            raise MuseScoreNotFoundError(f"{configured} is not MuseScore 4 or newer")
        return str(path)
    return find_musescore()


def convert(binary: str, source: Path, target: Path) -> Path:
    """`mscore -o target source`, raising MuseScoreError unless `target` appears."""
    if target.exists():
        target.unlink()
    proc = _run(binary, ["-o", str(target), str(source)])
    if not target.is_file() or target.stat().st_size == 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        raise MuseScoreError(
            f"MuseScore could not convert {source.name} to {target.suffix}"
            + ("\n  " + "\n  ".join(tail) if tail else "")
        )
    logger.info(
        "render: %s → %s (%d KB)",
        source.name,
        target.name,
        target.stat().st_size // 1024,
    )
    return target
