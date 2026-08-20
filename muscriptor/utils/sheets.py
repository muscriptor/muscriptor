"""Engraved sheet music: MIDI → PDF / MusicXML, via the MuseScore CLI.

MuseScore is an external program, not a Python dependency: `--format sheets`
only works on machines that have it installed (see `find_musescore`). Everything
here shells out to it and then checks that the files it was asked for actually
appeared — `mscore` exits 0 when it writes nothing at all, so a plain returncode
check would report success on an empty directory.

MuseScore 4 or newer is required. MuseScore 3 writes a different project format
(no <StringData> on its instruments, so nothing says how a guitar is tuned), and
`convert_to_tab_staves` would quietly produce a score with no tablature in it.
"""

import base64
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

# Names MuseScore 3 and 4 install themselves under, plus the AppImage that the
# Linux downloads page hands out (which people usually leave in $HOME).
_BINARY_NAMES = (
    "mscore",
    "musescore",
    "mscore4portable",
    "MuseScore4",
    "musescore4",
    "mscore3",
    "musescore3",
)
_APP_LOCATIONS = (
    "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
    "/Applications/MuseScore 3.app/Contents/MacOS/mscore",
    "~/MuseScore.AppImage",
    "~/Applications/MuseScore.AppImage",
)
# Checked before anything else, for a MuseScore that isn't on PATH.
MUSESCORE_ENV_VAR = "MUSCRIPTOR_MUSESCORE"

# MuseScore 3 and older are rejected: see the module docstring.
MINIMUM_MAJOR_VERSION = 4

_INSTALL_HINT = (
    "Downloads for every platform: https://musescore.org/en/download\n"
    f"If it is installed somewhere unusual, set ${MUSESCORE_ENV_VAR} to it."
)

# MuseScore's MIDI import settings, passed with -M. HumanPerformance and
# QuantValue are MuseScore 4's own defaults, but spell them out.
# Element names and the QuantValue index are MuseScore's own; 2 == 1/16.
_IMPORT_OPTIONS = """<?xml version="1.0" encoding="UTF-8"?>
<MidiOptions>
  <QuantValue>2</QuantValue>
  <HumanPerformance>true</HumanPerformance>
  <Duplets>false</Duplets>
  <Triplets>{triplets}</Triplets>
  <Quadruplets>false</Quadruplets>
  <Quintuplets>false</Quintuplets>
  <Septuplets>false</Septuplets>
  <Nonuplets>false</Nonuplets>
  <SimplifyDurations>true</SimplifyDurations>
  <DottedNotes>true</DottedNotes>
</MidiOptions>
"""


def import_options(quantized: bool) -> str:
    """The -M import settings. Only search for triplets if we know the grid is accurate
    thanks to quantization.

    This choice wasn't validated very closely, maybe setting `true` always is ok too.
    """
    return _IMPORT_OPTIONS.format(triplets="true" if quantized else "false")


# Tablature staff presets, by string count. MuseScore ships tab4Str…tab9Str;
# anything outside that range (or with no strings at all) gets no tab staff.
TAB_PRESETS = {n: f"tab{n}StrCommon" for n in range(4, 10)}

# From MuseScore's own preset table: tab staves are spaced 1.5x a normal staff.
_TAB_LINE_DISTANCE = "1.5"

# Per MuseScore invocation. Generous next to the ~1.5s a song takes; it is
# here so a wedged subprocess fails the run instead of hanging it forever.
RUN_TIMEOUT_S = 120


class MuseScoreNotFoundError(RuntimeError):
    """No MuseScore executable could be located."""


class MuseScoreError(RuntimeError):
    """MuseScore ran but did not produce what it was asked for."""


def musescore_version(binary: str) -> tuple[int, ...] | None:
    """`binary`'s version as a tuple, or None if it doesn't answer like MuseScore.

    `mscore --version` prints e.g. "MuseScore4 4.7.4"; the AppImage also writes
    shared-library chatter to stderr, so both streams are searched.
    """
    try:
        proc = _run(binary, ["--version"])
    except OSError:
        return None
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", f"{proc.stdout}\n{proc.stderr}")
    if match is None:
        return None
    return tuple(int(g) for g in match.groups() if g is not None)


def _candidates() -> list[str]:
    """Every MuseScore-looking executable worth probing, best guess first."""
    found = []
    for name in _BINARY_NAMES:
        path = shutil.which(name)
        if path:
            found.append(path)
    for location in _APP_LOCATIONS:
        path = Path(location).expanduser()
        if path.is_file():
            found.append(str(path))
    return found


def find_musescore() -> str:
    """Path to a MuseScore 4+ executable.

    Checks $MUSCRIPTOR_MUSESCORE first, then PATH, then the places the macOS
    and Linux downloads put it, and returns the first one new enough to use.
    Raises MuseScoreNotFoundError — naming any too-old MuseScore it did find,
    since "not found" is a confusing thing to read with `mscore` on your PATH.
    """
    override = os.environ.get(MUSESCORE_ENV_VAR)
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise MuseScoreNotFoundError(
                f"{MUSESCORE_ENV_VAR} is set to {override!r}, which is not a file."
            )
        version = musescore_version(str(path))
        if version is None or version[0] < MINIMUM_MAJOR_VERSION:
            shown = ".".join(map(str, version)) if version else "an unknown version"
            raise MuseScoreNotFoundError(
                f"{MUSESCORE_ENV_VAR} points at MuseScore {shown}, but "
                f"--format sheets needs MuseScore {MINIMUM_MAJOR_VERSION} or newer."
            )
        return str(path)

    too_old: list[str] = []
    for candidate in _candidates():
        version = musescore_version(candidate)
        if version is None:
            continue
        if version[0] >= MINIMUM_MAJOR_VERSION:
            return candidate
        too_old.append(f"{candidate} (MuseScore {'.'.join(map(str, version))})")

    if too_old:
        raise MuseScoreNotFoundError(
            f"--format sheets needs MuseScore {MINIMUM_MAJOR_VERSION} or newer, "
            "but the only MuseScore installed is:\n  "
            + "\n  ".join(too_old)
            + "\nMuseScore 3 cannot produce the guitar and bass tablature.\n"
            + _INSTALL_HINT
        )
    raise MuseScoreNotFoundError(
        "MuseScore was not found. --format sheets engraves the score with "
        f"MuseScore {MINIMUM_MAJOR_VERSION}+, which has to be installed "
        "separately:\n" + _INSTALL_HINT
    )


def _run(binary: str, args: list[str]) -> subprocess.CompletedProcess:
    """MuseScore with `args`, headless, returning the finished process.

    The returncode is deliberately not checked here: MuseScore exits 0 for
    several failures that write no file (an unreadable -M path, a score with no
    parts), so callers verify their outputs instead.

    A timeout is enforced so a MuseScore that decides to wait for something
    fails the run rather than hanging it indefinitely.
    """
    env = dict(os.environ)
    if platform.system() == "Linux":
        # Without these MuseScore tries to open an X11 display and dies on a
        # headless box ("no Qt platform plugin could be initialized"); harmless
        # when a display is present. MuseScore 4 sets Qt's platform from its own
        # MU_QT_QPA_PLATFORM and ignores QT_QPA_PLATFORM, so the second line is
        # the one that matters on the server; the first still covers MuseScore 3
        # and anything else Qt-based in the chain. Passing `-platform offscreen`
        # instead does not work: MuseScore's command-line parser reads the value
        # as an input file and the conversion silently loses its real argument.
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        env.setdefault("MU_QT_QPA_PLATFORM", "offscreen")
    try:
        return subprocess.run(
            [binary, *args],
            capture_output=True,
            text=True,
            errors="replace",
            env=env,
            timeout=RUN_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args, 1, "", f"timed out after {RUN_TIMEOUT_S}s"
        )


def _fail(what: str, proc: subprocess.CompletedProcess) -> None:
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
    detail = ("\n  " + "\n  ".join(tail)) if tail else ""
    raise MuseScoreError(f"MuseScore failed to {what}.{detail}")


def _normalize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def _slug(name: str) -> str:
    """Filename stem for a part, with the repetition in its name dropped.

    MuseScore names a part after the instrument it matched, then appends the
    track name the MIDI carried — which for a transcription usually says the
    same thing twice: "Electric Guitar, clean electric guitar". A segment whose
    words all appear in another one adds nothing, so it goes and the more
    specific name stays. Segments that genuinely differ are both kept, since
    there is no telling which one the reader wants ("Drum Kit, drums").
    """
    segments = [s for s in (_normalize(part) for part in name.split(",")) if s]
    words = [set(segment.split("_")) for segment in segments]
    keep = [
        segment
        for i, segment in enumerate(segments)
        # A strict subset says strictly less; identical segments would each
        # rule the other out, so only the first of those survives.
        if not any(
            words[i] < other or (words[i] == other and j < i)
            for j, other in enumerate(words)
            if j != i
        )
    ]
    return "_".join(keep) or "part"


def _string_counts(mscx_path: Path) -> list[int]:
    """How many strings MuseScore gave each part's instrument, in part order.

    A part counts as fretted when its instrument has a <StringData> block whose
    string count matches a tab preset; that finds guitars and basses (and
    5-string basses, 7-string guitars, …) without hardcoding instrument names.
    """
    score = ET.parse(mscx_path).getroot().find("Score")
    if score is None:
        raise MuseScoreError(f"{mscx_path} has no <Score> element")
    counts = []
    for part in score.findall("Part"):
        instrument = part.find("Instrument")
        strings = (
            len(instrument.findall("StringData/string"))
            if instrument is not None
            else 0
        )
        counts.append(strings)
    return counts


def fretted_parts(mscx_path: Path) -> list[int]:
    """Indices of the parts in `mscx_path` that can be engraved as tablature."""
    return [i for i, n in enumerate(_string_counts(mscx_path)) if n in TAB_PRESETS]


def convert_to_tab_staves(mscx_path: Path) -> list[int]:
    """Retype every fretted part's staff as tablature, in place.

    Rewrites the staff rather than adding one: this score exists only to render
    the tab-only PDFs, and the notation comes from the untouched score it was
    copied from. Returns the indices of the parts converted.
    """
    tree = ET.parse(mscx_path)
    score = tree.getroot().find("Score")
    if score is None:
        raise MuseScoreError(f"{mscx_path} has no <Score> element")

    converted = []
    for index, part in enumerate(score.findall("Part")):
        instrument = part.find("Instrument")
        strings = (
            len(instrument.findall("StringData/string"))
            if instrument is not None
            else 0
        )
        preset = TAB_PRESETS.get(strings)
        staff = part.find("Staff")
        if preset is None or staff is None:
            continue
        _retype_as_tablature(staff, preset, strings)
        converted.append(index)

    tree.write(mscx_path, encoding="UTF-8", xml_declaration=True)
    return converted


def _retype_as_tablature(staff: ET.Element, preset: str, strings: int) -> None:
    """Turn a notation <Staff> into a `strings`-line tablature staff, in place."""
    staff_type = staff.find("StaffType")
    if staff_type is None:
        staff_type = ET.SubElement(staff, "StaffType")
    staff_type.set("group", "tablature")

    name = staff_type.find("name")
    if name is None:
        name = ET.SubElement(staff_type, "name")
    name.text = preset

    # <name> is only a label to MuseScore's reader (it does not look the preset
    # up), so the geometry has to be spelled out or the staff keeps the default
    # 5 lines — a 6-string guitar on 5 lines.
    for tag in ("lines", "lineDistance"):
        for existing in staff_type.findall(tag):
            staff_type.remove(existing)
    ET.SubElement(staff_type, "lines").text = str(strings)
    ET.SubElement(staff_type, "lineDistance").text = _TAB_LINE_DISTANCE


def write_sheets(
    midi_bytes: bytes,
    out_dir: Path,
    musescore: str | None = None,
    quantized: bool = False,
) -> list[Path]:
    """Engrave `midi_bytes` into `out_dir`, returning the files written.

    Writes the MIDI, a MusicXML score, one PDF of the full score,
    and one PDF of standard notation per instrument. Guitar and bass parts
    additionally get a tablature PDF of their own, rendered from a second copy
    of the score whose staves are retyped as tab. `out_dir` is created if it
    does not exist.

    `quantized` says whether the notes are already snapped to a beat grid (by
    `muscriptor.utils.midi.quantized_notes`), which is what the engraving wants:
    it decides the triplet search, and unquantized input engraves the timing
    jitter as tied 128th notes.
    """
    binary = musescore or find_musescore()
    out_dir.mkdir(parents=True, exist_ok=True)

    midi_path = out_dir / "score.mid"
    midi_path.write_bytes(midi_bytes)
    written = [midi_path]

    with tempfile.TemporaryDirectory(prefix="muscriptor-sheets-") as tmp:
        tmp_dir = Path(tmp)
        options = tmp_dir / "import.xml"
        options.write_text(import_options(quantized))

        # MuseScore's own project format, so the score can be copied and edited
        # (staves retyped as tab) before anything is rendered from it.
        mscx = tmp_dir / "score.mscx"
        proc = _run(binary, ["-M", str(options), "-o", str(mscx), str(midi_path)])
        if not mscx.is_file():
            _fail("import the MIDI file", proc)

        musicxml = out_dir / "score.musicxml"
        proc = _run(binary, ["-o", str(musicxml), str(mscx)])
        if not musicxml.is_file():
            _fail("write MusicXML", proc)
        written.append(musicxml)

        full_score = out_dir / "full_score.pdf"
        proc = _run(binary, ["-o", str(full_score), str(mscx)])
        if not full_score.is_file():
            _fail("render the full score", proc)
        written.append(full_score)

        written.extend(_write_part_pdfs(binary, mscx, out_dir))

        fretted = fretted_parts(mscx)
        if fretted:
            tab_mscx = tmp_dir / "tab.mscx"
            shutil.copyfile(mscx, tab_mscx)
            convert_to_tab_staves(tab_mscx)
            written.extend(
                _write_part_pdfs(
                    binary, tab_mscx, out_dir, suffix="_tab", only=set(fretted)
                )
            )

    return written


def _write_part_pdfs(
    binary: str,
    mscx: Path,
    out_dir: Path,
    suffix: str = "",
    only: set[int] | None = None,
) -> list[Path]:
    """One PDF per instrument, extracted from --score-parts-pdf.

    MuseScore only writes per-part files for scores that already carry
    generated parts, which a MIDI import does not; --score-parts-pdf generates
    them on the fly, but hands them back as base64 in a JSON blob on stdout
    rather than writing files.

    `only` keeps just those part indices (the tab pass wants the fretted ones),
    and `suffix` goes on the filename before the extension. Numbering follows
    the part's position in the score either way, so a guitar's notation and
    tablature PDFs sort together.
    """
    proc = _run(binary, ["--score-parts-pdf", str(mscx)])
    try:
        payload = json.loads(proc.stdout)
        names, blobs = payload["parts"], payload["partsBin"]
    except (json.JSONDecodeError, KeyError, TypeError):
        _fail("generate the per-instrument PDFs", proc)

    written = []
    for index, (name, blob) in enumerate(zip(names, blobs)):
        if only is not None and index not in only:
            continue
        path = out_dir / f"{index + 1:02d}_{_slug(name)}{suffix}.pdf"
        path.write_bytes(base64.b64decode(blob))
        written.append(path)
    if not written:
        _fail("generate the per-instrument PDFs", proc)
    return written


def prepare_output_dir(path: Path) -> None:
    """Check `path` can be used as a sheets output directory.

    Wants a path that does not exist yet, or an existing empty directory, so a
    run cannot scatter PDFs among unrelated files or quietly overwrite a
    previous score. Raises ValueError otherwise; the directory itself is created
    later, by write_sheets.
    """
    if not path.exists():
        return
    if not path.is_dir():
        raise ValueError(f"{path} exists and is not a directory")
    if any(path.iterdir()):
        raise ValueError(f"{path} is not empty")


def sheets_are_available() -> bool:
    """Whether MuseScore can be found, for callers that want to check first."""
    try:
        find_musescore()
    except MuseScoreNotFoundError:
        return False
    return True


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    written = write_sheets(Path(sys.argv[1]).read_bytes(), Path(sys.argv[2]))
    for path in written:
        print(path)
