"""Tests for muscriptor/utils/sheets.py (--format sheets).

MuseScore is never invoked here: the subprocess layer is mocked, so these run
on machines that don't have it installed.
"""

import subprocess
import xml.etree.ElementTree as ET

import pytest

import muscriptor.utils.sheets as sheets


def _mscx(parts):
    """A minimal MuseScore project. `parts` is a list of (instrument id, strings)."""
    score = ET.Element("Score")
    for instrument_id, strings in parts:
        part = ET.SubElement(score, "Part")
        staff = ET.SubElement(part, "Staff")
        ET.SubElement(staff, "eid").text = "abc123"
        staff_type = ET.SubElement(staff, "StaffType", group="pitched")
        ET.SubElement(staff_type, "name").text = "stdNormal"
        instrument = ET.SubElement(part, "Instrument", id=instrument_id)
        ET.SubElement(instrument, "longName").text = instrument_id
        if strings:
            string_data = ET.SubElement(instrument, "StringData")
            for pitch in range(strings):
                ET.SubElement(string_data, "string").text = str(40 + pitch)
    for number in range(1, len(parts) + 1):
        block = ET.SubElement(score, "Staff", id=str(number))
        measure = ET.SubElement(block, "Measure")
        ET.SubElement(measure, "Tempo")
        ET.SubElement(measure, "voice")
    root = ET.Element("museScore", version="4.60")
    root.append(score)
    return ET.ElementTree(root)


def _write(tmp_path, parts):
    path = tmp_path / "score.mscx"
    _mscx(parts).write(path, encoding="UTF-8", xml_declaration=True)
    return path


def _parts(path):
    return ET.parse(path).getroot().find("Score").findall("Part")


def _blocks(path):
    return ET.parse(path).getroot().find("Score").findall("Staff")


def test_fretted_parts_finds_guitars_and_basses(tmp_path):
    path = _write(
        tmp_path, [("drumset", 0), ("electric-guitar", 6), ("electric-bass", 4)]
    )
    assert sheets.fretted_parts(path) == [1, 2]


def test_fretted_parts_ignores_string_counts_without_a_preset(tmp_path):
    path = _write(tmp_path, [("some-3-string-thing", 3), ("alto", 0)])
    assert sheets.fretted_parts(path) == []


def test_convert_to_tab_uses_the_preset_matching_the_string_count(tmp_path):
    path = _write(tmp_path, [("electric-guitar", 6), ("electric-bass", 4)])
    assert sheets.convert_to_tab_staves(path) == [0, 1]
    presets = [p.findtext("Staff/StaffType/name") for p in _parts(path)]
    assert presets == ["tab6StrCommon", "tab4StrCommon"]


def test_converted_staff_spells_out_its_line_count(tmp_path):
    """MuseScore's reader treats <name> as a label, so <lines> has to be written:
    without it a 6-string guitar gets the default 5-line staff."""
    path = _write(tmp_path, [("electric-guitar", 6), ("electric-bass", 4)])
    sheets.convert_to_tab_staves(path)
    lines = [p.findtext("Staff/StaffType/lines") for p in _parts(path)]
    assert lines == ["6", "4"]


def test_converted_staff_is_in_the_tablature_group(tmp_path):
    path = _write(tmp_path, [("electric-guitar", 6)])
    sheets.convert_to_tab_staves(path)
    assert _parts(path)[0].find("Staff/StaffType").get("group") == "tablature"


def test_convert_to_tab_leaves_unfretted_parts_alone(tmp_path):
    path = _write(tmp_path, [("drumset", 0), ("electric-guitar", 6)])
    assert sheets.convert_to_tab_staves(path) == [1]
    assert _parts(path)[0].find("Staff/StaffType").get("group") == "pitched"
    assert _parts(path)[0].findtext("Staff/StaffType/name") == "stdNormal"


def test_convert_to_tab_adds_no_staves_and_renumbers_nothing(tmp_path):
    """The tab score replaces staves rather than adding them: the notation comes
    from the separate, untouched score."""
    path = _write(tmp_path, [("electric-guitar", 6), ("drumset", 0)])
    sheets.convert_to_tab_staves(path)
    assert [len(p.findall("Staff")) for p in _parts(path)] == [1, 1]
    assert [b.get("id") for b in _blocks(path)] == ["1", "2"]


# --- output directory -------------------------------------------------------


def test_prepare_output_dir_accepts_a_missing_path(tmp_path):
    sheets.prepare_output_dir(tmp_path / "nope")  # does not raise


def test_prepare_output_dir_accepts_an_empty_directory(tmp_path):
    (tmp_path / "empty").mkdir()
    sheets.prepare_output_dir(tmp_path / "empty")


def test_prepare_output_dir_rejects_a_non_empty_directory(tmp_path):
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "x.txt").write_text("x")
    with pytest.raises(ValueError, match="not empty"):
        sheets.prepare_output_dir(tmp_path / "full")


def test_prepare_output_dir_rejects_a_file(tmp_path):
    (tmp_path / "file").write_text("x")
    with pytest.raises(ValueError, match="not a directory"):
        sheets.prepare_output_dir(tmp_path / "file")


# --- finding MuseScore ------------------------------------------------------


def _fake_versions(monkeypatch, versions):
    """Make musescore_version answer from `versions`, a {path: version} map."""
    monkeypatch.setattr(sheets, "musescore_version", lambda b: versions.get(b))


def test_find_musescore_skips_versions_that_are_too_old(monkeypatch):
    monkeypatch.delenv(sheets.MUSESCORE_ENV_VAR, raising=False)
    monkeypatch.setattr(sheets, "_candidates", lambda: ["/old", "/new"])
    _fake_versions(monkeypatch, {"/old": (3, 2, 3), "/new": (4, 7, 4)})
    assert sheets.find_musescore() == "/new"


def test_find_musescore_names_the_old_install_it_rejected(monkeypatch):
    monkeypatch.delenv(sheets.MUSESCORE_ENV_VAR, raising=False)
    monkeypatch.setattr(sheets, "_candidates", lambda: ["/usr/bin/mscore3"])
    _fake_versions(monkeypatch, {"/usr/bin/mscore3": (3, 2, 3)})
    with pytest.raises(sheets.MuseScoreNotFoundError) as e:
        sheets.find_musescore()
    assert "/usr/bin/mscore3" in str(e.value)
    assert "3.2.3" in str(e.value)


def test_find_musescore_reports_nothing_installed(monkeypatch):
    monkeypatch.delenv(sheets.MUSESCORE_ENV_VAR, raising=False)
    monkeypatch.setattr(sheets, "_candidates", lambda: [])
    with pytest.raises(sheets.MuseScoreNotFoundError, match="was not found"):
        sheets.find_musescore()


def test_env_var_pointing_at_a_missing_file_raises(monkeypatch, tmp_path):
    monkeypatch.setenv(sheets.MUSESCORE_ENV_VAR, str(tmp_path / "absent"))
    with pytest.raises(sheets.MuseScoreNotFoundError, match="not a file"):
        sheets.find_musescore()


def test_env_var_pointing_at_an_old_musescore_raises(monkeypatch, tmp_path):
    binary = tmp_path / "mscore3"
    binary.write_text("")
    monkeypatch.setenv(sheets.MUSESCORE_ENV_VAR, str(binary))
    _fake_versions(monkeypatch, {str(binary): (3, 2, 3)})
    with pytest.raises(sheets.MuseScoreNotFoundError, match="3.2.3"):
        sheets.find_musescore()


def test_env_var_wins_over_path(monkeypatch, tmp_path):
    binary = tmp_path / "mscore"
    binary.write_text("")
    monkeypatch.setenv(sheets.MUSESCORE_ENV_VAR, str(binary))
    _fake_versions(monkeypatch, {str(binary): (4, 7, 4)})
    monkeypatch.setattr(sheets, "_candidates", lambda: ["/should/not/be/used"])
    assert sheets.find_musescore() == str(binary)


# --- write_sheets -----------------------------------------------------------


class _FakeMuseScore:
    """Stands in for the MuseScore binary, writing plausible files for each call."""

    def __init__(self, parts=("Electric Guitar", "Drum Kit"), skip=()):
        self.parts = list(parts)
        self.skip = set(skip)  # extensions to *not* write, to simulate failures
        self.calls = []
        # Contents of every -M file, read here because the scratch directory
        # holding it is gone by the time write_sheets returns.
        self.options = []

    def __call__(self, binary, args):
        self.calls.append(args)
        if "-M" in args:
            self.options.append(
                __import__("pathlib").Path(args[args.index("-M") + 1]).read_text()
            )
        stdout = ""
        if "--score-parts-pdf" in args:
            import base64
            import json

            stdout = json.dumps(
                {
                    "parts": self.parts,
                    "partsBin": [
                        base64.b64encode(b"%PDF-" + name.encode()).decode()
                        for name in self.parts
                    ],
                }
            )
        elif "-o" in args:
            out = __import__("pathlib").Path(args[args.index("-o") + 1])
            if out.suffix.lstrip(".") not in self.skip:
                if out.suffix == ".mscx":
                    _mscx([("electric-guitar", 6), ("drumset", 0)]).write(out)
                else:
                    out.write_bytes(b"%PDF-fake")
        return subprocess.CompletedProcess(args, 0, stdout, "")


def test_write_sheets_writes_exactly_the_expected_files(tmp_path, monkeypatch):
    monkeypatch.setattr(sheets, "_run", _FakeMuseScore())
    out = tmp_path / "sheets"
    written = sheets.write_sheets(b"MThd-fake", out, musescore="/fake/mscore")
    assert sorted(p.name for p in out.iterdir()) == [
        "01_electric_guitar.pdf",
        "01_electric_guitar_tab.pdf",
        "02_drum_kit.pdf",
        "full_score.pdf",
        "score.mid",
        "score.musicxml",
    ]
    assert sorted(p.name for p in written) == sorted(p.name for p in out.iterdir())


def test_write_sheets_writes_the_midi_it_was_given(tmp_path, monkeypatch):
    monkeypatch.setattr(sheets, "_run", _FakeMuseScore())
    out = tmp_path / "sheets"
    sheets.write_sheets(b"MThd-original", out, musescore="/fake/mscore")
    assert (out / "score.mid").read_bytes() == b"MThd-original"


def test_write_sheets_creates_the_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(sheets, "_run", _FakeMuseScore())
    out = tmp_path / "deep" / "sheets"
    sheets.write_sheets(b"MThd-fake", out, musescore="/fake/mscore")
    assert out.is_dir()


def test_write_sheets_passes_the_import_options(tmp_path, monkeypatch):
    """HumanPerformance is what keeps the notation readable; if the -M file ever
    stops being passed, the score silently fills with 128th notes."""
    fake = _FakeMuseScore()
    monkeypatch.setattr(sheets, "_run", fake)
    sheets.write_sheets(b"MThd-fake", tmp_path / "sheets", musescore="/fake/mscore")
    assert "-M" in fake.calls[0]


def test_the_triplet_search_follows_the_quantized_flag(tmp_path, monkeypatch):
    """Snapping onto triplets is pointless if MuseScore then engraves them as ties,
    and searching for them in unquantized input reads the jitter as triplets."""
    fake = _FakeMuseScore()
    monkeypatch.setattr(sheets, "_run", fake)
    for quantized in (True, False):
        sheets.write_sheets(
            b"MThd-fake",
            tmp_path / str(quantized),
            musescore="/fake/mscore",
            quantized=quantized,
        )
    assert "<Triplets>true</Triplets>" in fake.options[0]
    assert "<Triplets>false</Triplets>" in fake.options[1]


def test_write_sheets_reports_a_pdf_that_never_appeared(tmp_path, monkeypatch):
    """MuseScore exits 0 while writing nothing, so missing output is the signal."""
    monkeypatch.setattr(sheets, "_run", _FakeMuseScore(skip={"pdf"}))
    with pytest.raises(sheets.MuseScoreError, match="render the full score"):
        sheets.write_sheets(b"MThd-fake", tmp_path / "sheets", musescore="/fake/mscore")


def test_write_sheets_reports_a_failed_midi_import(tmp_path, monkeypatch):
    monkeypatch.setattr(sheets, "_run", _FakeMuseScore(skip={"mscx"}))
    with pytest.raises(sheets.MuseScoreError, match="import the MIDI file"):
        sheets.write_sheets(b"MThd-fake", tmp_path / "sheets", musescore="/fake/mscore")


def test_write_sheets_reports_unparseable_parts_output(tmp_path, monkeypatch):
    fake = _FakeMuseScore()
    fake.parts = []
    monkeypatch.setattr(sheets, "_run", fake)
    with pytest.raises(sheets.MuseScoreError, match="per-instrument PDFs"):
        sheets.write_sheets(b"MThd-fake", tmp_path / "sheets", musescore="/fake/mscore")


@pytest.mark.parametrize(
    "name,expected",
    [
        (
            # MuseScore's instrument name adds nothing to the track name here.
            "Electric Guitar, distorted electric guitar",
            "distorted_electric_guitar",
        ),
        ("Electric Guitar, clean electric guitar", "clean_electric_guitar"),
        # Same name twice over: keep one copy, not two.
        ("Electric Bass, electric bass", "electric_bass"),
        # Neither says the other, so neither is dropped.
        ("Drum Kit, drums", "drum_kit_drums"),
        ("Alto, voice", "alto_voice"),
        ("Drum Kit", "drum_kit"),
        ("!!!", "part"),
        ("", "part"),
    ],
)
def test_slug(name, expected):
    assert sheets._slug(name) == expected


def test_slug_keeps_the_more_specific_segment(tmp_path, monkeypatch):
    """The dedup must not shorten a name into a different instrument."""
    assert sheets._slug("Guitar, acoustic guitar") == "acoustic_guitar"
    assert sheets._slug("Acoustic Guitar, guitar") == "acoustic_guitar"


def test_part_pdfs_are_named_without_the_repetition(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sheets, "_run", _FakeMuseScore(parts=["Electric Guitar, clean electric guitar"])
    )
    out = tmp_path / "sheets"
    sheets.write_sheets(b"MThd-fake", out, musescore="/fake/mscore")
    assert (out / "01_clean_electric_guitar.pdf").is_file()


def _captured_env(monkeypatch, system):
    """Run `_run` with subprocess mocked out, returning the env it passed."""
    captured = {}

    def fake_run(argv, **kwargs):
        captured.update(kwargs["env"])
        return subprocess.CompletedProcess(argv, 0, "MuseScore4 4.7.4", "")

    monkeypatch.setattr(sheets.platform, "system", lambda: system)
    monkeypatch.setattr(sheets.subprocess, "run", fake_run)
    sheets._run("/fake/mscore", ["--version"])
    return captured


def test_run_forces_the_offscreen_platform_on_linux(monkeypatch):
    """Both variables are required on a headless box.

    MuseScore 4 sets Qt's platform from its own MU_QT_QPA_PLATFORM and ignores
    QT_QPA_PLATFORM; Qt still needs QT_QPA_PLATFORM itself. With only one of
    them set, MuseScore tries to open an X11 display and aborts with "no Qt
    platform plugin could be initialized" — so a server or container with no
    display renders nothing at all.
    """
    env = _captured_env(monkeypatch, "Linux")
    assert env["QT_QPA_PLATFORM"] == "offscreen"
    assert env["MU_QT_QPA_PLATFORM"] == "offscreen"


def test_run_leaves_the_platform_alone_off_linux(monkeypatch):
    """macOS/Windows have a working native platform plugin; forcing offscreen
    there would only take away a display MuseScore can legitimately use."""
    env = _captured_env(monkeypatch, "Darwin")
    assert "QT_QPA_PLATFORM" not in env
    assert "MU_QT_QPA_PLATFORM" not in env


def test_run_does_not_override_an_explicit_platform(monkeypatch):
    """Someone who set the variables themselves (an Xvfb display, say) keeps
    them: both are setdefault, not assignment."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "xcb")
    monkeypatch.setenv("MU_QT_QPA_PLATFORM", "xcb")
    env = _captured_env(monkeypatch, "Linux")
    assert env["QT_QPA_PLATFORM"] == "xcb"
    assert env["MU_QT_QPA_PLATFORM"] == "xcb"
