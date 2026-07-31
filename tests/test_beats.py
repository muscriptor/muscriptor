"""Tests for muscriptor/utils/beats.py.

All synthetic: the maths is exercised without beat_this or a checkpoint, since
only detect_grid touches the model.
"""

import numpy as np

from muscriptor.utils.beats import (
    BAR_OFFSET_MARKER,
    MAX_TEMPO_RESIDUAL,
    BeatGrid,
    beat_positions,
    fit_tempo,
    infer_beats_per_bar,
    read_bar_offset,
)

# beat_this reports beats on a 50 Hz frame grid.
FRAME_SECONDS = 0.02


def _beats(bpm=120.0, n=64, start=0.0, drift=0.0):
    """Beat times at `bpm`, optionally with a linear tempo ramp of `drift`."""
    t = start + np.arange(n) * (60.0 / bpm)
    if drift:
        span = t[-1] - t[0]
        t = t[0] + (t - t[0]) * (1 + drift * (t - t[0]) / span)
    return t


def _quantised(t):
    """Beat times as a tracker reports them, rounded to its frame grid."""
    return np.round(np.asarray(t) / FRAME_SECONDS) * FRAME_SECONDS


def _dropped(t, share, seed=11):
    """`t` with `share` of its beats missing, as a tracker that loses beats."""
    keep = np.random.default_rng(seed).random(len(t)) > share
    keep[0] = keep[-1] = True
    return np.asarray(t)[keep]


def test_fit_tempo_recovers_tempo():
    bpm, residual = fit_tempo(_beats(103.437))
    assert abs(bpm - 103.437) / 103.437 < 1e-3
    assert residual < 1e-6


def test_tempo_residual_gate_rejects_drifting_beats():
    """A 15% tempo ramp must fail the constant-tempo gate; steady beats pass."""
    for drift, should_pass in ((0.0, True), (0.15, False)):
        bpm, residual = fit_tempo(_beats(96.0, drift=drift))
        passes = residual < MAX_TEMPO_RESIDUAL * (60.0 / bpm)
        assert passes is should_pass


def test_fit_tempo_survives_missing_beats():
    """Dropped beats must not drag the tempo down with them.

    Counting position by the beat length around each interval keeps the tempo on
    the beats that are there; fitting against array position credits every beat
    after a gap to an earlier place and tilts the line.
    """
    for share in (0.02, 0.05, 0.10):
        beats = _quantised(_dropped(_beats(118.6, n=200), share))
        bpm, _ = fit_tempo(beats)
        assert abs(bpm - 118.6) / 118.6 < 0.005, f"{share:.0%} missing gave {bpm:.2f}"


def test_fit_tempo_stays_gated_when_too_many_beats_are_missing():
    """Heavier loss is not recovered, and must not be reported as a steady tempo.

    Once dropouts run together they move the local beat length with them, so the
    count degrades. What has to hold is that the residual degrades with it: the
    tempo comes out wrong but rejected, never wrong and kept.
    """
    beats = _quantised(_dropped(_beats(118.6, n=200), 0.4))
    bpm, residual = fit_tempo(beats)
    assert abs(bpm - 118.6) / 118.6 > 0.02
    assert residual > MAX_TEMPO_RESIDUAL * (60.0 / bpm)


def test_fit_tempo_beats_frame_quantisation():
    """The fit resolves tempos the median inter-beat interval cannot reach.

    A 20 ms frame grid puts every interval on a whole number of frames, so
    60/median can only land on 3000/k — for 118.6 BPM, on 120. Averaging more
    beats does not help, since every interval rounds the same way.
    """
    beats = _quantised(_beats(118.6, n=200))
    assert 60.0 / float(np.median(np.diff(beats))) == 120.0
    bpm, residual = fit_tempo(beats)
    assert abs(bpm - 118.6) < 0.01
    # The residual floors at the quantisation itself: 20 ms uniform is 20/sqrt(12).
    assert residual < 2 * FRAME_SECONDS / np.sqrt(12)


def test_swung_beats_are_not_read_as_missing_beats():
    """Beats that alternate long and short hold the positions they have.

    Every interval sits next to intervals just as uneven, so none stands out as a
    dropout. Counting the long ones as gaps would halve the tempo of every swung
    take; the unevenness belongs in the residual, which is what rejects it.
    """
    beat = 60.0 / 118.6
    intervals = [beat * (1.24 if i % 2 else 0.76) for i in range(199)]
    beats = _quantised(np.concatenate(([0.0], np.cumsum(intervals))))

    positions = beat_positions(beats)
    counted = np.arange(len(beats)) if positions is None else positions
    assert np.array_equal(counted, np.arange(len(beats)))

    bpm, residual = fit_tempo(beats)
    assert residual > MAX_TEMPO_RESIDUAL * (60.0 / bpm)


def test_clustered_loss_is_counted_even_when_most_beats_are_gone():
    """Three beats found, five missed, repeating: 62% gone and still recovered.

    Each surviving run sets a local beat length that the six-beat gaps stand out
    against, which is what judging gaps locally buys over the whole take's median.
    """
    beats = _quantised(_beats(118.6, n=200))
    clustered = np.array(
        [beat for index, beat in enumerate(beats) if index % 8 in (0, 1, 2)]
    )
    bpm, _ = fit_tempo(clustered)
    assert abs(bpm - 118.6) < 0.05


def test_a_doubled_tempo_keeps_failing_the_constant_tempo_gate():
    """A take that changes tempo must keep being rejected, not counted as gaps.

    Half-speed beats and every-other-beat-missing are the same timings, so sizing
    a gap against the whole take would let a doubling fit one line perfectly and
    write a MIDI tempo the music never holds. This is the case that makes the
    local baseline necessary rather than merely tidier.
    """
    slow = [i * 60.0 / 80.0 for i in range(40)]
    fast = [slow[-1] + (i + 1) * 60.0 / 160.0 for i in range(40)]
    beats = _quantised(slow + fast)
    bpm, residual = fit_tempo(beats)
    assert residual > MAX_TEMPO_RESIDUAL * (60.0 / bpm)


def test_a_half_speed_passage_stays_visible_as_a_tempo_change():
    """A passage that slows down mid-take is a tempo change, not missing beats.

    It has to outlast the window the local beat length is measured over, which is
    what separates a tempo change from a momentary hesitation.
    """
    beats = [i * 60.0 / 160.0 for i in range(60)]
    for _ in range(20):
        beats.append(beats[-1] + 60.0 / 80.0)
    for _ in range(60):
        beats.append(beats[-1] + 60.0 / 160.0)

    bpm, residual = fit_tempo(_quantised(beats))
    assert residual > MAX_TEMPO_RESIDUAL * (60.0 / bpm)


def test_infer_beats_per_bar_unanimous():
    beats = _beats(103.437, n=200)
    downbeats = beats[::4]
    assert infer_beats_per_bar(beats, downbeats) == 4


def test_infer_beats_per_bar_rejects_inconsistent_bars():
    """The Tears In The Typing Pool case: 2 beats/bar in only ~64% of bars."""
    beat = 0.68
    spacings = [2 * beat] * 50 + [3 * beat] * 24 + [4 * beat] * 5
    downbeats = np.concatenate([[0.0], np.cumsum(spacings)])
    beats = np.arange(0, downbeats[-1] + beat, beat)
    assert infer_beats_per_bar(beats, downbeats) is None


def test_bar_offset_puts_a_bar_line_on_the_first_downbeat():
    grid = BeatGrid(bpm=103.437, beats_per_bar=4, first_downbeat=1.560)
    offset = grid.bar_offset()
    bar = grid.bar_seconds
    assert 0.0 <= offset < bar
    # After the shift the first downbeat sits on a bar boundary.
    shifted = grid.first_downbeat + offset
    assert abs(shifted % bar) < 1e-9 or abs(shifted % bar - bar) < 1e-9


def test_bar_offset_is_forward_only_for_any_downbeat():
    grid = BeatGrid(bpm=90.0, beats_per_bar=3, first_downbeat=0.0)
    for first in np.linspace(0.0, 10.0, 51):
        grid.first_downbeat = float(first)
        assert 0.0 <= grid.bar_offset() < grid.bar_seconds


def test_bar_offset_is_zero_without_a_meter():
    grid = BeatGrid(bpm=120.0, beats_per_bar=None, first_downbeat=3.7)
    assert grid.bar_seconds is None
    assert grid.bar_offset() == 0.0


class _FakeMidi:
    def __init__(self, texts):
        self.tracks = [[_FakeMarker(t) for t in texts]]


class _FakeMarker:
    type = "marker"

    def __init__(self, text):
        self.text = text


def test_read_bar_offset():
    assert read_bar_offset(_FakeMidi([f"{BAR_OFFSET_MARKER}0.7945"])) == 0.7945
    assert read_bar_offset(_FakeMidi([])) == 0.0
    assert read_bar_offset(_FakeMidi(["some other marker"])) == 0.0
    assert read_bar_offset(_FakeMidi([f"{BAR_OFFSET_MARKER}nonsense"])) == 0.0
