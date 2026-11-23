"""Song sections from the transcription: boundaries and repeat-aware letters.

A lightweight stand-in for an audio structure analyzer (the research document
recommends All-In-One, whose NATTEN dependency does not install on Windows /
recent PyTorch). Each bar is described by its harmony (chord tones), pitch
content, and how busy each instrument role is; a checkerboard kernel on the
bar-to-bar self-similarity matrix (Foote novelty) finds where the texture
changes, boundaries are snapped to 4-bar phrases, and sections that resemble
each other get the same letter (A, B, A, B, C ...).

Labels are letters on purpose: telling a verse from a chorus needs lyrics or
a trained model, and a wrong "Chorus" is worse than an honest "B".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from muscriptor.sheet.instruments import spec_for
from muscriptor.sheet.ir import PerformanceNote

logger = logging.getLogger(__name__)

_ROLES = [
    "voice",
    "keyboard",
    "guitar",
    "bass",
    "drums",
    "string_section",
    "strings",
    "winds",
    "brass",
    "synth",
]


@dataclass
class Section:
    start_bar: int  # 0-based
    end_bar: int  # exclusive
    label: str


def bar_features(
    notes: list[PerformanceNote],
    bar_starts: np.ndarray,
    bar_chord_pcs: list[set[int]],
) -> np.ndarray:
    """One feature vector per bar (rows), each block normalised."""
    n_bars = len(bar_starts) - 1
    pcp = np.zeros((n_bars, 12))
    dens = np.zeros((n_bars, len(_ROLES)))
    for n in notes:
        b = int(np.searchsorted(bar_starts, n.onset, side="right")) - 1
        if not 0 <= b < n_bars:
            continue
        role = spec_for(n.instrument).role
        if role in _ROLES:
            dens[b, _ROLES.index(role)] += 1
        if not n.is_drum:
            pcp[b, n.pitch % 12] += min(n.duration, 2.0)
    chords = np.zeros((n_bars, 12))
    for b, pcs in enumerate(bar_chord_pcs[:n_bars]):
        chords[b, list(pcs)] = 1.0

    def norm(x):
        s = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(s > 0, s, 1)

    dens = np.log1p(dens)
    return np.hstack([norm(pcp), norm(chords), 1.5 * norm(dens)])


def novelty(features: np.ndarray, half: int = 4) -> np.ndarray:
    n = len(features)
    f = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-9)
    ssm = f @ f.T
    size = 2 * half
    sign = np.ones((size, size))
    sign[:half, half:] = -1
    sign[half:, :half] = -1
    g = np.exp(-((np.arange(size) - half + 0.5) ** 2) / (2 * (half / 2) ** 2))
    kernel = sign * np.outer(g, g)
    pad = np.pad(ssm, half, mode="edge")
    out = np.zeros(n)
    for i in range(n):
        out[i] = (pad[i : i + size, i : i + size] * kernel).sum()
    return out


def detect_sections(
    notes: list[PerformanceNote],
    bar_starts: np.ndarray,
    bar_chord_pcs: list[set[int]],
    first_bar: int = 0,
    min_len: int = 4,
    phrase: int = 4,
) -> list[Section]:
    """Sections over bars [0, n_bars), boundaries on phrase lines where possible."""
    n_bars = len(bar_starts) - 1
    if n_bars < 2 * min_len:
        return [Section(0, n_bars, "A")]
    feats = bar_features(notes, bar_starts, bar_chord_pcs)
    nov = novelty(feats)
    nov = (nov - nov.mean()) / (nov.std() + 1e-9)

    # Candidate boundaries: local maxima of novelty above the mean, preferring
    # phrase lines (multiples of `phrase` bars from the first played bar).
    candidates = []
    for i in range(min_len, n_bars - min_len + 1):
        bonus = (
            0.6
            if (i - first_bar) % phrase == 0
            else (0.2 if (i - first_bar) % 2 == 0 else -0.5)
        )
        score = nov[i] + bonus
        if score > 0.9 and nov[i] >= nov[max(0, i - 1)] - 0.3:
            candidates.append((score, i))
    chosen: list[int] = []
    for score, i in sorted(candidates, reverse=True):
        if all(abs(i - j) >= min_len for j in chosen):
            chosen.append(i)
    bounds = [0] + sorted(chosen) + [n_bars]

    # Label by similarity of mean features.
    sections: list[Section] = []
    protos: list[np.ndarray] = []
    labels: list[str] = []
    for a, b in zip(bounds, bounds[1:]):
        mean = feats[a:b].mean(axis=0)
        mean = mean / max(np.linalg.norm(mean), 1e-9)
        label = None
        for proto, lab in zip(protos, labels):
            if float(mean @ proto) >= 0.93:
                label = lab
                break
        if label is None:
            label = chr(ord("A") + min(len(protos), 25))
            protos.append(mean)
            labels.append(label)
        sections.append(Section(a, b, label))
    logger.info(
        "structure: %s",
        " ".join(f"{s.label}({s.start_bar + 1}-{s.end_bar})" for s in sections),
    )
    return sections
