"""Minimal conditioners for muscriptor inference.

Contains only the classes needed to run the transcription model:
- ConditioningAttributes, ConditionType, WavCondition
- MelSpectrogramConditioner (audio → mel → linear projection)
- ClassConditioner (class index → embedding)
- ConditioningProvider
- nullify_all_conditions (for CFG at inference)
"""

import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import NamedTuple, Any
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

import muscriptor.accelerator
from muscriptor.modules.mel_spectrogram import _MelSpectrogram
from muscriptor.utils.sampling import length_to_mask


ConditionType = tuple[torch.Tensor, torch.Tensor]  # (embedding [B, T, D], mask [B, T])


class WavCondition(NamedTuple):
    wav: torch.Tensor
    length: torch.Tensor
    sample_rate: list[int]
    path: list[str | None] = []
    seek_time: list[float | None] = []


@dataclass
class ConditioningAttributes:
    text: dict[str, str | None] = field(default_factory=dict)
    wav: dict[str, WavCondition] = field(default_factory=dict)
    joint_embed: dict[str, Any] = field(default_factory=dict)
    symbolic: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, item):
        return getattr(self, item)

    @property
    def text_attributes(self):
        return self.text.keys()

    @property
    def wav_attributes(self):
        return self.wav.keys()

    @property
    def joint_embed_attributes(self):
        return self.joint_embed.keys()

    @property
    def symbolic_attributes(self):
        return self.symbolic.keys()

    @property
    def attributes(self):
        return {
            "text": self.text_attributes,
            "wav": self.wav_attributes,
            "joint_embed": self.joint_embed_attributes,
            "symbolic": self.symbolic_attributes,
        }

    @classmethod
    def condition_types(cls) -> list:
        return ["text", "wav", "joint_embed", "symbolic"]


def nullify_wav(cond: WavCondition) -> WavCondition:
    B = cond.wav.shape[0]
    return WavCondition(
        wav=torch.zeros(*cond.wav.shape[:-1], 1, device=cond.wav.device),
        length=torch.zeros(B, dtype=cond.length.dtype, device=cond.wav.device),
        sample_rate=cond.sample_rate,
        path=[None] * B,
        seek_time=[None] * B,
    )


def nullify_all_conditions(
    samples: list[ConditioningAttributes],
) -> list[ConditioningAttributes]:
    """Return a copy of ``samples`` with every wav/text condition nulled out.

    Used to build the unconditional batch for classifier-free guidance.
    """
    samples = deepcopy(samples)
    for sample in samples:
        for k in list(sample.wav):
            sample.wav[k] = nullify_wav(sample.wav[k])
        for k in list(sample.text):
            sample.text[k] = None
    return samples


class MelSpectrogramConditioner(nn.Module):
    """Log mel spectrogram conditioner. Projects mel bins to transformer dim."""

    def __init__(
        self,
        output_dim: int,
        device: torch.device | str,
        sample_rate: int,
        n_fft: int = 2048,
        frame_rate: int = 100,
        n_mel_bins: int = 512,
        normalize_audio: bool = False,
        log_scale: bool = True,
        eps: float = 1e-6,
        # unused arg kept for config compatibility
        fine_frame_rate: int = None,
    ):
        self.fine_frame_rate_ratio = 1
        if fine_frame_rate is not None:
            assert fine_frame_rate % frame_rate == 0
            self.fine_frame_rate_ratio = fine_frame_rate // frame_rate

        super().__init__()
        self.dim = n_mel_bins * self.fine_frame_rate_ratio
        self.output_dim = output_dim
        self.output_proj = nn.Linear(self.dim, output_dim)
        self.device = device
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.normalize_audio = normalize_audio
        self.log_scale = log_scale
        self.eps = eps

        if self.fine_frame_rate_ratio == 1:
            assert sample_rate % frame_rate == 0
            self.hop_length = sample_rate // frame_rate
        else:
            assert sample_rate % fine_frame_rate == 0
            self.hop_length = sample_rate // fine_frame_rate

        self.mel_spec_transform = _MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=self.hop_length,
            n_mels=n_mel_bins,
            power=1.0,
            center=True,
            pad_mode="reflect",
        ).to(device)

    def tokenize(self, x: WavCondition) -> WavCondition:
        wav, length, sample_rate, path, seek_time = x
        assert length is not None
        return WavCondition(
            wav.to(self.device), length.to(self.device), sample_rate, path, seek_time
        )

    def _mel_embedding(self, x: WavCondition) -> torch.Tensor:
        if x.wav.shape[-1] == 1:
            return torch.zeros(x.wav.shape[0], 1, self.dim, device=self.device)
        muscriptor.accelerator.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            wav = x.wav
            if self.normalize_audio:
                wav = wav / (wav.abs().max(dim=-1, keepdim=True).values + 1e-8)
            mel = self.mel_spec_transform(wav)
            mel = rearrange(mel, "b 1 d t -> b t d")
            if self.fine_frame_rate_ratio > 1:
                mel = rearrange(
                    mel[:, :-1], "b (t f) d -> b t (f d)", f=self.fine_frame_rate_ratio
                )
            if self.log_scale:
                mel = torch.log(mel + self.eps)
        muscriptor.accelerator.synchronize()
        print(
            f"[muscriptor] mel-spec ({wav.shape[0]} × {wav.shape[-1]} samples): "
            f"{time.perf_counter() - t0:.3f}s"
        )
        return mel

    def forward(self, x: WavCondition) -> ConditionType:
        _, lengths, *_ = x
        with torch.no_grad():
            embeds = self._mel_embedding(x)
        embeds = embeds.to(self.output_proj.weight)
        embeds = self.output_proj(embeds)

        if lengths is not None:
            lengths = lengths / (self.sample_rate // self.frame_rate)
            mask = length_to_mask(lengths, max_len=embeds.shape[1]).int()
        else:
            mask = torch.ones_like(embeds[..., 0])
        mask_f = mask.float().unsqueeze(-1).to(embeds.device)
        embeds = embeds * mask_f
        return embeds, mask


class ClassConditioner(nn.Module):
    """Conditioner that embeds class indices (e.g., instrument group, dataset name)."""

    def __init__(
        self,
        num_classes: int,
        output_dim: int,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.embed = nn.Embedding(num_classes + 1, output_dim).to(device)
        self.pad_idx = 0

    def tokenize(self, x: list[str | None]) -> torch.Tensor:
        int_x = [list(map(int, s.split())) if s is not None else [-1] for s in x]
        max_len = max(len(xi) for xi in int_x)
        int_x = [xi + [-1] * (max_len - len(xi)) for xi in int_x]
        int_x = 1 + torch.LongTensor(int_x).to(self.device)
        return int_x

    def forward(self, inputs: torch.Tensor) -> ConditionType:
        embeds = self.embed(inputs + 1)
        mask = torch.ones_like(embeds[..., 0])
        return embeds, mask


def collate_wavs(
    samples: list[ConditioningAttributes], wav_conditions: list[str]
) -> dict[str, WavCondition]:
    """Collate wav conditions from a list of ConditioningAttributes."""
    wavs = defaultdict(list)
    lengths = defaultdict(list)
    sample_rates: dict[str, list] = defaultdict(list)
    paths: dict[str, list] = defaultdict(list)
    seek_times: dict[str, list] = defaultdict(list)
    out: dict[str, WavCondition] = {}

    for sample in samples:
        for attribute in wav_conditions:
            wav, length, sample_rate, path, seek_time = sample.wav[attribute]
            assert wav.dim() == 3
            B, K, T = wav.shape
            assert B == 1
            if K == 2:
                wav = wav.mean(1, keepdim=True)
            wavs[attribute].append(wav)
            lengths[attribute].append(length)
            sample_rates[attribute].extend(sample_rate)
            paths[attribute].extend(path)
            seek_times[attribute].extend(seek_time)

    for attribute in wav_conditions:
        # Stack along batch dim
        all_wavs = wavs[attribute]
        max_len = max(w.shape[-1] for w in all_wavs)
        padded = torch.cat(
            [F.pad(w, (0, max_len - w.shape[-1])) for w in all_wavs], dim=0
        )
        out[attribute] = WavCondition(
            padded,
            torch.cat(lengths[attribute]),
            sample_rates[attribute],
            paths[attribute],
            seek_times[attribute],
        )
    return out


class ConditioningProvider(nn.Module):
    """Runs all conditioners and returns a dict of condition tensors."""

    def __init__(
        self,
        conditioners: dict[str, nn.Module],
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.conditioners = nn.ModuleDict(conditioners)

    @property
    def text_conditions(self):
        return [
            k for k, v in self.conditioners.items() if isinstance(v, ClassConditioner)
        ]

    @property
    def wav_conditions(self):
        return [
            k
            for k, v in self.conditioners.items()
            if isinstance(v, MelSpectrogramConditioner)
        ]

    def tokenize(self, inputs: list[ConditioningAttributes]) -> dict[str, Any]:
        output = {}
        # Collate text conditions
        text_batch: dict[str, list[str | None]] = defaultdict(list)
        for sample in inputs:
            for cond in self.text_conditions:
                text_batch[cond].append(sample.text.get(cond))
        for attr, batch in text_batch.items():
            output[attr] = self.conditioners[attr].tokenize(batch)

        # Collate wav conditions
        if self.wav_conditions:
            wav_batch = collate_wavs(inputs, self.wav_conditions)
            for attr, wav_cond in wav_batch.items():
                output[attr] = self.conditioners[attr].tokenize(wav_cond)

        return output

    def forward(self, tokenized: dict[str, Any]) -> dict[str, ConditionType]:
        output = {}
        for attribute, inputs in tokenized.items():
            condition, mask = self.conditioners[attribute](inputs)
            output[attribute] = (condition, mask)
        return output
