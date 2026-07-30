"""TranscriptionModel: main user-facing entry point."""

import contextlib
import gzip
import io
import json
import math
import re
import sys
import time
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

import muscriptor.accelerator
from muscriptor.events import (
    ChunkBoundary,
    NoteEndEvent,
    NoteStartEvent,
    OpenNoteTracker,
    ProgressEvent,
    _DrumHit,
    _EndNote,
    _StartNote,
    decode_model_tokens,
)
from muscriptor.models.lm import LMModel, TorchAutocast
from muscriptor.modules.conditioners import (
    ClassConditioner,
    ConditioningAttributes,
    ConditioningProvider,
    MelSpectrogramConditioner,
    WavCondition,
)
from muscriptor.tokenizer.mt3 import (
    MT3_FULL_PLUS_GROUP_NAMES,
    MT3Tokenizer,
    instrument_group_from_names,
)
from muscriptor.tokenizer.notes import (
    DRUM_PROGRAM,
    Note,
    NoteEvent,
    trim_overlapping_notes,
    validate_notes,
)
from muscriptor.utils.audio import load_audio, resample
from muscriptor.utils.download import download_companion, download_if_necessary
from muscriptor.utils.midi import notes_to_midi


@contextlib.contextmanager
def _timed(label: str, store: list[tuple[str, float]] | None = None):
    """Print and (optionally) record how long a block of work takes."""
    muscriptor.accelerator.synchronize()
    t0 = time.perf_counter()
    yield
    muscriptor.accelerator.synchronize()
    dt = time.perf_counter() - t0
    print(f"[muscriptor] {label}: {dt:.2f}s", file=sys.stderr)
    if store is not None:
        store.append((label, dt))


# Published model variants live at hf://MuScriptor/muscriptor-<size>. A bare
# size keyword ("small"/"medium"/"large") resolves to the matching repo; the
# architecture is then read from that repo's config.json (see _resolve_config).
_HF_REPO_TEMPLATE = "hf://MuScriptor/muscriptor-{size}/model.safetensors"
_MODEL_SIZES = ("small", "medium", "large")
_DEFAULT_SIZE = "medium"


def _resolve_source(weights_path: str | Path | None) -> str | Path:
    """Map a --model value to a weights location.

    A size keyword ("small"/"medium"/"large") — or None, which defaults to
    ``medium`` — becomes the corresponding HuggingFace repo URL. Anything else
    (a local path, an ``hf://`` or ``http(s)://`` URL) is passed through as-is.
    """
    if weights_path is None:
        weights_path = _DEFAULT_SIZE
    if isinstance(weights_path, str) and weights_path in _MODEL_SIZES:
        return _HF_REPO_TEMPLATE.format(size=weights_path)
    return weights_path

_SAMPLE_RATE = 16000
# Must match the segment duration used during training / evaluation.
_SEGMENT_DURATION = 5.0


@dataclass
class _ModelConfig:
    dim: int
    num_heads: int
    num_layers: int
    card: int


# Per-variant configs, keyed by the size that appears in the HF repo name
# (muscriptor-<size>). Each published repo also ships these values in its
# config.json; this table is the fallback when no config.json is present.
_CONFIGS: dict[str, _ModelConfig] = {
    "large": _ModelConfig(dim=1536, num_heads=24, num_layers=48, card=1395),
    "medium": _ModelConfig(dim=1024, num_heads=16, num_layers=24, card=1395),
    "small": _ModelConfig(dim=768, num_heads=12, num_layers=14, card=1393),
}

_DEFAULT_CONFIG = _CONFIGS["large"]

# Legacy local checkpoints identified by the 8-hex tag in their filename,
# mapped to the equivalent variant config.
_LEGACY_CONFIGS: dict[str, _ModelConfig] = {
    "01684fbb": _CONFIGS["large"],
    "0ac4ce03": _CONFIGS["small"],
    "8f59580c": _CONFIGS["medium"],
    "e84904c4": _CONFIGS["large"],
}

_CONFIG_FILENAME = "config.json"
_CONFIG_FIELDS = ("dim", "num_heads", "num_layers", "card")


def _config_from_json(path: Path) -> _ModelConfig:
    """Read a _ModelConfig from a HuggingFace-style config.json."""
    data = json.loads(path.read_text())
    return _ModelConfig(**{field: data[field] for field in _CONFIG_FIELDS})


def _resolve_config(source: str | Path, weights_path: Path) -> _ModelConfig:
    """Determine the model architecture for a set of weights.

    Resolution order, most to least authoritative:
      1. ``config.json`` sitting next to the weights — the self-describing,
         HuggingFace-idiomatic source of truth (local dir or hf:// repo).
      2. the ``muscriptor-<size>`` segment of an ``hf://`` repo name.
      3. the legacy 8-hex tag embedded in a local checkpoint filename.
    """
    config_path = weights_path.parent / _CONFIG_FILENAME
    if not config_path.exists():
        fetched = download_companion(source, _CONFIG_FILENAME)
        if fetched is not None:
            config_path = fetched
    if config_path.exists():
        return _config_from_json(config_path)

    m = re.search(r"muscriptor-(large|medium|small)", str(source))
    if m:
        return _CONFIGS[m.group(1)]

    m = re.search(r"_([0-9a-f]{8})_", weights_path.name)
    if m and m.group(1) in _LEGACY_CONFIGS:
        return _LEGACY_CONFIGS[m.group(1)]
    return _DEFAULT_CONFIG


def _remap_single_codebook_keys(state_dict: dict) -> dict:
    """Adapt legacy multi-codebook checkpoints to the single-stream LMModel.

    Older checkpoints store the token embedding and output head as the first
    entry of an ``nn.ModuleList`` (``emb.0.*`` / ``linears.0.*``). LMModel is
    single-stream, so those map to ``emb.*`` / ``linear.*``. Checkpoints with a
    second codebook (``emb.1.*`` etc.) are unsupported and rejected.
    """
    if any(k.startswith(("emb.1.", "linears.1.")) for k in state_dict):
        raise ValueError(
            "Checkpoint has more than one codebook (n_q > 1); "
            "only single-stream models are supported."
        )
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("emb.0."):
            key = "emb." + key[len("emb.0.") :]
        elif key.startswith("linears.0."):
            key = "linear." + key[len("linears.0.") :]
        remapped[key] = value
    return remapped


def _build_model(device: torch.device, cfg: _ModelConfig = _DEFAULT_CONFIG) -> LMModel:
    mel_cond = MelSpectrogramConditioner(
        output_dim=cfg.dim,
        device=device,
        sample_rate=_SAMPLE_RATE,
        n_fft=2048,
        frame_rate=100,
        n_mel_bins=512,
        log_scale=True,
        eps=1e-6,
        normalize_audio=False,
    )
    inst_cond = ClassConditioner(num_classes=1000, output_dim=cfg.dim, device=device)
    ds_cond = ClassConditioner(num_classes=4, output_dim=cfg.dim, device=device)

    condition_provider = ConditioningProvider(
        conditioners={
            "self_wav": mel_cond,
            "instrument_group": inst_cond,
            "dataset_name": ds_cond,
        },
        device=device,
    )

    # Disabled off-CUDA: on MPS half precision comes from native fp16 weights
    # (see load_model) — autocast there is measurably slower than fp32.
    autocast = TorchAutocast(enabled=False)
    if device.type == "cuda":
        autocast = TorchAutocast(enabled=True, device_type="cuda", dtype=torch.float16)

    model = LMModel(
        condition_provider=condition_provider,
        card=cfg.card,
        dim=cfg.dim,
        num_heads=cfg.num_heads,
        hidden_scale=4,
        cfg_coef=1.0,
        autocast=autocast,
        # StreamingTransformer kwargs (forwarded via **kwargs)
        num_layers=cfg.num_layers,
        max_period=10000,
        device=device,
    )
    return model


def _build_instrument_for_program(tokenizer: MT3Tokenizer) -> Callable[[int], str]:
    """Map a decoded program int → human-readable instrument name.

    MT3_FULL_PLUS groups multiple GM programs together; the decoded program
    is always the first program of the group. We map that representative
    back to the readable group name.
    """
    group_map = tokenizer.group_program_map
    program_to_name: dict[int, str] = {}
    for name, gid in MT3_FULL_PLUS_GROUP_NAMES.items():
        if gid in group_map and group_map[gid]:
            program_to_name[group_map[gid][0]] = name

    def lookup(program: int) -> str:
        if program == DRUM_PROGRAM:
            return "drums"
        return program_to_name.get(program, f"program_{program}")

    return lookup


class TranscriptionModel:
    """Transcribes audio to MIDI using the muscriptor model.

    Example::

        from pathlib import Path

        model = TranscriptionModel.load_model()
        for event in model.transcribe("audio.wav"):
            print(event)

        Path("out.mid").write_bytes(model.transcribe_to_midi("audio.wav"))
    """

    def __init__(self, model: LMModel, tokenizer: MT3Tokenizer, device: torch.device):
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._instrument_for_program = _build_instrument_for_program(tokenizer)

    @classmethod
    def load_model(
        cls,
        weights_path: str | Path | None = None,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
    ) -> "TranscriptionModel":
        """Load model weights and return a ready-to-use TranscriptionModel.

        Args:
            weights_path: A size keyword (``"small"``/``"medium"``/``"large"``)
                selecting a published HuggingFace variant, a local safetensors
                path, an ``hf://`` or ``https://`` URL, or None.  If None, the
                default ``medium`` variant is downloaded from HuggingFace.
                Remote URLs are cached under ~/.cache/muscriptor/.
            device: Torch device to use.  Defaults to the current accelerator
                (CUDA, MPS, ...) if one is available, else CPU.
            dtype: Transformer weight/compute dtype: ``"float32"``,
                ``"float16"``, ``"bfloat16"`` (or the torch dtypes). ``None``
                picks per device: float16 on MPS (halves memory traffic —
                decode is bandwidth-bound), float32 elsewhere (CUDA gets fp16
                compute via autocast instead). The conditioning pipeline
                (mel-spectrogram/class embeddings) always stays in fp32; its
                outputs are cast at the transformer boundary.
        """
        if device is None:
            device = (
                muscriptor.accelerator.current_accelerator()
                if muscriptor.accelerator.is_available()
                else torch.device("cpu")
            )
        elif isinstance(device, str):
            device = torch.device(device)

        if dtype is None:
            dtype = torch.float16 if device.type == "mps" else torch.float32
        elif isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        source = _resolve_source(weights_path)
        weights_path = download_if_necessary(source)
        model = _build_model(device, _resolve_config(source, weights_path))
        model.eval()

        state_dict = load_file(weights_path, device=str(device))
        state_dict = _remap_single_codebook_keys(state_dict)
        model.load_state_dict(state_dict)
        model.to(device)
        if dtype != torch.float32:
            model.to(dtype)
            # Conditioners keep fp32 numerics (log-mel of quiet passages
            # underflows in fp16); LMModel.forward casts their outputs.
            model.condition_provider.float()

        tokenizer = MT3Tokenizer(
            instrument_vocabulary="MT3_FULL_PLUS",
            max_shift_steps=1001,
        )

        return cls(model=model, tokenizer=tokenizer, device=device)

    # ------------------------------------------------------------------
    def transcribe(
        self,
        audio: str | Path | tuple[torch.Tensor, int],
        use_sampling: bool = False,
        temperature: float = 1.0,
        cfg_coef: float = 1.0,
        instruments: list[str] | None = None,
        batch_size: int | None = None,
        no_eos_is_ok: bool = True,
        beam_size: int = 1,
        prelude_forcing: bool = True,
        overlap: float = 0.0,
        allow_reset: bool = False,
    ) -> Iterator[NoteStartEvent | NoteEndEvent | ProgressEvent]:
        """Transcribe audio into a stream of note events.

        See the README for full argument documentation and the streaming /
        chunk-ordering guarantees. The audio is split into 5-second chunks;
        within each chunk events arrive in temporal order, and all events
        from chunk N are yielded before any event from chunk N+1.

        ``instruments``, when given, is a hard constraint: every program/drum
        token outside the listed groups is masked out during generation, so
        no other instrument can appear in the output. Leave it unset to let
        the model decode whatever instruments it detects.

        ``prelude_forcing`` (default True) teacher-forces each chunk's tie
        prologue — the tokens declaring which notes are sustained from the
        previous chunk — from the previous chunk's actually-unfinished notes,
        instead of letting the model guess (and occasionally re-enter with
        the wrong instruments). It requires chunks to be generated strictly
        in order, so while it is on the batch size defaults to (and must be)
        1; combining it with ``batch_size > 1`` raises ValueError — pass
        ``prelude_forcing=False`` explicitly to trade chunk-boundary quality
        for batched throughput.

        ``overlap`` (seconds, default 0) makes successive 5-second windows
        overlap by that much (stride ``5 - overlap``). It generalizes prelude
        forcing: instead of forcing only the tie prologue, each chunk after the
        first is teacher-forced to replay the *whole* note-event sequence the
        previous chunk predicted over the overlap region, so notes crossing the
        boundary continue with genuine left-context. It requires
        ``prelude_forcing`` (hence ``batch_size == 1``) and ``0 <= overlap <
        5``.

        ``allow_reset`` (default False, only effective with ``overlap > 0``)
        enables the gzip-based restart criterion: each overlapping chunk is
        generated twice — once with the overlap prompt and once without — and
        the un-prompted continuation is kept whenever the prompted one collapses
        into degenerate/repetitive output. It runs two generations per chunk and
        cannot stream tokens live, so it is an offline/CLI option; the streaming
        server path leaves it off.

        Interleaved with the note events are coarse :class:`ProgressEvent`
        anchors (``completed`` of ``total`` chunks): one up front with
        ``completed == 0``, then one as each chunk finishes. Consumers that
        only care about notes can ignore them.
        """
        if not 0.0 <= overlap < _SEGMENT_DURATION:
            raise ValueError(
                f"overlap={overlap} must be in [0, {_SEGMENT_DURATION}); it is "
                "the seconds two 5-second windows share, so it cannot reach the "
                "window length."
            )
        if overlap > 0 and not prelude_forcing:
            raise ValueError(
                "overlap > 0 teacher-forces each chunk from the previous one, "
                "so it requires prelude_forcing=True (batch_size 1)."
            )
        if allow_reset and overlap == 0:
            warnings.warn(
                "allow_reset has no effect with overlap=0: the gzip restart "
                "criterion only kicks in for overlapping chunks.",
                RuntimeWarning,
                stacklevel=2,
            )
            allow_reset = False

        batch_size = self._resolve_batch_size(batch_size, prelude_forcing)

        # Exact names only here — the CLI resolves abbreviations before
        # calling in (resolve_instrument_names).
        instrument_group = (
            instrument_group_from_names(instruments) if instruments else None
        )
        forbidden_tokens = None
        if instruments:
            forbidden_tokens = torch.tensor(
                self._tokenizer.forbidden_token_ids(instruments),
                device=self._device,
                dtype=torch.long,
            )

        timings: list[tuple[str, float]] = []
        t_total = time.perf_counter()

        if isinstance(audio, tuple):
            tensor, sample_rate = audio
            with _timed("load audio", timings):
                wav = self._load_wav(tensor, sample_rate)
        else:
            with _timed("load audio", timings):
                wav = self._load_wav(audio, None)

        total_samples = wav.shape[-1]
        total_duration = total_samples / _SAMPLE_RATE

        segment_samples = int(_SEGMENT_DURATION * _SAMPLE_RATE)
        # With overlap, windows advance by a stride shorter than the window, so
        # successive 5-second chunks share `overlap` seconds. overlap == 0 makes
        # stride == segment, i.e. the original adjacent-chunk layout.
        stride_sec = _SEGMENT_DURATION - overlap
        if total_duration <= _SEGMENT_DURATION:
            num_chunks = 1
        else:
            num_chunks = 1 + math.ceil(
                (total_duration - _SEGMENT_DURATION) / stride_sec
            )
        max_gen_len = 2000
        print(
            f"[muscriptor] audio: {total_duration:.1f}s → {num_chunks} chunk(s) of "
            f"{_SEGMENT_DURATION}s (stride {stride_sec}s, overlap {overlap}s)",
            file=sys.stderr,
        )

        with _timed("build conditions", timings):
            all_conditions: list[ConditioningAttributes] = []
            seek_times: list[float] = []
            for i in range(num_chunks):
                seek_time = i * stride_sec
                start = round(seek_time * _SAMPLE_RATE)
                chunk = wav[:, start : start + segment_samples]
                if chunk.shape[-1] < segment_samples:
                    chunk = F.pad(chunk, (0, segment_samples - chunk.shape[-1]))
                all_conditions.append(
                    self._build_conditions(chunk, instrument_group)[0]
                )
                seek_times.append(seek_time)

        t_gen = time.perf_counter()

        # Up-front anchor: tells consumers the total chunk count and gives them a
        # timing baseline (t0) for the first chunk, before any tokens are gen'd.
        yield ProgressEvent(completed=0, total=num_chunks)

        yield from decode_model_tokens(
            self._generate_token_stream(
                all_conditions,
                seek_times,
                batch_size,
                max_gen_len,
                use_sampling,
                temperature,
                cfg_coef,
                no_eos_is_ok,
                prelude_forcing,
                beam_size,
                forbidden_tokens,
                overlap,
                allow_reset,
            ),
            self._tokenizer._vocab,
            self._instrument_for_program,
            frame_rate=self._tokenizer.frame_rate,
        )

        muscriptor.accelerator.synchronize()
        print(
            f"[muscriptor] generate total: {time.perf_counter() - t_gen:.2f}s",
            file=sys.stderr,
        )
        print(
            f"[muscriptor] transcribe total: {time.perf_counter() - t_total:.2f}s "
            f"({total_duration:.1f}s audio)",
            file=sys.stderr,
        )

    def _resolve_batch_size(
        self, batch_size: int | None, prelude_forcing: bool
    ) -> int:
        """Default the batch size, favouring transcription quality.

        Prelude forcing needs chunks generated strictly in order, so while it
        is on (the default) the batch size defaults to — and must be — 1.
        Batching is an explicit quality trade-off: asking for both raises
        instead of silently dropping the forcing.
        """
        if batch_size is None:
            if prelude_forcing:
                return 1
            return 4 if self._device.type in ("cuda", "mps") else 1
        if prelude_forcing and batch_size > 1:
            raise ValueError(
                f"batch_size={batch_size} disables prelude forcing, which lowers "
                "quality at chunk boundaries; pass prelude_forcing=False to "
                "accept that trade-off"
            )
        return batch_size

    # ------------------------------------------------------------------
    def _generate_token_stream(
        self,
        all_conditions: list[ConditioningAttributes],
        seek_times: list[float],
        batch_size: int,
        max_gen_len: int,
        use_sampling: bool,
        temperature: float,
        cfg_coef: float,
        no_eos_is_ok: bool,
        prelude_forcing: bool = True,
        beam_size: int = 1,
        forbidden_tokens: torch.Tensor | None = None,
        overlap: float = 0.0,
        allow_reset: bool = False,
    ) -> Iterator[int | ChunkBoundary | ProgressEvent]:
        """Generate tokens and yield them per chunk, as soon as they are ready.

        Two paths, chosen by ``prelude_forcing``:

        * **Forcing** (``prelude_forcing`` and ``batch_size == 1``): chunks are
          generated strictly in order by :meth:`_forcing_stream`, so each chunk
          after the first can be teacher-forced from the previous one's output —
          the tie prologue always, plus (when ``overlap > 0``) the whole
          predicted overlap region, and (when ``allow_reset``) with the gzip
          restart criterion.

        * **Batched** (``prelude_forcing`` off): the model emits one token per
          chunk per timestep across the batch, but the decoder consumes whole
          chunks in order, so within each batch the active chunk streams live
          while the others buffer; once it hits EOS the next chunk's buffer is
          flushed and it streams live, and so on. ``overlap``/``allow_reset``
          have no effect here (transcribe() rejects overlap without forcing).
          EOS (and anything after it) is dropped.
        """
        eos_id = self._tokenizer.eos_id
        num_chunks = len(seek_times)

        # Forcing needs the previous chunk's output before the next one starts,
        # so it only works chunk-by-chunk (batch_size == 1). transcribe() rejects
        # forcing + batch_size > 1 up front (_resolve_batch_size); this guard
        # keeps the invariant for direct callers too.
        if prelude_forcing and batch_size == 1:
            yield from self._forcing_stream(
                all_conditions,
                seek_times,
                max_gen_len,
                use_sampling,
                temperature,
                cfg_coef,
                no_eos_is_ok,
                beam_size,
                forbidden_tokens,
                overlap,
                allow_reset,
            )
            return

        def boundary(chunk_index: int) -> ChunkBoundary:
            next_seek_time = (
                seek_times[chunk_index + 1] if chunk_index + 1 < num_chunks else None
            )
            return ChunkBoundary(seek_times[chunk_index], next_seek_time)

        for batch_start in range(0, num_chunks, batch_size):
            batch_conditions = all_conditions[batch_start : batch_start + batch_size]
            n = len(batch_conditions)
            buffers: list[list[int]] = [[] for _ in range(n)]
            done = [False] * n
            active = 0  # within-batch index of the chunk streaming live

            # The first chunk in the batch streams live from the start.
            yield boundary(batch_start)

            for step in self._model.generate(
                prompt=None,
                conditions=batch_conditions,
                max_gen_len=max_gen_len,
                use_sampling=use_sampling,
                temp=temperature,
                top_k=0,
                top_p=0.0,
                cfg_coef=cfg_coef,
                early_stop_on_token=eos_id,
                beam_size=beam_size,
                forbidden_tokens=forbidden_tokens,
            ):
                row = step.tolist()  # one token per chunk: [n]
                for j in range(n):
                    if done[j]:
                        continue
                    tok = row[j]
                    if tok == eos_id:
                        done[j] = True
                    else:
                        if j == active:
                            yield tok
                        else:
                            buffers[j].append(tok)
                # When the live chunk finishes, flush and stream the next one(s).
                while active < n and done[active]:
                    active += 1
                    if active < n:
                        yield boundary(batch_start + active)
                        yield from buffers[active]
                        buffers[active] = []

            # Any chunk still open never emitted EOS within max_gen_len.
            for j in range(active, n):
                if not done[j]:
                    chunk_index = batch_start + j
                    msg = (
                        f"chunk {chunk_index} (seek={seek_times[chunk_index]:.1f}s) "
                        f"did not emit EOS within {max_gen_len} tokens"
                    )
                    if no_eos_is_ok:
                        warnings.warn(msg, RuntimeWarning, stacklevel=2)
                    else:
                        raise RuntimeError(
                            msg + " (this is only raised under --strict-eos)"
                        )
                # The live (active) chunk has already streamed; emit the rest.
                if j != active:
                    yield boundary(batch_start + j)
                    yield from buffers[j]

            # This batch's chunks are fully generated: emit a completion anchor.
            # (batch_size=1 on the web path => one event per chunk.) The event
            # trails the chunk's tokens, so by the time it surfaces from
            # decode_model_tokens all of that chunk's notes have been yielded.
            yield ProgressEvent(completed=batch_start + n, total=num_chunks)

    # ------------------------------------------------------------------
    def _forcing_stream(
        self,
        all_conditions: list[ConditioningAttributes],
        seek_times: list[float],
        max_gen_len: int,
        use_sampling: bool,
        temperature: float,
        cfg_coef: float,
        no_eos_is_ok: bool,
        beam_size: int,
        forbidden_tokens: torch.Tensor | None,
        overlap: float,
        allow_reset: bool,
    ) -> Iterator[int | ChunkBoundary | ProgressEvent]:
        """In-order, teacher-forced generation (batch_size == 1).

        Each chunk after the first is prompted from the previous chunk's output.
        A single :class:`OpenNoteTracker` decodes the stream as it is produced,
        so ``open_keys()`` at a boundary is exactly the decoder's view of the
        notes to declare in the tie prologue. When ``overlap > 0`` the prompt
        also replays the previous chunk's note events over the shared window
        (see :meth:`_overlap_note_events`); when ``allow_reset`` those chunks are
        generated twice and the gzip restart criterion picks the better one.

        The prompt tokens flow back through this stream like generated ones, so
        the tracker and the downstream decoder stay consistent by construction.
        """
        eos_id = self._tokenizer.eos_id
        num_chunks = len(seek_times)
        tracker = OpenNoteTracker(self._tokenizer._vocab, self._tokenizer.frame_rate)
        prev_tokens: list[int] = []
        prev_seek = 0.0

        for i in range(num_chunks):
            seek = seek_times[i]
            next_seek = seek_times[i + 1] if i + 1 < num_chunks else None
            bnd = ChunkBoundary(seek, next_seek)
            # Feed the boundary first: it settles the tracker (a previous chunk
            # that never closed its tie prologue drops all open notes) so
            # open_keys() matches the decoder exactly.
            tracker.feed(bnd)

            prompt = None
            if i > 0:
                prompt_ids = self._forcing_prompt_ids(
                    tracker.open_keys(),
                    prev_tokens,
                    prev_seek,
                    seek,
                    overlap,
                    max_gen_len,
                    i,
                )
                if prompt_ids:
                    prompt = torch.tensor(
                        [prompt_ids], device=self._device, dtype=torch.long
                    )
            yield bnd

            if allow_reset and prompt is not None:
                # Generate with the overlap prompt and again with no prompt, then
                # keep the un-prompted continuation if the prompted one collapsed
                # (gzip restart criterion). Buffered, so no live streaming here.
                with_tokens, wp_ended = self._collect_chunk(
                    prompt, all_conditions[i], max_gen_len, use_sampling,
                    temperature, cfg_coef, beam_size, forbidden_tokens,
                )
                no_tokens, np_ended = self._collect_chunk(
                    None, all_conditions[i], max_gen_len, use_sampling,
                    temperature, cfg_coef, beam_size, forbidden_tokens,
                )
                chunk_tokens = self._select_with_gzip(
                    no_tokens, np_ended, with_tokens, wp_ended, seek
                )
                ended = wp_ended if chunk_tokens is with_tokens else np_ended
                for tok in chunk_tokens:
                    tracker.feed(tok)
                    yield tok
            else:
                chunk_tokens = []
                ended = False
                for step in self._model.generate(
                    prompt=prompt,
                    conditions=[all_conditions[i]],
                    max_gen_len=max_gen_len,
                    use_sampling=use_sampling,
                    temp=temperature,
                    top_k=0,
                    top_p=0.0,
                    cfg_coef=cfg_coef,
                    early_stop_on_token=eos_id,
                    beam_size=beam_size,
                    forbidden_tokens=forbidden_tokens,
                ):
                    tok = int(step[0])
                    if tok == eos_id:
                        ended = True
                        break
                    tracker.feed(tok)
                    chunk_tokens.append(tok)
                    yield tok

            if not ended:
                msg = (
                    f"chunk {i} (seek={seek:.1f}s) did not emit EOS within "
                    f"{max_gen_len} tokens"
                )
                if no_eos_is_ok:
                    warnings.warn(msg, RuntimeWarning, stacklevel=2)
                else:
                    raise RuntimeError(
                        msg + " (this is only raised under --strict-eos)"
                    )

            prev_tokens = chunk_tokens
            prev_seek = seek
            yield ProgressEvent(completed=i + 1, total=num_chunks)

    def _forcing_prompt_ids(
        self,
        open_keys: list[tuple[int, int]],
        prev_tokens: list[int],
        prev_seek: float,
        seek: float,
        overlap: float,
        max_gen_len: int,
        chunk_index: int,
    ) -> list[int]:
        """Build the teacher-forcing prompt for chunk ``chunk_index`` (> 0).

        Overlap forcing replays the previous chunk's note events over the shared
        window; with ``overlap == 0`` it's just the tie prologue. A runaway
        previous chunk (no EOS within ``max_gen_len`` — degenerate/repetitive)
        can make the overlap prompt longer than the generation buffer itself, so
        the prompt is capped to fit: too long → fall back to the tie prologue
        alone; still too long → no forcing at all (empty list). ``generate``
        writes the prompt into a ``max_gen_len + 1`` buffer and needs at least
        one free slot to decode into, so the cap is ``max_gen_len - 1``. The
        gzip restart criterion (``allow_reset``) is the quality-side remedy for
        such chunks; this cap only keeps generation from overflowing.
        """
        if overlap > 0:
            events = self._overlap_note_events(prev_tokens, prev_seek, seek, overlap)
            prompt_ids = self._tokenizer.overlap_prompt_token_ids(
                open_keys, events, seek
            )
        else:
            prompt_ids = self._tokenizer.tie_section_token_ids(open_keys)

        max_prompt = max_gen_len - 1
        if len(prompt_ids) <= max_prompt:
            return prompt_ids

        tie_only = self._tokenizer.tie_section_token_ids(open_keys)
        if len(tie_only) <= max_prompt:
            warnings.warn(
                f"chunk {chunk_index} (seek={seek:.1f}s): overlap prompt "
                f"({len(prompt_ids)} tokens) exceeds the generation budget "
                f"({max_gen_len}); forcing only the tie prologue for this chunk. "
                "The previous chunk likely ran away without emitting EOS.",
                RuntimeWarning,
                stacklevel=2,
            )
            return tie_only

        warnings.warn(
            f"chunk {chunk_index} (seek={seek:.1f}s): tie prologue "
            f"({len(tie_only)} tokens) exceeds the generation budget "
            f"({max_gen_len}); generating without forcing for this chunk.",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    def _collect_chunk(
        self,
        prompt: torch.Tensor | None,
        condition: ConditioningAttributes,
        max_gen_len: int,
        use_sampling: bool,
        temperature: float,
        cfg_coef: float,
        beam_size: int,
        forbidden_tokens: torch.Tensor | None,
    ) -> tuple[list[int], bool]:
        """Run one single-chunk generation to completion (used by the reset path).

        Returns the chunk's token ids (prompt echoes included, EOS stripped) and
        whether it terminated with EOS within the budget.
        """
        eos_id = self._tokenizer.eos_id
        tokens: list[int] = []
        ended = False
        for step in self._model.generate(
            prompt=prompt,
            conditions=[condition],
            max_gen_len=max_gen_len,
            use_sampling=use_sampling,
            temp=temperature,
            top_k=0,
            top_p=0.0,
            cfg_coef=cfg_coef,
            early_stop_on_token=eos_id,
            beam_size=beam_size,
            forbidden_tokens=forbidden_tokens,
        ):
            tok = int(step[0])
            if tok == eos_id:
                ended = True
                break
            tokens.append(tok)
        return tokens, ended

    def _overlap_note_events(
        self,
        prev_tokens: list[int],
        prev_seek: float,
        seek: float,
        overlap: float,
    ) -> list[NoteEvent]:
        """Note events the previous chunk predicted in ``[seek, seek+overlap)``.

        Decodes ``prev_tokens`` with a windowless :class:`OpenNoteTracker` (no
        ``next_seek_time`` cutoff, so the events past the boundary — exactly the
        ones the main decoder dropped and this chunk will replay — are kept) and
        returns those falling inside the overlap window, as absolute-time
        :class:`NoteEvent`s ready for :meth:`overlap_prompt_token_ids`.
        """
        scratch = OpenNoteTracker(
            self._tokenizer._vocab, self._tokenizer.frame_rate
        )
        actions = list(scratch.feed(ChunkBoundary(prev_seek, None)))
        for tok in prev_tokens:
            actions.extend(scratch.feed(tok))

        lo, hi = seek, seek + overlap
        events: list[NoteEvent] = []
        for a in actions:
            if not (lo <= a.time < hi):
                continue
            if isinstance(a, _StartNote):
                events.append(
                    NoteEvent(False, a.program, a.time, velocity=1, pitch=a.pitch)
                )
            elif isinstance(a, _EndNote):
                events.append(
                    NoteEvent(False, a.program, a.time, velocity=0, pitch=a.pitch)
                )
            elif isinstance(a, _DrumHit):
                events.append(
                    NoteEvent(True, DRUM_PROGRAM, a.time, velocity=1, pitch=a.pitch)
                )
        return events

    def _gzip_ratio(self, tokens: list[int]) -> float:
        """Repetitiveness proxy: ``len(symbols) / len(gzip(symbols))``.

        Each token becomes a short per-type symbol (shift/velocity/program/
        pitch/drum); a degenerate, looping chunk compresses far better, so a
        high ratio flags it. ``-1`` for an empty (all-special) chunk.
        """
        vocab = self._tokenizer._vocab
        parts: list[str] = []
        for t in tokens:
            e = vocab[t]
            if e.type == "shift":
                parts.append("s")
            elif e.type in ("PAD", "EOS"):
                break
            elif e.type == "velocity":
                parts.append(f"v{e.value}")
            elif e.type == "program":
                parts.append(f"r{e.value}")
            elif e.type == "pitch":
                parts.append(f"p{e.value}")
            elif e.type == "drum":
                parts.append(f"d{e.value}")
        raw = "".join(parts).encode()
        if len(raw) == 0:
            return -1.0
        return len(raw) / len(gzip.compress(raw))

    def _select_with_gzip(
        self,
        no_tokens: list[int],
        np_ended: bool,
        with_tokens: list[int],
        wp_ended: bool,
        seek: float,
    ) -> list[int]:
        """Pick the overlap-prompted or the un-prompted chunk (restart criterion).

        Reset to the un-prompted generation when it terminated cleanly and the
        prompted one looks degenerate — it never terminated, produced nothing,
        or is markedly more repetitive (gzip ratio ≥ 3 and worse than the
        un-prompted one). Otherwise keep the prompted continuation. Returns the
        chosen list identity so the caller can recover its EOS flag.
        """
        np_gzip = self._gzip_ratio(no_tokens)
        wp_gzip = self._gzip_ratio(with_tokens)
        if np_ended and (
            wp_gzip < 0.0 or not wp_ended or (wp_gzip >= 3 and np_gzip < wp_gzip)
        ):
            warnings.warn(
                f"chunk (seek={seek:.1f}s): overlap prompt drove the model "
                "degenerate; resetting and ignoring the previous chunk's overlap.",
                RuntimeWarning,
                stacklevel=2,
            )
            return no_tokens
        return with_tokens

    # ------------------------------------------------------------------
    def transcribe_to_midi(
        self,
        audio: str | Path | tuple[torch.Tensor, int],
        use_sampling: bool = False,
        temperature: float = 1.0,
        cfg_coef: float = 1.0,
        instruments: list[str] | None = None,
        batch_size: int | None = None,
        no_eos_is_ok: bool = True,
        beam_size: int = 1,
        prelude_forcing: bool = True,
        overlap: float = 0.0,
        allow_reset: bool = False,
    ) -> bytes:
        """Same as :meth:`transcribe` but returns a MIDI file as bytes."""
        events = self.transcribe(
            audio,
            use_sampling=use_sampling,
            temperature=temperature,
            cfg_coef=cfg_coef,
            instruments=instruments,
            batch_size=batch_size,
            no_eos_is_ok=no_eos_is_ok,
            beam_size=beam_size,
            prelude_forcing=prelude_forcing,
            overlap=overlap,
            allow_reset=allow_reset,
        )
        return self.events_to_midi_bytes(events)

    def events_to_midi_bytes(
        self, events: Iterator[NoteStartEvent | NoteEndEvent | ProgressEvent]
    ) -> bytes:
        """Reassemble Notes from a NoteStart/NoteEnd stream and serialize MIDI.

        Shared by :meth:`transcribe_to_midi` and the HTTP server, so the MIDI
        bytes are identical regardless of how the events were obtained.
        """
        notes: list[Note] = []
        open_notes: dict[int, Note] = {}
        program_names: dict[int, str] = {}
        for ev in events:
            if isinstance(ev, ProgressEvent):
                continue
            if isinstance(ev, NoteStartEvent):
                is_drum = ev.instrument == "drums"
                program = (
                    DRUM_PROGRAM
                    if is_drum
                    else self._program_for_instrument(ev.instrument)
                )
                program_names[program] = ev.instrument.replace("_", " ")
                note = Note(
                    is_drum=is_drum,
                    program=program,
                    onset=ev.start_time,
                    offset=ev.start_time,  # patched on NoteEndEvent
                    pitch=ev.pitch,
                )
                open_notes[ev.index] = note
            else:  # NoteEndEvent
                note = open_notes.pop(ev.start_event_index)
                note.offset = ev.end_time
                notes.append(note)

        # Match the legacy decoder's note-cleanup pass so the MIDI bytes
        # don't drift from earlier reference outputs.
        notes = validate_notes(notes, fix=True)
        notes = trim_overlapping_notes(notes, sort=True)
        midi = notes_to_midi(notes, program_names=program_names)
        buf = io.BytesIO()
        midi.save(file=buf)
        return buf.getvalue()

    def _program_for_instrument(self, instrument: str) -> int:
        """Inverse of `_instrument_for_program` for non-drum instruments."""
        if not hasattr(self, "_inst_to_program"):
            group_map = self._tokenizer.group_program_map
            self._inst_to_program = {
                name: group_map[gid][0]
                for name, gid in MT3_FULL_PLUS_GROUP_NAMES.items()
                if gid in group_map and group_map[gid]
            }
        if instrument in self._inst_to_program:
            return self._inst_to_program[instrument]
        # fallback for unknown names like "program_42"
        if instrument.startswith("program_"):
            return int(instrument.removeprefix("program_"))
        raise ValueError(f"Unknown instrument name: {instrument!r}")

    # ------------------------------------------------------------------
    def _load_wav(
        self, audio: str | Path | torch.Tensor, sample_rate: int | None
    ) -> torch.Tensor:
        """Return mono float32 waveform at 16 kHz, shape [1, T]."""
        if isinstance(audio, (str, Path)):
            wav = load_audio(audio, target_sr=_SAMPLE_RATE)
        else:
            wav = audio.float()
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            if wav.dim() == 3:
                wav = wav.squeeze(0)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            if sample_rate is not None and sample_rate != _SAMPLE_RATE:
                wav = resample(wav, sample_rate, _SAMPLE_RATE)
        return wav.to(self._device)

    def _build_conditions(
        self,
        wav: torch.Tensor,
        instrument_group: str | None = None,
    ) -> list[ConditioningAttributes]:
        """Build a single-element list of ConditioningAttributes for one 5-second chunk."""
        T = wav.shape[-1]
        wav_3d = wav.unsqueeze(0)  # [1, 1, T]
        length = torch.tensor([T], device=self._device)
        wav_cond = WavCondition(
            wav=wav_3d,
            length=length,
            sample_rate=[_SAMPLE_RATE],
            path=[None],
            seek_time=[0.0],
        )
        return [
            ConditioningAttributes(
                wav={"self_wav": wav_cond},
                text={
                    "instrument_group": instrument_group,
                    # Always unconditional on dataset: the null/pad class.
                    "dataset_name": None,
                },
            )
        ]
