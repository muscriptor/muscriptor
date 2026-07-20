"""TranscriptionModel: main user-facing entry point."""

import contextlib
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
    decode_model_tokens,
)
from muscriptor.models.lm import LMModel, TorchAutocast
from muscriptor.utils.download import download_companion, download_if_necessary
from muscriptor.modules.conditioners import (
    MelSpectrogramConditioner,
    ClassConditioner,
    ConditioningProvider,
    ConditioningAttributes,
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
    trim_overlapping_notes,
    validate_notes,
)
from muscriptor.utils.audio import load_audio, resample
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

        Interleaved with the note events are coarse :class:`ProgressEvent`
        anchors (``completed`` of ``total`` chunks): one up front with
        ``completed == 0``, then one as each chunk finishes. Consumers that
        only care about notes can ignore them.
        """
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
        num_chunks = math.ceil(total_samples / segment_samples)
        max_gen_len = 2000
        print(
            f"[muscriptor] audio: {total_duration:.1f}s → {num_chunks} chunk(s) of {_SEGMENT_DURATION}s",
            file=sys.stderr,
        )

        with _timed("build conditions", timings):
            all_conditions: list[ConditioningAttributes] = []
            seek_times: list[float] = []
            for i in range(num_chunks):
                start = i * segment_samples
                chunk = wav[:, start : start + segment_samples]
                if chunk.shape[-1] < segment_samples:
                    chunk = F.pad(chunk, (0, segment_samples - chunk.shape[-1]))
                all_conditions.append(
                    self._build_conditions(chunk, instrument_group)[0]
                )
                seek_times.append(i * _SEGMENT_DURATION)

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
    ) -> Iterator[int | ChunkBoundary | ProgressEvent]:
        """Generate tokens and yield them per chunk, as soon as they are ready.

        The model emits one token per chunk per timestep across the batch, but
        the decoder consumes whole chunks in order. So within each batch we
        stream the first chunk's tokens live as they are generated and buffer
        the others; once the first chunk hits EOS we flush the next chunk's
        buffered tokens and stream it live, and so on. EOS (and anything after
        it) is dropped.

        With ``prelude_forcing`` (and ``batch_size == 1``, so chunks generate
        strictly in order), every chunk after the first has its tie prologue
        teacher-forced: the notes left open by the previous chunk are encoded
        as ``(program, pitch)…tie`` tokens and passed to ``generate`` as a
        prompt, so the model can't restate them with the wrong instruments.
        The forced tokens flow through this stream like generated ones, which
        keeps the downstream decoder's view consistent by construction.
        """
        eos_id = self._tokenizer.eos_id
        num_chunks = len(seek_times)

        # Chunks in a batch generate concurrently, so with batch_size > 1 the
        # previous chunk's open notes aren't known when the next one starts —
        # forcing is only possible chunk-by-chunk. transcribe() rejects that
        # combination up front (_resolve_batch_size); this guard keeps the
        # invariant for direct callers too.
        tracker = None
        if prelude_forcing and batch_size == 1:
            tracker = OpenNoteTracker(
                self._tokenizer._vocab, self._tokenizer.frame_rate
            )

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
            bnd = boundary(batch_start)
            prompt = None
            if tracker is not None:
                # Feed the boundary first: it settles the tracker (e.g. a
                # previous chunk that never emitted its tie token drops all
                # open notes) so open_keys() is exactly the decoder's view.
                tracker.feed(bnd)
                if batch_start > 0:
                    prompt = torch.tensor(
                        [self._tokenizer.tie_section_token_ids(tracker.open_keys())],
                        device=self._device,
                        dtype=torch.long,
                    )
            yield bnd

            for step in self._model.generate(
                prompt=prompt,
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
                        if tracker is not None:
                            tracker.feed(tok)
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
