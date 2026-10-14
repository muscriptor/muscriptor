import { useEffect, useState } from "react";
import { Button } from "./Button";
import { Waveform, useWaveform } from "./Waveform";
import { useAudioHistory } from "../hooks/useAudioHistory";
import {
  copyRegion,
  cutRegion,
  decodeAudioFile,
  formatTime,
  type AudioEdit,
} from "../audio-edit";

type Range = [number, number] | null;

export function WaveformEditor(props: {
  file: File;
  setEdit: (edit: AudioEdit | null) => void;
}) {
  const { file, setEdit } = props;

  const [buffer, setBuffer] = useState<AudioBuffer | null>(null);
  const [original, setOriginal] = useState<AudioBuffer | null>(null);
  const [range, setRange] = useState<Range>(null);
  const [failed, setFailed] = useState(false);
  const [playing, setPlaying] = useState(false);

  const duration = buffer?.duration ?? 0;
  const edited = buffer !== null && buffer !== original;
  const { containerRef, playPause, pause } = useWaveform({
    file,
    buffer,
    edited,
    range,
    onRangeChange: setRange,
    onPlayingChange: setPlaying,
    onError: () => setFailed(true),
    active: !failed,
  });

  function showAudio(next: AudioBuffer) {
    pause();
    setRange(null);
    setBuffer(next);
  }

  const replaceAudio = useAudioHistory({ buffer, file, showAudio });
  const selected = range === null ? 0 : range[1] - range[0];
  const canEdit = selected > 0.01 && selected < duration - 0.01;

  // Report the span to upload; it is encoded once, when the user submits.
  useEffect(() => {
    if (buffer === null || (!edited && range === null)) {
      setEdit(null);
      return;
    }
    setEdit({ buffer, start: range?.[0] ?? 0, end: range?.[1] ?? duration });
  }, [buffer, range, edited, duration, setEdit]);

  useEffect(() => {
    let cancelled = false;
    setFailed(false);
    setBuffer(null);
    setRange(null);
    setOriginal(null);
    decodeAudioFile(file)
      .then((decoded) => {
        if (cancelled) return;
        setOriginal(decoded);
        setBuffer(decoded);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, [file]);

  function edit(apply: typeof cutRegion) {
    if (buffer !== null && range !== null) {
      replaceAudio(apply(buffer, range[0], range[1]));
    }
  }

  if (failed) return null;

  return (
    <section className="card flex flex-col gap-2 p-4">
      <Waveform ref={containerRef} />

      <div className="flex flex-wrap items-center gap-3">
        <Button onClick={playPause} disabled={duration === 0}>
          {playing ? "Pause" : range === null ? "Play" : "Play selection"}
        </Button>
        <Button
          onClick={() => edit(cutRegion)}
          disabled={!canEdit}
          title="Delete the selection and close the gap"
        >
          Cut selection
        </Button>
        <Button
          onClick={() => edit(copyRegion)}
          disabled={!canEdit}
          title="Keep only the selection"
        >
          Crop to selection
        </Button>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 text-sm text-muted">
        <span>
          {duration === 0 ? (
            "Reading the audio…"
          ) : range === null ? (
            <>
              {edited ? "Edited audio" : "Whole track"}, {formatTime(duration)}.
              Drag to select a part, scroll to zoom.
            </>
          ) : (
            <>
              <span className="text-content">
                {formatTime(range[0])} – {formatTime(range[1])}
              </span>{" "}
              of {formatTime(duration)}. Drag it to move it.
            </>
          )}
        </span>
        {range !== null && (
          <Button
            kind="ghost"
            className="px-1 py-0.5 text-sm text-muted underline underline-offset-4 hover:bg-transparent enabled:hover:text-content"
            onClick={() => setRange(null)}
          >
            Clear selection
          </Button>
        )}
      </div>
    </section>
  );
}
