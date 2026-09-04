export async function decodeAudioFile(file: File): Promise<AudioBuffer> {
  const ctx = new OfflineAudioContext(1, 1, 44100);
  return await ctx.decodeAudioData(await file.arrayBuffer());
}

/** The buffer's channels, unwrapped into the array shape everything here uses. */
export function channelsOf(buffer: AudioBuffer): Float32Array[] {
  return Array.from({ length: buffer.numberOfChannels }, (_, c) =>
    buffer.getChannelData(c),
  );
}

/** Sample offsets of a start/end span in seconds, clamped to the buffer. */
function boundsOf(buffer: AudioBuffer, start: number, end: number) {
  return [
    Math.max(0, Math.round(start * buffer.sampleRate)),
    Math.min(buffer.length, Math.round(end * buffer.sampleRate)),
  ];
}

function sliceChannels(
  buffer: AudioBuffer,
  start: number,
  end: number,
): Float32Array[] {
  const [from, to] = boundsOf(buffer, start, end);
  return channelsOf(buffer).map((data) => data.slice(from, to));
}

/** The selection on its own. */
export function copyRegion(
  buffer: AudioBuffer,
  start: number,
  end: number,
): AudioBuffer {
  const channels = sliceChannels(buffer, start, end);
  const out = new AudioBuffer({
    length: channels[0].length,
    numberOfChannels: channels.length,
    sampleRate: buffer.sampleRate,
  });
  channels.forEach((data, c) => out.getChannelData(c).set(data));
  return out;
}

/** Everything but the selection, with the gap closed. */
export function cutRegion(
  buffer: AudioBuffer,
  start: number,
  end: number,
): AudioBuffer {
  const [from, to] = boundsOf(buffer, start, end);
  const out = new AudioBuffer({
    length: buffer.length - (to - from),
    numberOfChannels: buffer.numberOfChannels,
    sampleRate: buffer.sampleRate,
  });
  channelsOf(buffer).forEach((src, c) => {
    const dst = out.getChannelData(c);
    dst.set(src.subarray(0, from), 0);
    dst.set(src.subarray(to), from);
  });
  return out;
}

/**
 * Min/max pair per bucket, handed to WaveSurfer with the blob so it renders
 * from samples we already hold instead of decoding the audio a second time.
 */
export function peakEnvelope(
  buffer: AudioBuffer,
  pointsPerSecond = 2000,
): Float32Array[] {
  const buckets = Math.ceil((buffer.duration * pointsPerSecond) / 2);
  const per = Math.floor(buffer.length / buckets);
  const channels = channelsOf(buffer);
  if (per < 2) return channels;

  return channels.map((src) => {
    const out = new Float32Array(buckets * 2);
    for (let b = 0; b < buckets; b++) {
      const from = b * per;
      const to = b === buckets - 1 ? buffer.length : from + per;
      let min = src[from];
      let max = min;
      for (let i = from + 1; i < to; i++) {
        const v = src[i];
        if (v < min) min = v;
        else if (v > max) max = v;
      }
      out[b * 2] = min;
      out[b * 2 + 1] = max;
    }
    return out;
  });
}

export function encodeWav(channels: Float32Array[], sampleRate: number): Blob {
  const numChannels = channels.length;
  const frames = channels[0].length;
  const dataBytes = frames * numChannels * 2;
  const buf = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buf);

  const ascii = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++)
      view.setUint8(offset + i, s.charCodeAt(i));
  };
  ascii(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * 2, true);
  view.setUint16(32, numChannels * 2, true);
  view.setUint16(34, 16, true);
  ascii(36, "data");
  view.setUint32(40, dataBytes, true);

  let offset = 44;
  for (let i = 0; i < frames; i++) {
    for (let c = 0; c < numChannels; c++) {
      const s = Math.max(-1, Math.min(1, channels[c][i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }
  }
  return new Blob([buf], { type: "audio/wav" });
}

/** The span of an edited buffer that will be uploaded, kept unencoded. */
export type AudioEdit = {
  buffer: AudioBuffer;
  start: number;
  end: number;
};

/**
 * Encode a pending edit as the WAV that gets uploaded. Called once, at submit
 * time: encoding walks every sample, so doing it eagerly would re-run on each
 * region drag and each cut.
 */
export function editToWavFile(file: File, edit: AudioEdit): File {
  const { buffer, start, end } = edit;
  const blob = encodeWav(sliceChannels(buffer, start, end), buffer.sampleRate);
  const stem = file.name.replace(/\.[^/.]+$/, "");
  return new File(
    [blob],
    `${stem} (${formatTime(start)}-${formatTime(end)}).wav`,
    { type: "audio/wav" },
  );
}

export function formatTime(seconds: number): string {
  const whole = Math.round(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}
