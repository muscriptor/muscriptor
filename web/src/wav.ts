/**
 * Encode a decoded AudioBuffer as a mono 16-bit PCM WAV file.
 *
 * Uploads are sent as WAV built from the browser's own decode
 * (`decodeAudioData`) so the server never has to decode compressed formats
 * itself — the browser handles anything the user's machine can play (m4a/AAC
 * included, which the server's libsndfile can't read). Mono because both
 * server endpoints downmix anyway; the buffer's native sample rate is kept
 * (the server resamples to what it needs).
 */
export function audioBufferToWavFile(buffer: AudioBuffer, filename: string): File {
  const n = buffer.length;
  const channels = Array.from(
    { length: buffer.numberOfChannels },
    (_, c) => buffer.getChannelData(c),
  );

  const header = 44;
  const bytes = new DataView(new ArrayBuffer(header + n * 2));
  const writeAscii = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) bytes.setUint8(offset + i, s.charCodeAt(i));
  };
  writeAscii(0, "RIFF");
  bytes.setUint32(4, 36 + n * 2, true);
  writeAscii(8, "WAVE");
  writeAscii(12, "fmt ");
  bytes.setUint32(16, 16, true); // fmt chunk size
  bytes.setUint16(20, 1, true); // PCM
  bytes.setUint16(22, 1, true); // mono
  bytes.setUint32(24, buffer.sampleRate, true);
  bytes.setUint32(28, buffer.sampleRate * 2, true); // byte rate
  bytes.setUint16(32, 2, true); // block align
  bytes.setUint16(34, 16, true); // bits per sample
  writeAscii(36, "data");
  bytes.setUint32(40, n * 2, true);

  for (let i = 0; i < n; i++) {
    let sample = 0;
    for (const ch of channels) sample += ch[i];
    sample /= channels.length;
    sample = Math.max(-1, Math.min(1, sample));
    bytes.setInt16(header + i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }

  return new File([bytes.buffer], filename, { type: "audio/wav" });
}
