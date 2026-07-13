//! Real-time microphone capture → streaming transcription.
//! Captures audio from the default mic, buffers into overlapping chunks,
//! and runs model inference on each chunk to emit detected notes.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{bounded, Sender};

use crate::mel::MelSpectrogram;
use crate::model::LMModel;
use crate::vocab::{
    build_event_vocab, decode_tokens, events_to_notes, instrument_group_from_names, Event, Note,
};

const SAMPLE_RATE: u32 = 16000;
const CHUNK_DURATION_SECS: f64 = 4.0;
const OVERLAP_SECS: f64 = 1.0;
const BUFFER_CAPACITY: usize = SAMPLE_RATE as usize * 30; // 30s

pub struct RealtimeTranscriber {
    model: LMModel,
    mel_spec: MelSpectrogram,
    vocab: Vec<Event>,
    inst_tokens: Option<String>,
    max_gen_len: usize,
    sampling: bool,
    temperature: f64,
    top_k: usize,
    top_p: f64,
}

impl RealtimeTranscriber {
    pub fn new(
        model: LMModel,
        inst_names: Option<Vec<String>>,
        max_gen_len: usize,
        sampling: bool,
        temperature: f64,
        top_k: usize,
        top_p: f64,
    ) -> Self {
        let mel_spec = MelSpectrogram::new(SAMPLE_RATE, 2048, 160, 512);
        let vocab = build_event_vocab(1001);
        let inst_tokens = inst_names
            .map(|names| instrument_group_from_names(&names).unwrap_or_default());
        Self { model, mel_spec, vocab, inst_tokens, max_gen_len, sampling, temperature, top_k, top_p }
    }

    /// Process one audio chunk (16 kHz mono f32) and return detected notes.
    pub fn transcribe_chunk(
        &self,
        chunk_audio: &[f32],
        chunk_start_time: f64,
    ) -> Result<Vec<Note>, Box<dyn std::error::Error>> {
        if chunk_audio.is_empty() {
            return Ok(vec![]);
        }

        let target_len = (CHUNK_DURATION_SECS * SAMPLE_RATE as f64) as usize;
        let mut audio_buf = chunk_audio.to_vec();
        audio_buf.resize(target_len, 0.0);

        let raw_mel = self.mel_spec.compute(&audio_buf);
        let log_mel = self.mel_spec.log_mel(&raw_mel, 1e-6);
        let t_frames = log_mel.len();
        let mel_flat: Vec<f32> = log_mel.into_iter().flatten().collect();
        let mel_t = candle_core::Tensor::from_vec(
            mel_flat, (1, t_frames, 512), &self.model.device,
        )?;
        let inst_t = self.model.ic.tokenize(
            &[self.inst_tokens.as_ref().map(|s| {
                s.split_whitespace()
                    .filter_map(|v| v.parse::<i64>().ok())
                    .collect()
            })],
            &self.model.device,
        )?;
        let ds_t = self
            .model
            .dc
            .tokenize(&[None], &self.model.device)?;

        let tokens = self.model.generate(
            &mel_t, &inst_t, &ds_t,
            self.max_gen_len, self.sampling, self.temperature,
            self.top_k, self.top_p,
        )?;

        let events = decode_tokens(&tokens, &self.vocab, chunk_start_time, None);
        let notes = events_to_notes(&events);

        let window_end = chunk_start_time + CHUNK_DURATION_SECS;
        Ok(notes
            .into_iter()
            .filter(|n| n.onset >= chunk_start_time && n.onset < window_end)
            .collect())
    }
}

/// Start microphone capture. Returns a receiver yielding (audio_chunk, start_time).
pub fn start_mic_capture(
) -> Result<
    (cpal::Stream, crossbeam_channel::Receiver<(Vec<f32>, f64)>),
    Box<dyn std::error::Error>,
> {
    let host = cpal::default_host();
    let input_device = host
        .default_input_device()
        .ok_or("No input device available")?;
    let config = input_device.default_input_config()?;
    let channels = config.channels() as usize;

    let ring: Arc<Mutex<VecDeque<f32>>> =
        Arc::new(Mutex::new(VecDeque::with_capacity(BUFFER_CAPACITY)));
    let ring_clone = ring.clone();
    let err_fn = |err| eprintln!("Audio stream error: {err}");

    let stream = input_device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut buf = ring_clone.lock().unwrap();
            if channels > 1 {
                for ch in data.chunks(channels) {
                    let mono: f32 = ch.iter().sum::<f32>() / channels as f32;
                    if buf.len() >= BUFFER_CAPACITY {
                        buf.pop_front();
                    }
                    buf.push_back(mono);
                }
            } else {
                for &s in data {
                    if buf.len() >= BUFFER_CAPACITY {
                        buf.pop_front();
                    }
                    buf.push_back(s);
                }
            }
        },
        err_fn,
        None,
    )?;

    let (tx, rx): (Sender<(Vec<f32>, f64)>, _) = bounded(10);
    let chunk_samples = (CHUNK_DURATION_SECS * SAMPLE_RATE as f64) as usize;
    let hop_samples = ((CHUNK_DURATION_SECS - OVERLAP_SECS) * SAMPLE_RATE as f64) as usize;

    thread::spawn(move || {
        let mut last_time = 0.0f64;
        loop {
            thread::sleep(Duration::from_millis(100));
            let mut read_buf = vec![0.0f32; chunk_samples];
            let filled = {
                let buf = ring.lock().unwrap();
                if buf.len() >= chunk_samples {
                    let start = buf.len() - chunk_samples;
                    for (i, s) in buf.range(start..).enumerate() {
                        read_buf[i] = *s;
                    }
                    chunk_samples
                } else {
                    0
                }
            };
            if filled > 0 {
                let t = last_time;
                last_time += hop_samples as f64 / SAMPLE_RATE as f64;
                if tx.send((read_buf, t)).is_err() {
                    break;
                }
            }
        }
    });

    stream.play()?;
    Ok((stream, rx))
}
