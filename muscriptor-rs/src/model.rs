use std::collections::HashMap;
use std::path::Path;

use candle_core::{Device, Result, Tensor, DType, safetensors};
use candle_nn::{Module, VarBuilder};

use crate::config::ModelConfig;

// ---------------------------------------------------------------------------
// Sinusoidal positional embeddings
// ---------------------------------------------------------------------------

fn sin_embedding(pos: &Tensor, dim: usize, max_period: f64) -> Result<Tensor> {
    let half = dim / 2;
    let device = pos.device();
    let dtype = pos.dtype();
    // arange(0, half) -> [1, 1, half]
    let arange: Vec<f32> = (0..half).map(|i| i as f32).collect();
    let adim = Tensor::from_vec(arange, (1, 1, half), device)?.to_dtype(dtype)?;
    let log_mt = Tensor::new((max_period as f32).ln(), device)?.to_dtype(dtype)?;
    let hf = (half as f32) - 1.0f32;
    let adim_ratio = adim.broadcast_div(&Tensor::new(hf, device)?.to_dtype(dtype)?)?;
    let divisor = adim_ratio.broadcast_mul(&log_mt)?.exp()?;
    let phase = pos.broadcast_div(&divisor)?;
    Tensor::cat(&[&phase.cos()?, &phase.sin()?], 2)
}

// ---------------------------------------------------------------------------
// ScaledEmbedding wrapper
// ---------------------------------------------------------------------------

struct ScaledEmb {
    inner: candle_nn::Embedding,
}

impl ScaledEmb {
    fn new(vs: usize, d: usize, vb: &VarBuilder) -> Result<Self> {
        let w = vb.get((vs, d), "weight")?;
        Ok(Self { inner: candle_nn::Embedding::new(w, d) })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let neg = x.lt(0)?.unsqueeze(2)?.to_dtype(DType::F32)?;
        let emb = self.inner.forward(&x.clamp(0, i64::MAX)?)?;
        // Zero out positions where input was negative
        let ones = Tensor::new(1.0f32, x.device())?.to_dtype(emb.dtype())?;
        emb.broadcast_mul(&(ones.broadcast_sub(&neg))?)
    }
}

// ---------------------------------------------------------------------------
// Multi-head attention with KV cache
// ---------------------------------------------------------------------------

struct Attn {
    dh: usize,
    nh: usize,
    in_w: Tensor,
    out_w: Tensor,
}

impl Attn {
    fn new(d: usize, nh: usize, vb: &VarBuilder) -> Result<Self> {
        let three_d = 3usize * d;
        Ok(Self {
            dh: d / nh, nh,
            in_w: vb.get((three_d, d), "in_proj_weight")?,
            out_w: vb.get((d, d), "out_proj.weight")?,
        })
    }

    fn forward(
        &self, x: &Tensor,
        kc: &mut Option<Tensor>, vc: &mut Option<Tensor>,
        off: &mut usize,
    ) -> Result<Tensor> {
        let dev = x.device();
        let dt = x.dtype();
        let (b, t, d) = x.dims3()?;
        let nh = self.nh;
        let dh = self.dh;

        let p = matmul_3d(x, &self.in_w.t()?)?.reshape((b, t, 3usize, nh, dh))?;
        let q = p.narrow(2, 0, 1)?.squeeze(2)?.transpose(1, 2)?;
        let k = p.narrow(2, 1, 1)?.squeeze(2)?.transpose(1, 2)?;
        let v = p.narrow(2, 2, 1)?.squeeze(2)?.transpose(1, 2)?;

        let (k, v) = if let (Some(kk), Some(vv)) = (kc.as_mut(), vc.as_mut()) {
            let s = *off;
            let kf = Tensor::cat(&[&kk.narrow(2, 0, s)?, &k], 2)?;
            let vf = Tensor::cat(&[&vv.narrow(2, 0, s)?, &v], 2)?;
            *kc = Some(kf.clone());
            *vc = Some(vf.clone());
            (kf, vf)
        } else {
            *kc = Some(k.clone());
            *vc = Some(v.clone());
            (k, v)
        };
        *off += t;

        let tq = q.dim(2)?;
        let tk = k.dim(2)?;
        let scale = Tensor::new((dh as f64).powf(-0.5) as f32, dev)?;
        let mut a = q.matmul(&k.transpose(2, 3)?)?.broadcast_mul(&scale)?;

        // Causal mask (bottom-right triangular)
        let mut mask_data: Vec<u8> = Vec::with_capacity(tq * tk);
        for i in 0..tq {
            for j in 0..tk {
                mask_data.push(if j <= i + tk - tq { 1 } else { 0 });
            }
        }
        let mask = Tensor::from_vec(mask_data, (tq, tk), dev)?.to_dtype(dt)?;
        let one = Tensor::new(1.0f32, dev)?.to_dtype(dt)?;
        let ni = Tensor::new(f32::NEG_INFINITY, dev)?;
        let am = (one.broadcast_sub(&mask))?.broadcast_mul(&ni)?;
        a = (a.broadcast_mul(&mask)?.broadcast_add(&am))?;

        let a = candle_nn::ops::softmax(&a, 3)?;
        let o = a.matmul(&v)?;
        let o = o.transpose(1, 2)?.reshape((b, tq, nh * dh))?;
        matmul_3d(&o, &self.out_w.t()?)
    }
}

fn matmul_3d(x: &Tensor, wt: &Tensor) -> Result<Tensor> {
    let nd = x.dims().len();
    if nd <= 2 {
        x.matmul(wt)
    } else {
        let (b, t, d) = x.dims3()?;
        let x2 = x.reshape((b * t, d))?;
        let y2 = x2.matmul(wt)?;
        y2.reshape((b, t, wt.dim(1)?))
    }
}

// ---------------------------------------------------------------------------
// Linear (no bias) helper
// ---------------------------------------------------------------------------

struct LinNB { w: Tensor }
impl LinNB {
    fn new(in_dim: usize, out_dim: usize, vb: &VarBuilder, name: &str) -> Result<Self> {
        let w = vb.get((out_dim, in_dim), &format!("{}.weight", name))?;
        Ok(Self { w })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let wt = self.w.t()?;
        let nd = x.dims().len();
        if nd <= 2 {
            x.matmul(&wt)
        } else {
            let (b, t, d) = x.dims3()?;
            let x2 = x.reshape((b * t, d))?;
            let y2 = x2.matmul(&wt)?;
            y2.reshape((b, t, wt.dim(1)?))
        }
    }
}

// ---------------------------------------------------------------------------
// Linear with bias helper
// ---------------------------------------------------------------------------

struct LinB { w: Tensor, b: Tensor }
impl LinB {
    fn new(in_dim: usize, out_dim: usize, vb: &VarBuilder, name: &str) -> Result<Self> {
        let w = vb.get((out_dim, in_dim), &format!("{}.weight", name))?;
        let b = vb.get(out_dim, &format!("{}.bias", name))?;
        Ok(Self { w, b })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = matmul_3d(x, &self.w.t()?)?;
        y.broadcast_add(&self.b)
    }
}

// ---------------------------------------------------------------------------
// LayerNorm helper (manual implementation)
// ---------------------------------------------------------------------------

struct LN { w: Tensor, b: Tensor, eps: f64 }
impl LN {
    fn new(dim: usize, eps: f64, vb: &VarBuilder, name: &str) -> Result<Self> {
        let w = vb.get(dim, &format!("{}.weight", name))?;
        let b = vb.get(dim, &format!("{}.bias", name))?;
        Ok(Self { w, b, eps })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(2)?;
        let x_center = x.broadcast_sub(&mean)?;
        let var = x_center.sqr()?.mean_keepdim(2)?;
        let eps_t = Tensor::new(self.eps as f32, var.device())?;
        let denom = (var.broadcast_add(&eps_t))?.sqrt()?;
        let x_norm = x_center.broadcast_div(&denom)?;
        x_norm.broadcast_mul(&self.w.unsqueeze(0)?)?
            .broadcast_add(&self.b.unsqueeze(0)?)
    }
}

// ---------------------------------------------------------------------------
// Transformer Layer
// ---------------------------------------------------------------------------

struct TLayer {
    attn: Attn,
    n1: LN, n2: LN,
    l1: LinNB, l2: LinNB,
}

impl TLayer {
    fn new(d: usize, nh: usize, ff: usize, vb: &VarBuilder, prefix: &str) -> Result<Self> {
        Ok(Self {
            attn: Attn::new(d, nh, &vb.pp(&format!("{}.self_attn", prefix)))?,
            n1: LN::new(d, 1e-5, vb, &format!("{}.norm1", prefix))?,
            n2: LN::new(d, 1e-5, vb, &format!("{}.norm2", prefix))?,
            l1: LinNB::new(d, ff, vb, &format!("{}.linear1", prefix))?,
            l2: LinNB::new(ff, d, vb, &format!("{}.linear2", prefix))?,
        })
    }
    fn forward(&self, x: &Tensor, kc: &mut Option<Tensor>, vc: &mut Option<Tensor>, off: &mut usize) -> Result<Tensor> {
        let r = x;
        let x = self.n1.forward(x)?;
        let x = self.attn.forward(&x, kc, vc, off)?;
        let x = (x + r)?;
        let r = &x;
        let x = self.n2.forward(&x)?;
        let x = self.l2.forward(&self.l1.forward(&x)?.gelu_erf()?)?;
        Ok((x + r)?)
    }
}

// ---------------------------------------------------------------------------
// Transformer
// ---------------------------------------------------------------------------

struct TF {
    layers: Vec<TLayer>,
    max_period: f64,
    dim: usize,
}

impl TF {
    fn new(cfg: &ModelConfig, vb: &VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(TLayer::new(cfg.dim, cfg.num_heads, cfg.dim_feedforward(), vb, &format!("layers.{}", i))?);
        }
        Ok(Self { layers, max_period: cfg.max_period, dim: cfg.dim })
    }

    fn forward(&self, x: &Tensor, offs: &[usize], kcs: &mut [Option<Tensor>], vcs: &mut [Option<Tensor>], los: &mut [usize]) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let dev = x.device();
        let dt = x.dtype();
        let pos: Vec<f32> = (0..t).flat_map(|ti| offs.iter().map(move |&o| (o + ti) as f32)).collect();
        let pt = Tensor::from_vec(pos, (b, t, 1), dev)?.to_dtype(dt)?;
        let mut h = (x + sin_embedding(&pt, d, self.max_period)?)?;
        for (i, l) in self.layers.iter().enumerate() {
            h = l.forward(&h, &mut kcs[i], &mut vcs[i], &mut los[i])?;
        }
        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// Conditioners
// ---------------------------------------------------------------------------

struct MelC { proj: LinB, dim: usize }
impl MelC {
    fn new(od: usize, vb: &VarBuilder) -> Result<Self> {
        Ok(Self { proj: LinB::new(512, od, vb, "output_proj")?, dim: od })
    }
    fn forward(&self, m: &Tensor) -> Result<(Tensor, Tensor)> {
        let nd = m.dims().len();
        let e = if nd == 3 {
            let (b, t, _) = m.dims3()?;
            let flat = m.reshape((b * t, 512))?;
            self.proj.forward(&flat)?.reshape((b, t, self.dim))?
        } else {
            self.proj.forward(m)?
        };
        let mask = Tensor::ones(e.dims(), e.dtype(), e.device())?;
        Ok((e, mask))
    }
}

pub struct ClsC {
    pub emb: candle_nn::Embedding,
}
impl ClsC {
    fn new(nc: usize, d: usize, vb: &VarBuilder, name: &str) -> Result<Self> {
        let w = vb.get((nc + 1, d), &format!("{}.weight", name))?;
        Ok(Self { emb: candle_nn::Embedding::new(w, d) })
    }
    pub fn tokenize(&self, inp: &[Option<Vec<i64>>], dev: &Device) -> Result<Tensor> {
        let b = inp.len();
        let ml = inp.iter().map(|c| c.as_ref().map_or(0, |v| v.len())).max().unwrap_or(1).max(1);
        let mut data = Vec::with_capacity(b * ml);
        for c in inp {
            match c {
                Some(v) => {
                    for &x in v { data.push(x + 1); }
                    data.resize(data.len() + (ml - v.len()), 0i64);
                }
                None => { data.resize(data.len() + ml, 0i64); }
            }
        }
        Tensor::from_vec(data, (b, ml), dev)
    }
    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let e = self.emb.forward(x)?;
        let mask = Tensor::ones(e.dims(), e.dtype(), e.device())?;
        Ok((e, mask))
    }
}

// ---------------------------------------------------------------------------
// LMModel
// ---------------------------------------------------------------------------

pub struct LMModel {
    emb: ScaledEmb,
    tf: TF,
    on: LN,
    lin: LinNB,
    mc: MelC,
    pub ic: ClsC,
    pub dc: ClsC,
    pub card: usize,
    pub dim: usize,
}

impl LMModel {
    pub fn load<P: AsRef<Path>>(path: P, cfg: &ModelConfig, device: &Device) -> Result<Self> {
        // safetensors::load in candle 0.10 takes a path directly
        let tensors_map = safetensors::load(path.as_ref(), device)?;
        let mut tensors = HashMap::new();
        for (k, v) in &tensors_map {
            let nk = k.replacen("emb.0.", "emb.", 1).replacen("linears.0.", "linear.", 1);
            log::debug!("  tensor {} -> {}", k, nk);
            tensors.insert(nk, v.clone());
        }
        log::info!("Loaded {} tensors", tensors.len());
        for (k, _) in &tensors {
            log::debug!("  available: {}", k);
        }
        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
        Self::new(&vb, cfg)
    }

    pub fn new(vb: &VarBuilder, cfg: &ModelConfig) -> Result<Self> {
        Ok(Self {
            emb: ScaledEmb::new(cfg.card + 1, cfg.dim, &vb.pp("emb"))?,
            tf: TF::new(cfg, &vb.pp("transformer"))?,
            on: LN::new(cfg.dim, 1e-5, vb, "out_norm")?,
            lin: LinNB::new(cfg.dim, cfg.card, vb, "linear")?,
            mc: MelC::new(cfg.dim, &vb.pp("condition_provider.conditioners.self_wav"))?,
            ic: ClsC::new(1000, cfg.dim, &vb.pp("condition_provider.conditioners.instrument_group"), "embed")?,
            dc: ClsC::new(4, cfg.dim, &vb.pp("condition_provider.conditioners.dataset_name"), "embed")?,
            card: cfg.card, dim: cfg.dim,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self, seq: &Tensor, mel: &Tensor, inst: &Tensor, ds: &Tensor,
        first: bool, kcs: &mut [Option<Tensor>], vcs: &mut [Option<Tensor>], los: &mut [usize],
    ) -> Result<Tensor> {
        let (b, s) = seq.dims2()?;
        let mut h = self.emb.forward(seq)?;
        let mut pl = 0;
        if first {
            let (mc, _) = self.mc.forward(mel)?;
            let (ic, _) = self.ic.forward(inst)?;
            let (dc, _) = self.dc.forward(ds)?;
            h = Tensor::cat(&[&ic, &dc, &mc, &h], 1)?;
            pl = h.dim(1)? - s;
        }
        let offs = &[0usize];
        h = self.tf.forward(&h, offs, kcs, vcs, los)?;
        h = if pl > 0 { self.on.forward(&h.narrow(1, pl, s)?)? } else { self.on.forward(&h)? };
        self.lin.forward(&h)
    }

    pub fn generate(
        &self, mel: &Tensor, inst: &Tensor, ds: &Tensor,
        max_len: usize, sample: bool, temp: f64, _top_k: usize, _top_p: f64,
    ) -> Result<Vec<i64>> {
        let dev = mel.device();
        let eos = 1i64;
        let init = self.card as i64;
        let nl = self.tf.layers.len();
        let mut kcs: Vec<Option<Tensor>> = vec![None; nl];
        let mut vcs: Vec<Option<Tensor>> = vec![None; nl];
        let mut los: Vec<usize> = vec![0; nl];
        let mut out = Vec::new();

        for step in 0..max_len {
            let first = step == 0;
            let seq_data: Vec<i64> = if first {
                let mut v = vec![init];
                v.extend_from_slice(&out);
                v
            } else if out.is_empty() {
                vec![init]
            } else {
                let last = out[step - 1];
                if last == eos { break; }
                vec![last]
            };
            let sl = seq_data.len();
            let seq = Tensor::from_vec(seq_data, (1, sl), dev)?;

            let logits = self.forward(&seq, mel, inst, ds, first, &mut kcs, &mut vcs, &mut los)?;
            let logits = logits.narrow(1, sl - 1, 1)?.squeeze(1)?;
            let logits = logits.to_dtype(DType::F32)?;

            // (No OOV mask needed: the linear layer outputs card logits,
            //  so indices >= card already have no logit.)
            let logits = logits;

            let next = if sample && temp > 0.0 {
                let temp_t = Tensor::new(temp as f32, dev)?;
                let probs = candle_nn::ops::softmax(&(&logits / temp_t)?, 1)?;
                sample_token(&probs, dev)?.to_scalar::<i64>()?
            } else {
                logits.argmax(1)?.squeeze(0)?.to_dtype(DType::I64)?.to_scalar::<i64>()?
            };
            out.push(next);
        }
        Ok(out)
    }
}

fn sample_token(probs: &Tensor, dev: &Device) -> Result<Tensor> {
    let b = probs.dim(0)?;
    let vs = probs.dim(1)?;
    let mut result = Vec::with_capacity(b);
    for i in 0..b {
        let row = probs.narrow(0, i, 1)?.squeeze(0)?.to_vec1::<f32>()?;
        let r: f32 = fastrand::f32();
        let mut cum = 0.0f32;
        let mut chosen = (vs - 1) as i64;
        for (j, &v) in row.iter().enumerate() {
            cum += v;
            if r < cum { chosen = j as i64; break; }
        }
        result.push(chosen);
    }
    Tensor::from_vec(result, (b, 1), dev)
}
