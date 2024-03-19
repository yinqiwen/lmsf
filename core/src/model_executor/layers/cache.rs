use candle_core::{Device, Result, Tensor};

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub struct Cache {
    masks: Arc<Mutex<HashMap<usize, Tensor>>>,
    // pub use_kv_cache: bool,
    // #[allow(clippy::type_complexity)]
    // kvs: Arc<Mutex<Vec<Option<(Tensor, Tensor)>>>>,
    // cos: Tensor,
    // sin: Tensor,
    device: Device,
}

impl Cache {
    pub fn new(device: &Device) -> Result<Self> {
        // precompute freqs_cis
        // let n_elem = config.hidden_size / config.num_attention_heads;
        // let theta: Vec<_> = (0..n_elem)
        //     .step_by(2)
        //     .map(|i| 1f32 / config.rope_theta.powf(i as f32 / n_elem as f32))
        //     .collect();
        // let theta = Tensor::new(theta.as_slice(), device)?;
        // let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
        //     .to_dtype(DType::F32)?
        //     .reshape((MAX_SEQ_LEN, 1))?
        //     .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        // // This is different from the paper, see:
        // // https://github.com/huggingface/transformers/blob/6112b1c6442aaf7affd2b0676a1cd4eee30c45cf/src/transformers/models/llama/modeling_llama.py#L112
        // let idx_theta = Tensor::cat(&[&idx_theta, &idx_theta], D::Minus1)?;
        // let cos = idx_theta.cos()?.to_dtype(dtype)?;
        // let sin = idx_theta.sin()?.to_dtype(dtype)?;
        Ok(Self {
            masks: Arc::new(Mutex::new(HashMap::new())),
            // use_kv_cache,
            // kvs: Arc::new(Mutex::new(vec![None; config.num_hidden_layers])),
            device: device.clone(),
            // cos,
            // sin,
        })
    }

    pub fn mask(&self, t: usize) -> Result<Tensor> {
        let mut masks = self.masks.lock().unwrap();
        if let Some(mask) = masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (t, t), &self.device)?;
            masks.insert(t, mask.clone());
            Ok(mask)
        }
    }
}
