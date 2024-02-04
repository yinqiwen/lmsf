use candle_core::{DType, Device, Tensor};

pub trait AttentionBias {
    // """
    // Materializes the bias as a `torch.Tensor`. This is very slow
    // and we don't attempt to make it fast. Only use for debugging/testing.

    // Shape should be like `[*, q_seqlen, k_seqlen]`
    // """
    fn materialize(&self, dtype: DType, device: &Device) -> candle_core::Result<Tensor>;
}

#[derive(Clone)]
struct _SeqLenInfo {
    seqstart: Tensor,
    max_seqlen: u32,
    min_seqlen: u32,

    seqstart_py: Vec<u32>,
}

impl _SeqLenInfo {
    pub fn from_seqlens<'a>(seqlens: impl Iterator<Item = &'a u32>) -> candle_core::Result<Self> {
        let mut seqstart_py = vec![0];
        let mut max_seqlen: i32 = -1;
        let mut min_seqlen: i32 = -1;
        for seqlen in seqlens {
            min_seqlen = if min_seqlen != -1 {
                *seqlen as i32
            } else {
                std::cmp::min(min_seqlen, *seqlen as i32)
            };
            max_seqlen = std::cmp::max(max_seqlen, *seqlen as i32);
            let append_v = *seqstart_py.last().unwrap() + seqlen;
            seqstart_py.push(append_v);
        }
        let device = candle_core::Device::Cpu;
        let seqstart = Tensor::from_vec(seqstart_py.clone(), seqstart_py.len(), &device)?;
        Ok(Self {
            seqstart,
            max_seqlen: max_seqlen as u32,
            min_seqlen: min_seqlen as u32,
            seqstart_py,
        })
    }
}

pub struct BlockDiagonalMask {
    q_seqinfo: _SeqLenInfo,
    k_seqinfo: _SeqLenInfo,
    _batch_sizes: Option<Vec<usize>>,
}

impl BlockDiagonalMask {
    pub fn from_seqlens(
        q_seqlen: Vec<u32>,
        kv_seqlen: Option<Vec<u32>>,
    ) -> candle_core::Result<Self> {
        let q_seqinfo = _SeqLenInfo::from_seqlens(q_seqlen.iter())?;

        let k_seqinfo = if kv_seqlen.is_none() || &q_seqlen == kv_seqlen.as_ref().unwrap() {
            q_seqinfo.clone()
        } else {
            _SeqLenInfo::from_seqlens(kv_seqlen.unwrap().iter())?
        };

        Ok(Self {
            q_seqinfo,
            k_seqinfo,
            _batch_sizes: None,
        })
    }
    pub fn make_local_attention(self, window_size: usize) -> BlockDiagonalCausalLocalAttentionMask {
        BlockDiagonalCausalLocalAttentionMask::from(self, window_size)
    }
}

impl AttentionBias for BlockDiagonalMask {
    fn materialize(&self, dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
        todo!()
    }
}

pub struct BlockDiagonalCausalLocalAttentionMask {
    block: BlockDiagonalMask,
    _window_size: usize,
}
impl BlockDiagonalCausalLocalAttentionMask {
    pub fn from(block: BlockDiagonalMask, window: usize) -> Self {
        Self {
            block,
            _window_size: window,
        }
    }
}

impl AttentionBias for BlockDiagonalCausalLocalAttentionMask {
    fn materialize(&self, dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
        todo!()
    }
}

pub struct LowerTriangularMaskWithTensorBias {
    bias: Tensor,
}

impl LowerTriangularMaskWithTensorBias {
    pub fn from(bias: Tensor) -> Self {
        Self { bias }
    }
}

impl AttentionBias for LowerTriangularMaskWithTensorBias {
    fn materialize(&self, dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
        todo!()
    }
}

pub fn memory_efficient_attention_forward(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    attn_bias: &Box<dyn AttentionBias>,
    p: f32,
    scale: f32,
) {
    todo!()
}
