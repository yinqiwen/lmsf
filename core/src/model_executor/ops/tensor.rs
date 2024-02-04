use anyhow::Result;
use candle_core::{scalar::TensorOrScalar, Device, Tensor, WithDType};
use candle_transformers::models::bert::DTYPE;

fn pad_to_max<D: WithDType>(mut x: Vec<D>, max_len: usize, pad: D) -> Vec<D> {
    assert!(x.len() <= max_len);
    x.extend([pad].repeat(max_len - x.len()));

    x
}

pub fn make_tensor_with_pad<D: WithDType>(
    device: &Device,
    x: Vec<Vec<D>>,
    max_len: usize,
    pad: D,
) -> Result<Tensor> {
    let padded_x: Vec<_> = x.into_iter().map(|v| pad_to_max(v, max_len, pad)).collect();
    let tensor = Tensor::new(padded_x, device)?;

    Ok(tensor)
}

pub fn masked_fill<D: WithDType>(
    on_false: &Tensor,
    mask: &Tensor,
    on_true: D,
) -> candle_core::Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

pub fn masked_fill_neg_inf(on_false: &Tensor, mask: &Tensor) -> candle_core::Result<Tensor> {
    let shape = mask.shape();
    let on_true =
        match on_false.dtype() {
            candle_core::DType::U8 => {
                Tensor::new(0_u8, on_false.device())?.broadcast_as(shape.dims())?
            }
            candle_core::DType::F16 => Tensor::new(half::f16::NEG_INFINITY, on_false.device())?
                .broadcast_as(shape.dims())?,
            candle_core::DType::F32 => {
                Tensor::new(f32::NEG_INFINITY, on_false.device())?.broadcast_as(shape.dims())?
            }
            candle_core::DType::BF16 => Tensor::new(half::bf16::NEG_INFINITY, on_false.device())?
                .broadcast_as(shape.dims())?,
            candle_core::DType::F64 => {
                Tensor::new(f64::NEG_INFINITY, on_false.device())?.broadcast_as(shape.dims())?
            }
            _ => {
                candle_core::bail!(
                    "not supported dtype:{:?} for masked_fill_neg_inf",
                    on_false.dtype()
                );
            }
        };
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}
