use std::ops::Deref;

use anyhow::Result;
use candle::{
    cuda_backend::cudarc::driver::{CudaSlice, DevicePtr},
    Storage, Tensor,
};

// mod attention;
mod attn_bias;
// pub mod cache;
// mod cuda_utils;
// mod macros;
// mod pos_encoding;
// pub mod inplace;
// mod sort;
// mod inplace_ops;
pub mod tensor;

// pub use inplace_ops::cuda_inplace_div;

// pub use attention::PagedAttentionOps;
pub use attn_bias::{
    memory_efficient_attention_forward, AttentionBias, BlockDiagonalMask,
    LowerTriangularMaskWithTensorBias,
};
// pub use pos_encoding::PosEncoding;

fn get_tensor_cuda_slice_ptr_int(tensor: &Tensor) -> Result<i64> {
    let (storage, _) = tensor.storage_and_layout();
    let data = match storage.deref() {
        // I think I need to leak the CudaSlice here so that it does not get freed by candle.
        // something like this, except `.leak()` requires ownership: `cuda_storage.as_cuda_slice::<f32>()?.leak()`
        Storage::Cuda(cuda_storage) => {
            let ptr = cuda_storage.as_cuda_slice::<i64>()?.device_ptr();

            unsafe {
                let u8_value = std::ptr::read(ptr as *const u64);
                u8_value as i64
            }
        }
        _ => unreachable!("unexpected storage type"),
    };
    Ok(data)
}
