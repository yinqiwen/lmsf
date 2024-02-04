use std::ops::Deref;

use candle_core::{
    cuda_backend::cudarc::driver::{
        result::{memcpy_dtod_async, memcpy_dtoh_async, memcpy_htod_async},
        CudaFunction, CudaSlice, CudaStream, DevicePtr, DeviceRepr, LaunchAsync, LaunchConfig,
    },
    CpuStorage, CudaDevice, DType, Device, IndexOp, Layout, Shape, Storage, Tensor, D,
};

fn scatter_add() -> candle_core::Result<()> {
    // let device = Device::new_cuda(0)?;
    let device = Device::new_cuda(0)?;
    let logits_idx_end = 32000_usize;
    let logits_idx = Tensor::arange(0_u32, logits_idx_end as u32, &device)?.reshape((1, 32000))?;
    let logits_idx_inv = Tensor::zeros_like(&logits_idx)?;
    let src = Tensor::arange(0_u32, logits_idx_end as u32, logits_idx.device())?
        .expand(logits_idx.shape())?
        .contiguous()?;
    let start = std::time::Instant::now();
    // let logits_idx_inv = candle_ext::F::scatter(&logits_idx_inv, &logits_idx, &src, D::Minus1)?;
    tops::cuda_scatter(
        &logits_idx_inv,
        &logits_idx,
        &src,
        D::Minus1,
        std::ptr::null_mut(),
    );
    println!(
        "1 scatter cost {:?}/{}",
        start.elapsed(),
        logits_idx_inv.to_string()
    );
    // let logits_idx_inv = candle_ext::F::scatter(&logits_idx_inv, &logits_idx, &src, D::Minus1)?;
    tops::cuda_scatter(
        &logits_idx_inv,
        &logits_idx,
        &src,
        D::Minus1,
        std::ptr::null_mut(),
    );
    match device {
        Device::Cuda(cuda_dev) => {
            cuda_dev.synchronize();
        }
        _ => {}
    }
    println!(
        "2 scatter cost {:?}/{}",
        start.elapsed(),
        logits_idx_inv.to_string()
    );
    Ok(())
}
fn main() -> candle_core::Result<()> {
    let a = Tensor::new(&[1.0], &Device::Cpu)?;
    let a = a.reshape((1, 1))?;
    let start = std::time::Instant::now();
    let a = a.repeat((1, 32000))?;
    println!(
        "cpu tensor repeat to {:?} cost {:?}",
        a.shape(),
        start.elapsed()
    );

    let device = Device::new_cuda(0)?;

    let b = Tensor::new(&[1.0], &device)?;
    let b = b.reshape((1, 1))?;
    let start = std::time::Instant::now();
    let b = b.repeat((1, 32000))?;
    println!(
        "cuda tensor repeat to {:?} cost {:?}",
        b.shape(),
        start.elapsed()
    );
    scatter_add()?;
    scatter_add()?;
    Ok(())
}
