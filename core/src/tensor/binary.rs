use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DeviceRepr, LaunchAsync};
use candle_core::cuda_backend::WrapErr;
use candle_core::{
    backend::BackendStorage, cuda_backend::cudarc::driver::LaunchConfig, shape::Dim, CpuStorage,
    CudaStorage, DType, Layout, Shape, Storage,
};
use candle_core::{Device, Tensor};
use common::cuda_ext::get_tensor_cuda_device_ptr;

use std::ops::Deref;

fn kernel_name(root: &str, dtype: DType) -> String {
    let dtype = dtype.as_str();
    format!("{root}_{dtype}")
}

fn cuda_inplace_binary_op(a: &Tensor, b: &Tensor, op: &str) -> candle_core::Result<()> {
    let device = match a.device() {
        Device::Cuda(cuda_dev) => cuda_dev,
        _ => {
            candle_core::bail!("unexpected device")
        }
    };
    let shape = a.shape();
    let func = device.get_or_load_func(&kernel_name(op, a.dtype()), candle_kernels::BINARY)?;
    let dims = shape.dims();
    let elem_count = shape.elem_count();
    let dims_and_strides = device
        .htod_copy([dims, a.stride(), b.stride()].concat())
        .w()?;
    let cfg = LaunchConfig::for_num_elems(elem_count as u32);
    let lhs = get_tensor_cuda_device_ptr(a)?;
    let rhs = get_tensor_cuda_device_ptr(b)?;
    let out = lhs.clone();
    let params = (elem_count, dims.len(), &dims_and_strides, lhs, rhs, out);
    unsafe { func.launch(cfg, params) }.w()?;
    Ok(())
}

pub fn cuda_div_(a: &Tensor, b: &Tensor) -> candle_core::Result<()> {
    cuda_inplace_binary_op(a, b, "bdiv")
}
pub fn cuda_sub_(a: &Tensor, b: &Tensor) -> candle_core::Result<()> {
    cuda_inplace_binary_op(a, b, "bsub")
}
pub fn cuda_add_(a: &Tensor, b: &Tensor) -> candle_core::Result<()> {
    cuda_inplace_binary_op(a, b, "badd")
}

#[test]
fn test_inplace_div() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let a = Tensor::new(&[[0.3810, 1.2774, -0.2972, -0.3719, 0.4637]], &device)?;
    let b = Tensor::new(&[0.5, 0.5, 0.5, 0.5, 0.5], &device)?;

    cuda_div_(&a, &b)?;
    println!("afer inplace a div:{}", a.to_string());
    Ok(())
}

#[test]
fn test_inplace_sub() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let a = Tensor::new(&[[0.3810, 1.2774, -0.2972, -0.3719, 0.4637]], &device)?;
    let b = Tensor::new(&[0.5, 0.5, 0.5, 0.5, 0.5], &device)?;

    cuda_sub_(&a, &b)?;
    println!("afer inplace a sub:{}", a.to_string());
    Ok(())
}
