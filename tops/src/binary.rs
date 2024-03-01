use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DeviceRepr, LaunchAsync};
use candle_core::cuda_backend::WrapErr;
use candle_core::{
    backend::BackendStorage, cuda_backend::cudarc::driver::LaunchConfig, shape::Dim, CpuStorage,
    CudaStorage, DType, Layout, Shape, Storage,
};
use candle_core::{Device, Tensor};
use common::cuda_ext::get_tensor_cuda_device_ptr;

use std::ops::Deref;

use common::{DefaultTensorCreator, TensorCreator};

fn kernel_name(root: &str, lhs_dtype: DType, rhs_dtype: DType) -> String {
    let lhs_dtype = lhs_dtype.as_str();
    let rhs_dtype = rhs_dtype.as_str();
    format!("{root}_{lhs_dtype}_{rhs_dtype}")
}

fn cuda_binary_op(a: &Tensor, b: &Tensor, c: &Tensor, op: &str) -> candle_core::Result<()> {
    let device = match a.device() {
        Device::Cuda(cuda_dev) => cuda_dev,
        _ => {
            candle_core::bail!("unexpected device")
        }
    };
    let shape = a.shape();
    let func =
        device.get_or_load_func(&kernel_name(op, a.dtype(), b.dtype()), candle_patch::BINARY)?;
    let dims = shape.dims();
    let elem_count = shape.elem_count();
    let dims_and_strides = device
        .htod_copy([dims, a.stride(), b.stride()].concat())
        .w()?;
    let cfg = LaunchConfig::for_num_elems(elem_count as u32);
    let lhs = get_tensor_cuda_device_ptr(a)?;
    let rhs = get_tensor_cuda_device_ptr(b)?;
    let out = get_tensor_cuda_device_ptr(c)?;
    let params = (elem_count, dims.len(), &dims_and_strides, lhs, rhs, out);
    unsafe { func.launch(cfg, params) }.w()?;
    Ok(())
}

pub fn cuda_tensor_mul(lhs: &Tensor, rhs: &Tensor, dtype: DType) -> candle_core::Result<Tensor> {
    let mut default_creator = DefaultTensorCreator {};
    cuda_tensor_mul_(lhs, rhs, dtype, &mut default_creator)
}

pub fn cuda_tensor_mul_<F: TensorCreator>(
    lhs: &Tensor,
    rhs: &Tensor,
    dtype: DType,
    tensor_creator: &mut F,
) -> candle_core::Result<Tensor> {
    if lhs.dtype() == rhs.dtype() {
        return lhs.mul(rhs);
    }
    let out = tensor_creator.new(lhs.shape(), dtype, lhs.device(), false)?;
    cuda_binary_op(lhs, rhs, &out, "bmul")?;
    Ok(out)
}

pub fn cuda_tensor_broadcast_mul(
    lhs: &Tensor,
    rhs: &Tensor,
    dtype: DType,
) -> candle_core::Result<Tensor> {
    let mut default_creator = DefaultTensorCreator {};
    cuda_tensor_broadcast_mul_(lhs, rhs, dtype, &mut default_creator)
}

pub fn cuda_tensor_broadcast_mul_<F: TensorCreator>(
    lhs: &Tensor,
    rhs: &Tensor,
    dtype: DType,
    tensor_creator: &mut F,
) -> candle_core::Result<Tensor> {
    if lhs.dtype() == rhs.dtype() {
        return lhs.broadcast_mul(rhs);
    }
    let lhs = lhs.broadcast_as(rhs.shape())?;
    let out = tensor_creator.new(lhs.shape(), dtype, lhs.device(), false)?;
    cuda_binary_op(&lhs, rhs, &out, "bmul")?;
    Ok(out)
}

#[test]
fn test_mul() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let a = Tensor::new(&[2.0], &device)?.to_dtype(DType::F16)?;
    let b = Tensor::new(&[1_i64, 2, 3, 4, 5], &device)?;

    let c = a.broadcast_mul(&b.to_dtype(DType::F32)?.to_dtype(DType::F16)?)?;
    println!("{}", c.to_string());

    let a = a.broadcast_as(b.shape())?;
    let c = cuda_tensor_mul(&a, &b, a.dtype())?;
    println!("{}", c.to_string());

    Ok(())
}
