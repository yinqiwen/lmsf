use candle::cuda_backend::cudarc::driver::sys::CUstream;
use candle::{CpuStorage, CudaStorage, DType, Device, Layout, Shape, Tensor};
use common::{
    ffi::get_scalar_type,
    ffi::{CTensorView, ScalarType},
};
use libc::c_ulonglong;

use std::os::raw::{c_uint, c_void};

use crate::common::get_column_major_dim;
use crate::reset_random_seed;

extern "C" {
    fn cuda_create_exponential_tensor(lambd: f32, stream: CUstream, output: CTensorView);
}

pub fn cuda_tensor_exponential(t: &Tensor, lambd: f32, stream: CUstream) -> candle::Result<()> {
    let output_view = common::ffi::CTensorView::from(t, false)?;
    unsafe {
        cuda_create_exponential_tensor(lambd, stream, output_view);
    };
    Ok(())
}

#[test]
fn test_sort() -> candle::Result<()> {
    let device = Device::new_cuda(0)?;
    let a = Tensor::zeros(8, DType::F32, &device)?;
    let stream: CUstream = std::ptr::null_mut();

    reset_random_seed(0);
    cuda_tensor_exponential(&a, 1.0, stream)?;
    println!("out:{:?}", a.to_string());

    Ok(())
}
