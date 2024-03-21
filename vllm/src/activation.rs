use candle::cuda_backend::cudarc::driver::sys::CUstream;
use candle::Module;
use candle::{Device, Shape, Tensor};
use common::cuda_ext::get_tensor_cuda_device_ptr;
use common::ffi::{get_scalar_type, CShapeView, CTensorView, ScalarType};
use common::{DefaultTensorCreator, TensorCreator};
use libc::c_float;

use std::os::raw::{c_uint, c_void};

#[derive(Debug)]
#[repr(C)]
pub struct SiluMulKernelParams {
    pub stream: CUstream,
    pub d: i32,
    pub num_tokens: i32,
    pub dtype: ScalarType,
}

extern "C" {
    fn vllm_silu_and_mul(out: *mut c_void, input: *mut c_void, params: SiluMulKernelParams);
}

pub fn silu_and_mul(input: &Tensor) -> candle::Result<Tensor> {
    // let last_dim_size = *input.dims().last().unwrap();
    // let num_tokens = input.elem_count() / last_dim_size;
    // let d = last_dim_size / 2;
    // let params = SiluMulKernelParams {
    //     stream: std::ptr::null_mut(),
    //     d: d as i32,
    //     num_tokens: num_tokens as i32,
    //     dtype: get_scalar_type(input.dtype()),
    // };
    // let mut output_shape = Vec::from(input.dims());
    // let last_idx = output_shape.len() - 1;
    // output_shape[last_idx] = d;
    // let output_shape = Shape::from_dims(&output_shape);
    // let out = Tensor::zeros(output_shape, input.dtype(), input.device())?;

    // let input_data = get_tensor_cuda_device_ptr(input)?;
    // let out_data = get_tensor_cuda_device_ptr(&out)?;
    // unsafe {
    //     vllm_silu_and_mul(out_data.as_ffi_ptr(), input_data.as_ffi_ptr(), params);
    // }

    let mut default_creator = DefaultTensorCreator {};
    silu_and_mul_(input, &mut default_creator)
}

pub fn silu_and_mul_<F: TensorCreator>(
    input: &Tensor,
    tensor_creator: &mut F,
) -> candle::Result<Tensor> {
    let last_dim_size = *input.dims().last().unwrap();
    let num_tokens = input.elem_count() / last_dim_size;
    let d = last_dim_size / 2;
    let params = SiluMulKernelParams {
        stream: std::ptr::null_mut(),
        d: d as i32,
        num_tokens: num_tokens as i32,
        dtype: get_scalar_type(input.dtype()),
    };
    let mut output_shape = Vec::from(input.dims());
    let last_idx = output_shape.len() - 1;
    output_shape[last_idx] = d;
    let output_shape = Shape::from_dims(&output_shape);
    let out = tensor_creator.new(output_shape, input.dtype(), input.device(), false)?;
    // let out = Tensor::zeros(output_shape, input.dtype(), input.device())?;

    let input_data = get_tensor_cuda_device_ptr(input)?;
    let out_data = get_tensor_cuda_device_ptr(&out)?;
    unsafe {
        vllm_silu_and_mul(out_data.as_ffi_ptr(), input_data.as_ffi_ptr(), params);
    }
    Ok(out)
}
