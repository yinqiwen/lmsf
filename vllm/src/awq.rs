
use candle::{cuda_backend::cudarc::driver::sys::CUstream, Tensor};
use common::{
    cuda_ext::get_tensor_cuda_device_ptr, TensorCreator,
};
use libc::c_void;

#[repr(C)]
pub struct AWQDequntizeParams {
    pub in_c: i32,
    pub qout_c: i32,
    pub thx: i32,
    pub thy: i32,
    pub scaling_factors_size: i32,
}

#[repr(C)]
#[derive(Debug)]
pub struct AWQGemmParams {
    pub num_in_feats: i32,
    pub num_in_channels: i32,
    pub num_out_feats: i32,
    pub num_out_channels: i32,
    pub split_k_iters: i32,
    pub scaling_factors_size: i32,
}

extern "C" {
    fn vllm_awq_dequantize(
        kernel_data: *mut c_void,
        scaling_factors_data: *mut c_void,
        zero_data: *mut c_void,
        de_kernel_data: *mut c_void,
        stream: CUstream,
        params: AWQDequntizeParams,
    );

    fn vllm_awq_gemm(
        in_feats_data: *mut c_void,
        kernel_data: *mut c_void,
        scaling_factors_data: *mut c_void,
        zero_data: *mut c_void,
        out_feats_data: *mut c_void,
        stream: CUstream,
        params: AWQGemmParams,
    );

}

pub fn awq_dequantize<F: TensorCreator>(
    kernel: &Tensor,
    scaling_factors: &Tensor,
    zeros: &Tensor,
    thx: i32,
    thy: i32,
    tensor_creator: &mut F,
) -> candle::Result<Tensor> {
    //   int in_c = _kernel.size(0);
    //   int qout_c = _kernel.size(1);
    let in_c = kernel.dims()[0] as i32;
    let qout_c = kernel.dims()[1] as i32;
    let scaling_factors_size = scaling_factors.dims()[0] as i32;
    let params = AWQDequntizeParams {
        in_c,
        qout_c,
        thx,
        thy,
        scaling_factors_size,
    };

    let out_c = kernel.dims()[0] * 8;
    let de_kernel = tensor_creator.new(
        (in_c as usize, out_c),
        scaling_factors.dtype(),
        scaling_factors.device(),
        false,
    )?;

    let kernel_data = get_tensor_cuda_device_ptr(kernel)?;
    let scaling_factors_data = get_tensor_cuda_device_ptr(scaling_factors)?;
    let zeros_data = get_tensor_cuda_device_ptr(zeros)?;
    let de_kernel_data = get_tensor_cuda_device_ptr(&de_kernel)?;
    unsafe {
        vllm_awq_dequantize(
            kernel_data.as_ffi_ptr(),
            scaling_factors_data.as_ffi_ptr(),
            zeros_data.as_ffi_ptr(),
            de_kernel_data.as_ffi_ptr(),
            common::cuda_ext::cuda_get_default_stream(kernel.device())?,
            params,
        );
    }
    Ok(de_kernel)
}

pub fn awq_gemm<F: TensorCreator>(
    in_feats: &Tensor,
    kernel: &Tensor,
    scaling_factors: &Tensor,
    zeros: &Tensor,
    split_k_iters: i32,
    tensor_creator: &mut F,
) -> candle::Result<Tensor> {
    let num_in_feats = in_feats.dims()[0];
    let num_in_channels = in_feats.dims()[1];
    let out_feats_last_dim = kernel.dims()[1] * 8;

    let out_feats = tensor_creator.new(
        (split_k_iters as usize, num_in_feats, out_feats_last_dim),
        in_feats.dtype(),
        in_feats.device(),
        false,
    )?;

    let num_out_feats = num_in_feats;
    let num_out_channels = out_feats_last_dim;
    let scaling_factors_size = scaling_factors.dims()[0] as i32;

    let params = AWQGemmParams {
        num_in_feats: num_in_feats as i32,
        num_in_channels: num_in_channels as i32,
        num_out_feats: num_out_feats as i32,
        num_out_channels: num_out_channels as i32,
        split_k_iters,
        scaling_factors_size,
    };

    let kernel_data = get_tensor_cuda_device_ptr(kernel)?;
    let scaling_factors_data = get_tensor_cuda_device_ptr(scaling_factors)?;
    let zeros_data = get_tensor_cuda_device_ptr(zeros)?;
    let in_feats_data = get_tensor_cuda_device_ptr(in_feats)?;
    let out_feats_data = get_tensor_cuda_device_ptr(&out_feats)?;
    unsafe {
        vllm_awq_gemm(
            in_feats_data.as_ffi_ptr(),
            kernel_data.as_ffi_ptr(),
            scaling_factors_data.as_ffi_ptr(),
            zeros_data.as_ffi_ptr(),
            out_feats_data.as_ffi_ptr(),
            common::cuda_ext::cuda_get_default_stream(kernel.device())?,
            params,
        );
    }

    out_feats.sum(0)
}
