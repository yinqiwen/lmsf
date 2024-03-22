
use candle::{cuda_backend::cudarc::driver::sys::CUstream, Tensor};
use common::{
    cuda_ext::get_tensor_cuda_device_ptr,
};
use libc::c_void;

#[repr(C)]
pub struct SqueezeLLMParams {
    pub height: i32,
    pub width: i32,
    pub batch: i32,
    pub vec_height: i32,
}

extern "C" {
    fn vllm_squeezellm_gemm(
        vec: *mut c_void,
        mat: *mut c_void,
        mul: *mut c_void,
        lookup_table: *mut c_void,
        stream: CUstream,
        params: SqueezeLLMParams,
    );
}

pub fn squeezellm_gemm(
    vec: &Tensor,
    mat: &Tensor,
    mul: &Tensor,
    lookup_table: &Tensor,
) -> candle::Result<()> {
    let height = mat.dims()[0] as i32;
    let width = mat.dims()[1] as i32;
    let batch = vec.dims()[0] as i32;
    let vec_height = vec.dims()[1] as i32;

    let params = SqueezeLLMParams {
        height,
        width,
        batch,
        vec_height,
    };

    let vec_data = get_tensor_cuda_device_ptr(vec)?;
    let mat_data = get_tensor_cuda_device_ptr(mat)?;
    let mul_data = get_tensor_cuda_device_ptr(mul)?;
    let lookup_table_data = get_tensor_cuda_device_ptr(lookup_table)?;

    unsafe {
        vllm_squeezellm_gemm(
            vec_data.as_ffi_ptr(),
            mat_data.as_ffi_ptr(),
            mul_data.as_ffi_ptr(),
            lookup_table_data.as_ffi_ptr(),
            common::cuda_ext::cuda_get_default_stream(vec.device())?,
            params,
        );
    }
    Ok(())
}
