use candle_core::cuda_backend::cudarc::driver::sys::CUstream;
use candle_core::Module;
use candle_core::{Device, Shape, Tensor};
use common::cuda_ext::get_tensor_cuda_device_ptr;
use common::ffi::{get_scalar_type, CShapeView, CTensorView, ScalarType};
use common::{DefaultTensorCreator, TensorCreator};
use libc::c_float;

use std::os::raw::{c_uint, c_void};

#[derive(Debug)]
#[repr(C)]
pub struct RmsNormKernelParams {
    pub stream: CUstream,
    pub epsilon: f32,
    pub hidden_size: i32,
    pub num_tokens: i32,
    pub dtype: ScalarType,
}

extern "C" {
    // void cuda_oneflow_rms_norm(CTensorView x, CTensorView weight, CShapeView normalized_shape, float epsilon,
    //     cudaStream_t stream, CTensorView inv_rms, CTensorView y)

    fn vllm_fused_add_rms_norm(
        inout: *mut c_void,
        residual: *mut c_void,
        weight: *const c_void,
        params: RmsNormKernelParams,
    );
    fn vllm_rms_norm(
        output: *mut c_void,
        input: *const c_void,
        weight: *const c_void,
        params: RmsNormKernelParams,
    );
}

#[derive(Clone, Debug)]
pub struct RmsNorm {
    weight: Tensor,
    normalized_shape: Shape,
    eps: f64,
}

impl RmsNorm {
    // pub fn new<S: Into<Shape>>(s: S, eps: f64, elementwise_affine: bool) -> Self {
    //     Self {
    //         weight: None,
    //         normalized_shape: s.into(),
    //         eps,
    //         elementwise_affine,
    //     }
    // }
    pub fn load<S: Into<Shape>>(
        s: S,
        eps: f64,
        vb: candle_nn::VarBuilder,
    ) -> candle_core::Result<Self> {
        let normalized_shape = s.into();
        let weight = vb.get_with_hints(
            normalized_shape.clone(),
            "weight",
            candle_nn::Init::Const(1.),
        )?;
        Ok(Self {
            weight,
            normalized_shape,
            eps,
        })
    }
    pub fn forward_residual(
        &self,
        xs: Tensor,
        residual: Tensor,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let hidden_size = *xs.dims().last().unwrap();
        let num_tokens = xs.elem_count() / hidden_size;
        let params = RmsNormKernelParams {
            stream: std::ptr::null_mut(),
            epsilon: self.eps as f32,
            hidden_size: hidden_size as i32,
            num_tokens: num_tokens as i32,
            dtype: get_scalar_type(xs.dtype()),
        };
        let weight_data = get_tensor_cuda_device_ptr(&self.weight)?;
        let xs_data = get_tensor_cuda_device_ptr(&xs)?;
        let residual_data = get_tensor_cuda_device_ptr(&residual)?;
        unsafe {
            vllm_fused_add_rms_norm(
                xs_data.as_ffi_ptr(),
                residual_data.as_ffi_ptr(),
                weight_data.as_ffi_ptr(),
                params,
            );
        }
        Ok((xs, residual))
    }

    pub fn forward_<F: TensorCreator>(
        &self,
        xs: &Tensor,
        tensor_creator: &mut F,
    ) -> candle_core::Result<Tensor> {
        let hidden_size = *xs.dims().last().unwrap();
        let num_tokens = xs.elem_count() / hidden_size;
        let params = RmsNormKernelParams {
            stream: std::ptr::null_mut(),
            epsilon: self.eps as f32,
            hidden_size: hidden_size as i32,
            num_tokens: num_tokens as i32,
            dtype: get_scalar_type(xs.dtype()),
        };
        let weight_data = get_tensor_cuda_device_ptr(&self.weight)?;
        let xs_data = get_tensor_cuda_device_ptr(&xs)?;
        let out = tensor_creator.new(xs.shape(), xs.dtype(), xs.device(), false)?;
        let out_data = get_tensor_cuda_device_ptr(&out)?;
        unsafe {
            vllm_rms_norm(
                out_data.as_ffi_ptr(),
                xs_data.as_ffi_ptr(),
                weight_data.as_ffi_ptr(),
                params,
            );
        }
        Ok(out)
    }
    pub fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut default_creator = DefaultTensorCreator {};
        self.forward_(xs, &mut default_creator)
    }
}

// #[test]
// fn test_rms_norm() -> candle_core::Result<()> {
//     let device = Device::new_cuda(0)?;
//     let a = Tensor::new(
//         &[
//             -0.16046895,
//             -1.03667831,
//             -0.34974465,
//             0.26505867,
//             -1.24111986,
//             -0.53806001,
//             1.72426331,
//             0.43572459,
//             -0.77390957,
//             -0.42610624,
//             0.16398858,
//             -1.35760343,
//             1.07541728,
//             0.11008703,
//             0.26361224,
//             -0.48663723,
//         ],
//         &device,
//     )?
//     .reshape((2, 2, 2, 2))?;
//     let rms_norm = RmsNorm::new(2, 1e-5, true);
//     let r = rms_norm.forward(&a)?;
//     println!("r:{}", r.to_string());

//     Ok(())
// }
