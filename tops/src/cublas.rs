use std::sync::Arc;

use candle_core::{
    cuda_backend::{cudarc::driver::sys::CUstream, DeviceId},
    DType, Device, Tensor,
};
use common::ffi::{CTensorView, ScalarType};
use libc::{c_int, c_void};

extern "C" {
    fn new_cublas_wrapper(device: c_int, stream: CUstream, dtype: c_int) -> *mut c_void;
    fn delete_cublas_wrapper(cublas_wrapper: *mut c_void);
    fn cublas_gemm(
        cublas_wrapper: *mut c_void,
        transa: c_int,
        transb: c_int,
        A: CTensorView,
        B: CTensorView,
        C: CTensorView,
    );
}

pub struct CublasWrapper {
    wrapper: *mut c_void,
}
unsafe impl Send for CublasWrapper {}
impl Drop for CublasWrapper {
    fn drop(&mut self) {
        unsafe {
            delete_cublas_wrapper(self.wrapper);
        }
        self.wrapper = std::ptr::null_mut();
    }
}

impl CublasWrapper {
    pub fn new(
        device: &Device,
        dtype: DType,
        stream: CUstream,
    ) -> candle_core::Result<CublasWrapper> {
        let device_id = match device {
            Device::Cuda(cuda) => *cuda.cu_device(),
            _ => {
                candle_core::bail!("not supported device")
            }
        };

        let p = match dtype {
            DType::F16 => unsafe {
                new_cublas_wrapper(device_id, stream, ScalarType::DATA_F16 as i32)
            },
            DType::BF16 => unsafe {
                new_cublas_wrapper(device_id, stream, ScalarType::DATA_BF16 as i32)
            },
            DType::F32 => unsafe {
                new_cublas_wrapper(device_id, stream, ScalarType::DATA_F32 as i32)
            },
            _ => {
                candle_core::bail!("not supported dtype")
            }
        };
        Ok(Self { wrapper: p })
    }
    pub fn linear(&self, input: &Tensor, weight: &Tensor) -> candle_core::Result<Tensor> {
        let (batch, num, _) = input.dims3()?;
        let batch_size = batch * num;
        let (output_dims, input_dims) = weight.dims2()?;
        let output = Tensor::zeros((batch, num, output_dims), input.dtype(), input.device())?;
        let input_view = common::ffi::CTensorView::from(input, false)?;
        let weight_view = common::ffi::CTensorView::from(weight, false)?;
        let output_view = common::ffi::CTensorView::from(&output, false)?;

        unsafe {
            cublas_gemm(self.wrapper, 0, 1, input_view, weight_view, output_view);
        }
        // println!("###After cublas");
        // cublas_wrapper_->Gemm(CUBLAS_OP_N,
        //     CUBLAS_OP_N,
        //     weight.output_dims,
        //     batch_size,
        //     weight.input_dims,
        //     (const T*)weight.kernel,
        //     weight.output_dims,
        //     input_data,
        //     weight.input_dims,
        //     output_data,
        //     weight.output_dims);

        Ok(output)
    }
}
