use candle_core::cuda_backend::cudarc::driver::sys::CUstream;
use candle_core::{shape::Dim, CpuStorage, CudaStorage, DType, Device, Layout, Shape, Tensor};
use common::{
    ffi::get_scalar_type,
    ffi::{CTensorView, ScalarType},
};
use common::{DefaultTensorCreator, TensorCreator};

use std::os::raw::{c_int, c_void};

use crate::common::get_column_major_dim;
use crate::unsafe_tensor_dtod_copy;

extern "C" {
    fn cuda_softmax_tensor(input: CTensorView, algo: c_int, stream: CUstream, output: CTensorView);
}

pub fn cuda_softmax<D: Dim>(
    intput: &Tensor,
    dim: D,
    stream: CUstream,
) -> candle_core::Result<Tensor> {
    let mut default_creator = DefaultTensorCreator {};
    cuda_softmax_(intput, dim, &mut default_creator, stream)
}
pub fn cuda_softmax_<F: TensorCreator, D: Dim>(
    input: &Tensor,
    dim: D,
    tensor_creator: &mut F,
    stream: CUstream,
) -> candle_core::Result<Tensor> {
    let out = tensor_creator.new(input.shape(), input.dtype(), input.device(), false)?;

    let input_view = common::ffi::CTensorView::from(input, false)?;
    let output_view = common::ffi::CTensorView::from(&out, false)?;

    unsafe {
        cuda_softmax_tensor(input_view, 0, stream, output_view);
    };
    Ok(out)
}

pub fn cuda_log_softmax<D: Dim>(
    intput: &Tensor,
    dim: D,
    stream: CUstream,
) -> candle_core::Result<Tensor> {
    let mut default_creator = DefaultTensorCreator {};
    cuda_log_softmax_(intput, dim, &mut default_creator, stream)
}
pub fn cuda_log_softmax_<F: TensorCreator, D: Dim>(
    input: &Tensor,
    dim: D,
    tensor_creator: &mut F,
    stream: CUstream,
) -> candle_core::Result<Tensor> {
    let out = tensor_creator.new(input.shape(), input.dtype(), input.device(), false)?;

    let input_view = common::ffi::CTensorView::from(input, false)?;
    let output_view = common::ffi::CTensorView::from(&out, false)?;

    unsafe {
        cuda_softmax_tensor(input_view, 1, stream, output_view);
    };
    Ok(out)
}
