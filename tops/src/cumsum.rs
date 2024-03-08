use candle_core::cuda_backend::cudarc::driver::sys::CUstream;
use candle_core::shape::Dim;
use candle_core::{CpuStorage, CudaStorage, DType, Device, Layout, Shape, Tensor};
use common::{
    ffi::get_scalar_type,
    ffi::{CTensorView, ScalarType},
};
use common::{DefaultTensorCreator, TensorCreator};

use std::os::raw::{c_uint, c_void};

use crate::common::get_column_major_dim;

extern "C" {
    fn cuda_cumsum_tensor(input: CTensorView, dim: c_uint, stream: CUstream, output: CTensorView);
}

pub fn cuda_cumsum(t: &Tensor, dim: usize, stream: CUstream) -> candle_core::Result<Tensor> {
    let mut default_creator = DefaultTensorCreator {};
    cuda_cumsum_(t, dim, &mut default_creator, stream)
}

pub fn cuda_cumsum_<F: TensorCreator, D: Dim>(
    t: &Tensor,
    dim: D,
    tensor_creator: &mut F,
    stream: CUstream,
) -> candle_core::Result<Tensor> {
    let out = tensor_creator.new(t.shape(), t.dtype(), t.device(), false)?;
    //let out = Tensor::zeros(t.shape(), t.dtype(), t.device())?;
    let dim = dim.to_index(t.shape(), "cuda_cumsum_")?;
    // let dim = get_column_major_dim(t.shape(), dim)?;

    // let input_view = CTensorView::from(t, true)?;
    // let output_view = CTensorView::from(&out, true)?;

    let input_view = CTensorView::from(t, false)?;
    let output_view = CTensorView::from(&out, false)?;
    unsafe {
        cuda_cumsum_tensor(input_view, dim as u32, stream, output_view);
    };
    Ok(out)
}
