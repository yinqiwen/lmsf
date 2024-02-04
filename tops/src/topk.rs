use candle_core::cuda_backend::cudarc::driver::sys::CUstream;
use candle_core::{CpuStorage, CudaStorage, DType, Device, Layout, Shape, Tensor};
use common::{
    ffi::get_scalar_type,
    ffi::{CTensorView, ScalarType},
};

use std::os::raw::{c_int, c_void};

use crate::common::get_column_major_dim;

extern "C" {
    fn cuda_topk_tensor(
        input: CTensorView,
        topk: c_int,
        dim: c_int,
        topk_type: c_int,
        stream: CUstream,
        output: CTensorView,
        indices: CTensorView,
    );
}

pub fn cuda_topk(
    t: &Tensor,
    topk: usize,
    dim: usize,
    stream: CUstream,
) -> candle_core::Result<(Tensor, Tensor)> {
    let dim = get_column_major_dim(t.shape(), dim)?;

    let mut indices_dims = Vec::from(t.dims());
    *indices_dims.last_mut().unwrap() = topk;
    let indices = Tensor::zeros(indices_dims.clone(), DType::U32, t.device())?;
    let out = Tensor::zeros(indices_dims, t.dtype(), t.device())?;

    let input_view = common::ffi::CTensorView::from(t, true)?;
    let output_view = common::ffi::CTensorView::from(&out, true)?;
    let indices_view = common::ffi::CTensorView::from(&indices, true)?;
    unsafe {
        cuda_topk_tensor(
            input_view,
            topk as i32,
            dim as i32,
            0,
            stream,
            output_view,
            indices_view,
        );
    };
    Ok((out, indices))
}
