use candle_core::cuda_backend::cudarc::driver::sys::CUstream;
use candle_core::{CpuStorage, CudaStorage, DType, Device, Layout, Shape, Tensor};
use common::{
    ffi::get_scalar_type,
    ffi::{CTensorView, ScalarType},
};
use common::{DefaultTensorCreator, TensorCreator};

use std::os::raw::{c_uint, c_void};

use crate::common::get_column_major_dim;
use crate::unsafe_tensor_dtod_copy;

extern "C" {
    fn cuda_sort_tensor(
        input: CTensorView,
        dim: c_uint,
        ascend: bool,
        stream: CUstream,
        output: CTensorView,
        indices: CTensorView,
    );
}

pub fn cuda_sort(
    t: Tensor,
    dim: usize,
    descending: bool,
    stream: CUstream,
) -> candle_core::Result<(Tensor, Tensor)> {
    let mut default_creator = DefaultTensorCreator {};
    cuda_sort_(t, dim, descending, &mut default_creator, stream)

    // let out = t.copy()?;
    // let indices = Tensor::zeros(out.shape(), DType::U32, t.device())?;
    // let dim = get_column_major_dim(t.shape(), dim)?;

    // let input_view = common::ffi::CTensorView::from(&t, true)?;
    // let output_view = common::ffi::CTensorView::from(&out, true)?;
    // let indices_view = common::ffi::CTensorView::from(&indices, true)?;
    // unsafe {
    //     cuda_sort_tensor(
    //         input_view,
    //         dim as u32,
    //         !descending,
    //         stream,
    //         output_view,
    //         indices_view,
    //     );
    // };
    // Ok((out, indices))
}

pub fn cuda_sort_<F: TensorCreator>(
    t: Tensor,
    dim: usize,
    descending: bool,
    tensor_creator: &mut F,
    stream: CUstream,
) -> candle_core::Result<(Tensor, Tensor)> {
    let out = tensor_creator.new(t.shape(), t.dtype(), t.device(), false)?;
    unsafe_tensor_dtod_copy(&out, &t)?;
    let indices = tensor_creator.new(out.shape(), DType::U32, t.device(), true)?;
    //let out = t.copy()?;
    // let indices = Tensor::zeros(out.shape(), DType::U32, t.device())?;
    let dim = get_column_major_dim(t.shape(), dim)?;

    let input_view = common::ffi::CTensorView::from(&t, true)?;
    let output_view = common::ffi::CTensorView::from(&out, true)?;
    let indices_view = common::ffi::CTensorView::from(&indices, true)?;
    unsafe {
        cuda_sort_tensor(
            input_view,
            dim as u32,
            !descending,
            stream,
            output_view,
            indices_view,
        );
    };
    Ok((out, indices))
}

#[test]
fn test_sort() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let a = Tensor::new(
        &[
            0.6719, -0.2162, 2.3332, 0.0608, 0.9497, -0.5793, 0.6058, 0.0061, 0.3343, -0.5071,
            1.0960, 0.9553,
        ],
        &device,
    )?;
    let a = a.reshape((3, 4))?;

    let stream: CUstream = std::ptr::null_mut();
    let (out, indices) = cuda_sort(a, 1, false, stream)?;
    println!("out:{:?}", out.to_string());
    println!("indices:{:?}", indices.to_string());
    Ok(())
}
