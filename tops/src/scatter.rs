use candle_core::cuda_backend::cudarc::driver::sys::CUstream;
use candle_core::shape::Dim;
use candle_core::{CpuStorage, CudaStorage, DType, Device, Layout, Shape, Tensor};
use common::{
    ffi::get_scalar_type,
    ffi::{CTensorView, ScalarType},
};

use std::os::raw::{c_uint, c_void};

use crate::common::get_column_major_dim;
use crate::cuda_sort;

extern "C" {
    fn cuda_scatter_tensor(
        index: CTensorView,
        src: CTensorView,
        dim: c_uint,
        output: CTensorView,
        stream: CUstream,
    );
}

pub fn cuda_scatter<D: Dim>(
    dst: &Tensor,
    index: &Tensor,
    src: &Tensor,
    dim: D,
    stream: CUstream,
) -> candle_core::Result<()> {
    let dim = dim.to_index(dst.shape(), "scatter")?;
    let index_view = common::ffi::CTensorView::from(index, false)?;
    let output_view = common::ffi::CTensorView::from(dst, false)?;
    let src_view = common::ffi::CTensorView::from(src, false)?;
    // println!(
    //     "index_view:{:?}, output_view:{:?},src_view:{:?},",
    //     index_view, output_view, src_view
    // );
    unsafe {
        cuda_scatter_tensor(index_view, src_view, dim as c_uint, output_view, stream);
    };
    Ok(())
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
