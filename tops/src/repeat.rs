use candle_core::cuda_backend::cudarc::driver::sys::CUstream;
use candle_core::{CpuStorage, CudaStorage, DType, Device, Layout, Shape, Tensor};
use common::{
    ffi::get_scalar_type,
    ffi::{CTensorView, ScalarType},
};

use std::os::raw::{c_int, c_void};

use crate::common::get_column_major_dim;

extern "C" {
    fn cuda_repeat_tensor(
        input: CTensorView,
        dim0: c_int,
        dim1: c_int,
        dim2: c_int,
        dim3: c_int,
        stream: CUstream,
        output: CTensorView,
    );
}

pub fn cuda_repeat<S: Into<Shape>>(
    t: &Tensor,
    s: S,
    stream: CUstream,
) -> candle_core::Result<Tensor> {
    let s = s.into();
    if s.dims().len() != t.dims().len() {
        return Err(candle_core::Error::ShapeMismatchBinaryOp {
            lhs: t.shape().clone(),
            rhs: s,
            op: "cuda_repeat",
        }
        .bt());
    }
    let mut dim0 = 1_usize;
    let mut dim1 = 1_usize;
    let dim2 = 1_usize;
    let dim3 = 1_usize;
    let output = if s.dims().len() == 1 {
        dim0 = s.dims()[0];
        Tensor::zeros(dim0 * t.dims1()?, t.dtype(), t.device())?
    } else if s.dims().len() == 2 {
        dim0 = s.dims()[1];
        dim1 = s.dims()[0];
        let (tdim0, tdim1) = t.dims2()?;
        Tensor::zeros(
            (s.dims()[0] * tdim0, s.dims()[1] * tdim1),
            t.dtype(),
            t.device(),
        )?
    } else {
        return Err(candle_core::Error::UnexpectedNumberOfDims {
            expected: 2,
            got: s.dims().len(),
            shape: s,
        }
        .bt());
    };

    let input_view = common::ffi::CTensorView::from(t, true)?;
    let output_view = common::ffi::CTensorView::from(&output, true)?;
    unsafe {
        cuda_repeat_tensor(
            input_view,
            dim0 as i32,
            dim1 as i32,
            dim2 as i32,
            dim3 as i32,
            stream,
            output_view,
        );
    }
    Ok(output)
}

#[test]
fn test_repeate() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let a = Tensor::new(&[1.0], &device)?;
    let a = a.reshape((1, 1))?;
    let start = std::time::Instant::now();
    let b = cuda_repeat(
        &a,
        Shape::from_dims(&[1_usize, 32000_usize]),
        std::ptr::null_mut(),
    )?;
    println!(
        "cuda tensor repeat to {:?} cost {:?}",
        b.to_string(),
        start.elapsed()
    );
    Ok(())
}
