use candle::cuda_backend::cudarc::driver::sys::CUstream;

use candle::{Tensor};
use common::ffi::{CTensorView};



use std::os::raw::{c_int};

extern "C" {
    fn fastertransformer_silu_activation(
        a: CTensorView,
        b: CTensorView,
        num_token: c_int,
        inter_size: c_int,
        stream: CUstream,
    );
}

pub fn cuda_silu_activation(a: &Tensor, b: &Tensor, stream: CUstream) -> candle::Result<()> {
    let (bsize, num_token, inter_size) = a.dims3()?;
    let a_view = common::ffi::CTensorView::from(a, false)?;
    let b_view = common::ffi::CTensorView::from(b, false)?;
    unsafe {
        fastertransformer_silu_activation(
            a_view,
            b_view,
            (bsize * num_token) as i32,
            inter_size as i32,
            stream,
        );
    }
    Ok(())
}
