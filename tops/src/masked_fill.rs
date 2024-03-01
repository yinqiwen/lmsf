use candle_core::cuda_backend::cudarc::driver::sys::CUstream;
use candle_core::{Device, Tensor, WithDType};
use common::{
    ffi::get_scalar_type,
    ffi::{CTensorView, ScalarType},
};
use common::{DefaultTensorCreator, TensorCreator};
use libc::{c_double, c_longlong};

extern "C" {

    fn cuda_masked_fill_float_(
        input: CTensorView,
        mask: CTensorView,
        scalar_operand: c_double,
        stream: CUstream,
    );
    fn cuda_masked_fill_int_(
        input: CTensorView,
        mask: CTensorView,
        scalar_operand: c_longlong,
        stream: CUstream,
    );

}

pub fn cuda_masked_fill_<D: WithDType>(
    t: &Tensor,
    mask: &Tensor,
    scalar_operand: D,
    stream: CUstream,
) -> candle_core::Result<()> {
    if D::DTYPE != t.dtype() {
        return Err(candle_core::Error::UnexpectedDType {
            msg: "invalid dtype for given fill value",
            expected: t.dtype(),
            got: D::DTYPE,
        }
        .bt());
    }
    let output_view = CTensorView::from(t, false)?;
    let mask_view = CTensorView::from(mask, false)?;
    if D::DTYPE.is_float() {
        unsafe {
            cuda_masked_fill_float_(output_view, mask_view, scalar_operand.to_f64(), stream);
        }
    } else {
        unsafe {
            cuda_masked_fill_int_(
                output_view,
                mask_view,
                scalar_operand.to_f64() as i64,
                stream,
            );
        }
    }
    Ok(())
}

pub fn cuda_masked_fill_neg_inf_(on_false: &Tensor, mask: &Tensor) -> candle_core::Result<()> {
    match on_false.dtype() {
        candle_core::DType::U8 => cuda_masked_fill_(on_false, mask, 0_u8, std::ptr::null_mut()),
        candle_core::DType::F16 => cuda_masked_fill_(
            on_false,
            mask,
            half::f16::NEG_INFINITY,
            std::ptr::null_mut(),
        ),
        candle_core::DType::F32 => {
            cuda_masked_fill_(on_false, mask, f32::NEG_INFINITY, std::ptr::null_mut())
        }
        candle_core::DType::BF16 => cuda_masked_fill_(
            on_false,
            mask,
            half::bf16::NEG_INFINITY,
            std::ptr::null_mut(),
        ),
        candle_core::DType::F64 => {
            cuda_masked_fill_(on_false, mask, f64::NEG_INFINITY, std::ptr::null_mut())
        }
        _ => {
            candle_core::bail!(
                "not supported dtype:{:?} for masked_fill_neg_inf",
                on_false.dtype()
            );
        }
    }
}

#[test]
fn test_masked_fill() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    // let a = cuda_arange(0_u32, 100, &device)?;
    // println!("{}", a.to_string());
    Ok(())
}
