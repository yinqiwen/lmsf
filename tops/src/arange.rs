use candle::cuda_backend::cudarc::driver::sys::CUstream;
use candle::{Device, Tensor, WithDType};
use common::{
    ffi::get_scalar_type,
    ffi::{CTensorView, ScalarType},
};
use common::{DefaultTensorCreator, TensorCreator};
use libc::{c_double, c_longlong};

extern "C" {

    fn cuda_arrange_int_tensor(
        start: c_longlong,
        delta: c_longlong,
        arange_elem_cnt: c_longlong,
        output: CTensorView,
        stream: CUstream,
    );
    fn cuda_arrange_float_tensor(
        start: c_double,
        delta: c_double,
        arange_elem_cnt: c_longlong,
        output: CTensorView,
        stream: CUstream,
    );

}

pub fn cuda_arange_<D: WithDType, F: TensorCreator>(
    start: D,
    end: D,
    device: &Device,
    tensor_creator: &mut F,
    stream: CUstream,
) -> candle::Result<Tensor> {
    let arange_elem_cnt = (end - start).to_f64() as i64;
    let out = tensor_creator.new(arange_elem_cnt as usize, D::DTYPE, device, false)?;
    let output_view = CTensorView::from(&out, false)?;
    if out.dtype().is_int() {
        unsafe {
            cuda_arrange_int_tensor(
                start.to_f64() as i64,
                1_i64,
                arange_elem_cnt,
                output_view,
                stream,
            );
        }
    } else {
        unsafe {
            cuda_arrange_float_tensor(start.to_f64(), 1_f64, arange_elem_cnt, output_view, stream);
        }
    }
    Ok(out)
}

pub fn cuda_arange<D: WithDType>(start: D, end: D, device: &Device) -> candle::Result<Tensor> {
    let mut default_creator = DefaultTensorCreator {};
    cuda_arange_(
        start,
        end,
        device,
        &mut default_creator,
        std::ptr::null_mut(),
    )
}

#[test]
fn test_arrange() -> candle::Result<()> {
    let device = Device::new_cuda(0)?;
    let a = cuda_arange(0_u32, 100, &device)?;
    println!("{}", a.to_string());
    Ok(())
}
