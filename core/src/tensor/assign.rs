use candle::cuda_backend::cudarc::driver::LaunchAsync;
use candle::cuda_backend::WrapErr;
use candle::{cuda_backend::cudarc::driver::LaunchConfig, DType, WithDType};
use candle::{Device, Tensor};
use common::cuda_ext::get_tensor_cuda_device_ptr;

fn kernel_name(root: &str, dtype: DType) -> String {
    let dtype = dtype.as_str();
    format!("{root}_{dtype}")
}

pub fn cuda_assign<D: WithDType>(t: &Tensor, v: D) -> candle::Result<()> {
    let device = match t.device() {
        Device::Cuda(cuda_dev) => cuda_dev,
        _ => {
            candle::bail!("unexpected device")
        }
    };
    if D::DTYPE != t.dtype() {}

    let elem_count = t.shape().elem_count();
    let dims = t.shape().dims();
    let cfg = LaunchConfig::for_num_elems(elem_count as u32);

    let dims_and_strides = device.htod_copy([dims, t.stride()].concat()).w()?;
    let data = get_tensor_cuda_device_ptr(t)?;

    let func = device.get_or_load_func(&kernel_name("assign", t.dtype()), candle_patch::UNARY)?;

    match t.dtype() {
        DType::U8 => {
            let params = (
                elem_count,
                dims.len(),
                &dims_and_strides,
                data,
                v.to_f64() as u8,
            );
            unsafe { func.launch(cfg, params) }.w()?;
        }
        DType::U32 => {
            let params = (
                elem_count,
                dims.len(),
                &dims_and_strides,
                data,
                v.to_f64() as u32,
            );
            unsafe { func.launch(cfg, params) }.w()?;
        }
        DType::F16 => {
            let params = (
                elem_count,
                dims.len(),
                &dims_and_strides,
                data,
                half::f16::from_f64(v.to_f64()),
            );
            unsafe { func.launch(cfg, params) }.w()?;
        }
        DType::F32 => {
            let params = (
                elem_count,
                dims.len(),
                &dims_and_strides,
                data,
                v.to_f64() as f32,
            );
            unsafe { func.launch(cfg, params) }.w()?;
        }
        DType::F64 => {
            let params = (elem_count, dims.len(), &dims_and_strides, data, v.to_f64());
            unsafe { func.launch(cfg, params) }.w()?;
        }
        DType::BF16 => {
            let params = (
                elem_count,
                dims.len(),
                &dims_and_strides,
                data,
                half::bf16::from_f64(v.to_f64()),
            );
            unsafe { func.launch(cfg, params) }.w()?;
        }
        DType::I64 => {
            let params = (
                elem_count,
                dims.len(),
                &dims_and_strides,
                data,
                v.to_f64() as i64,
            );
            unsafe { func.launch(cfg, params) }.w()?;
        }
    }

    Ok(())
}

#[test]
fn test_assign() -> candle::Result<()> {
    use candle::IndexOp;
    let device = candle::Device::new_cuda(0).unwrap();

    let test = Tensor::rand(1_f32, 10.0, (2, 10), &device)?;
    println!("{}", test.to_string());
    let test2 = test.i((.., 9..10))?;
    println!("{:?} {:?}", test2.shape(), test2.stride());
    //tops::unsafe_tensor_zero(&test2)?;
    cuda_assign(&test2, 0_f32);
    println!("{}", test2.to_string());
    println!("{}", test.to_string());

    // println!("{} {:?}", test.to_string(), test.stride());
    // let src = Tensor::arange(0u32, 2, &device)?
    //     .reshape((2, 1))?
    //     .to_dtype(DType::F32)?;
    // let test2 = test.slice_assign(&[0..2, 7..8], &src)?;
    // println!("{} {:?}", test.to_string(), test.stride());
    // println!("{} {:?}", test2.to_string(), test2.stride());
    // // let last = test.i((.., 7))?;
    // println!(
    //     "{} {:?} {:?}",
    //     last.to_string(),
    //     last.shape(),
    //     last.stride()
    // );
    Ok(())
}
