use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DeviceRepr, LaunchAsync};
use candle_core::cuda_backend::WrapErr;
use candle_core::{
    backend::BackendStorage, cuda_backend::cudarc::driver::LaunchConfig, shape::Dim, CpuStorage,
    CudaStorage, DType, Layout, Shape, Storage,
};
use candle_core::{Device, IndexOp, Tensor};
use common::cuda_ext::get_tensor_cuda_device_ptr;

fn kernel_name(root: &str, dtype: DType) -> String {
    let dtype = dtype.as_str();
    format!("{root}_{dtype}")
}

pub fn cuda_tensor_ones(t: &Tensor) -> candle_core::Result<()> {
    let device = match t.device() {
        Device::Cuda(cuda_dev) => cuda_dev,
        _ => {
            candle_core::bail!("unexpected device")
        }
    };
    let elem_count = t.shape().elem_count();
    let cfg = LaunchConfig::for_num_elems(elem_count as u32);
    let data = get_tensor_cuda_device_ptr(t)?;
    //let data = DeviceDataPtr::new(data_ptr);
    let func = device.get_or_load_func(&kernel_name("fill", t.dtype()), candle_kernels::FILL)?;
    match t.dtype() {
        DType::U8 => {
            let params = (data, 1_u8, elem_count);
            unsafe { func.launch(cfg, params) }.w()?;
        }
        DType::U32 => {
            let params = (data, 1_u32, elem_count);
            unsafe { func.launch(cfg, params) }.w()?;
        }
        DType::F16 => {
            let params = (data, half::f16::from_f32(1.0_f32), elem_count);
            unsafe { func.launch(cfg, params) }.w()?;
        }
        DType::F32 => {
            let params = (data, 1_f32, elem_count);
            unsafe { func.launch(cfg, params) }.w()?;
        }
        DType::F64 => {
            let params = (data, 1_f64, elem_count);
            unsafe { func.launch(cfg, params) }.w()?;
        }
        DType::BF16 => {
            let params = (data, half::bf16::from_f32(1.0_f32), elem_count);
            unsafe { func.launch(cfg, params) }.w()?;
        }
        DType::I64 => {
            let params = (data, 1_i64, elem_count);
            unsafe { func.launch(cfg, params) }.w()?;
        }
    }
    Ok(())
}

#[test]
fn test_cast() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let src = Tensor::ones((5), DType::U8, &device)?;
    //src.to_dtype(dtype)
    let dst = src.to_dtype(DType::U32)?;
    println!("dst:{}", dst.to_string());
    Ok(())
}

#[test]
fn test_ones() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let src = Tensor::zeros(8, DType::F16, &device)?;
    let src1 = src.i(4..)?;
    //src.to_dtype(dtype)
    cuda_tensor_ones(&src1)?;
    println!("dst:{:?}", src.to_string());
    println!("dst:{:?}", src1.to_string());
    Ok(())
}
