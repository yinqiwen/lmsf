use candle::cuda_backend::cudarc::driver::{DevicePtr, DeviceRepr, LaunchAsync};
use candle::cuda_backend::WrapErr;
use candle::{
    backend::BackendStorage, cuda_backend::cudarc::driver::LaunchConfig, shape::Dim, CpuStorage,
    CudaStorage, DType, Layout, Shape, Storage, WithDType,
};
use candle::{Device, IndexOp, Tensor};
use common::cuda_ext::get_tensor_cuda_device_ptr;

fn kernel_name(root: &str, dtype: DType) -> String {
    let dtype = dtype.as_str();
    format!("{root}_{dtype}")
}

pub fn cuda_copy(dst: &Tensor, src: &Tensor) -> candle::Result<()> {
    let device = match dst.device() {
        Device::Cuda(cuda_dev) => cuda_dev,
        _ => {
            candle::bail!("unexpected device")
        }
    };
    if dst.dtype() != src.dtype() {
        return Err(candle::Error::DTypeMismatchBinaryOp {
            lhs: dst.dtype(),
            rhs: src.dtype(),
            op: "cuda_copy",
        }
        .bt());
    }

    if !src.is_contiguous() {
        return candle::bail!("src tensor must be contiguous");
    }
    if dst.is_contiguous() {
        return tops::unsafe_tensor_dtod_copy(dst, src);
    }

    if dst.shape() != src.shape() {
        return Err(candle::Error::UnexpectedShape {
            msg: format!("shape mismatch for cuda_copy"),
            expected: dst.shape().clone(),
            got: src.shape().clone(),
        }
        .bt())?;
    }

    let elem_count = dst.shape().elem_count();
    let dims = dst.shape().dims();
    let cfg = LaunchConfig::for_num_elems(elem_count as u32);

    let dims_and_strides = device.htod_copy([dims, dst.stride()].concat()).w()?;
    let dst_data = get_tensor_cuda_device_ptr(dst)?;
    let src_data = get_tensor_cuda_device_ptr(src)?;
    let func = device.get_or_load_func(&kernel_name("copy", dst.dtype()), candle_patch::UNARY)?;

    let params = (
        elem_count,
        dims.len(),
        &dims_and_strides,
        src_data,
        dst_data,
    );
    unsafe { func.launch(cfg, params) }.w()?;

    Ok(())
}

#[test]
fn test_copy() -> candle::Result<()> {
    let device = candle::Device::new_cuda(0).unwrap();

    let test = Tensor::zeros((2, 10), DType::U32, &device)?;

    let test1 = test.narrow(1, 0, 5)?;
    println!("test1 {:?}/{}", test1.shape(), test1.is_contiguous());

    let x = Tensor::ones((2, 5), DType::U32, &device)?;

    cuda_copy(&test1, &x)?;
    println!("x {:?}", x.to_string());
    println!("test1 {:?}", test1.to_string());
    println!("test {:?}", test.to_string());
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
