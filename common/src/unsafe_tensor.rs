use std::{ffi::c_void, ops::Deref};

use candle::{cuda_backend::cudarc::driver::DeviceRepr, Storage, Tensor};

pub fn get_tensor_kernel_param(tensor: &Tensor) -> candle::Result<*mut std::ffi::c_void> {
    let (storage, _) = tensor.storage_and_layout();
    let data = match storage.deref() {
        Storage::Cpu(cpu_storage) => match tensor.dtype() {
            candle::DType::U8 => cpu_storage.as_slice::<u8>()?.as_ptr() as *mut c_void,
            candle::DType::F16 => cpu_storage.as_slice::<half::f16>()?.as_ptr() as *mut c_void,
            candle::DType::BF16 => cpu_storage.as_slice::<half::bf16>()?.as_ptr() as *mut c_void,
            candle::DType::F32 => cpu_storage.as_slice::<f32>()?.as_ptr() as *mut c_void,
            candle::DType::U32 => cpu_storage.as_slice::<u32>()?.as_ptr() as *mut c_void,
            candle::DType::F64 => cpu_storage.as_slice::<f64>()?.as_ptr() as *mut c_void,
            candle::DType::I64 => cpu_storage.as_slice::<i64>()?.as_ptr() as *mut c_void,
        },
        Storage::Cuda(cuda_storage) => match tensor.dtype() {
            candle::DType::U8 => cuda_storage.as_cuda_slice::<u8>()?.as_kernel_param(),
            candle::DType::F16 => cuda_storage.as_cuda_slice::<half::f16>()?.as_kernel_param(),
            candle::DType::BF16 => cuda_storage
                .as_cuda_slice::<half::bf16>()?
                .as_kernel_param(),
            candle::DType::F32 => cuda_storage.as_cuda_slice::<f32>()?.as_kernel_param(),
            candle::DType::U32 => cuda_storage.as_cuda_slice::<u32>()?.as_kernel_param(),
            candle::DType::F64 => cuda_storage.as_cuda_slice::<f64>()?.as_kernel_param(),
            candle::DType::I64 => cuda_storage.as_cuda_slice::<i64>()?.as_kernel_param(),
        },
        _ => unreachable!("unexpected storage type"),
    };

    Ok(data)
}

// pub fn get_tensor_slice_kernel_param(
//     tensor: &Tensor,
//     _range: impl RangeBounds<usize>,
// ) -> candle::Result<*mut std::ffi::c_void> {
//     let (storage, _) = tensor.storage_and_layout();
//     let data = match storage.deref() {
//         Storage::Cpu(cpu_storage) => match tensor.dtype() {
//             candle::DType::U8 => cpu_storage.as_slice::<u8>()?.as_ptr() as *mut c_void,
//             candle::DType::F16 => cpu_storage.as_slice::<half::f16>()?.as_ptr() as *mut c_void,
//             candle::DType::BF16 => cpu_storage.as_slice::<half::bf16>()?.as_ptr() as *mut c_void,
//             candle::DType::F32 => cpu_storage.as_slice::<f32>()?.as_ptr() as *mut c_void,
//             candle::DType::U32 => cpu_storage.as_slice::<u32>()?.as_ptr() as *mut c_void,
//             candle::DType::F64 => cpu_storage.as_slice::<f64>()?.as_ptr() as *mut c_void,
//             candle::DType::I64 => cpu_storage.as_slice::<i64>()?.as_ptr() as *mut c_void,
//         },
//         Storage::Cuda(cuda_storage) => match tensor.dtype() {
//             candle::DType::U8 => cuda_storage.as_cuda_slice::<u8>()?.as_kernel_param(),
//             candle::DType::F16 => cuda_storage.as_cuda_slice::<half::f16>()?.as_kernel_param(),
//             candle::DType::BF16 => cuda_storage
//                 .as_cuda_slice::<half::bf16>()?
//                 .as_kernel_param(),
//             candle::DType::F32 => cuda_storage.as_cuda_slice::<f32>()?.as_kernel_param(),
//             candle::DType::U32 => cuda_storage.as_cuda_slice::<u32>()?.as_kernel_param(),
//             candle::DType::F64 => cuda_storage.as_cuda_slice::<f64>()?.as_kernel_param(),
//             candle::DType::I64 => cuda_storage.as_cuda_slice::<i64>()?.as_kernel_param(),
//         },
//         _ => unreachable!("unexpected storage type"),
//     };

//     Ok(data)
// }

#[test]
fn test_() -> candle::Result<()> {
    use candle::{DType, Device};
    let device = Device::Cpu;
    let a = Tensor::zeros(3, DType::F32, &device)?;
    let ptr = get_tensor_kernel_param(&a)?;
    let data = vec![1.0_f32, 2.0, 3.0];
    let data_ptr = data.as_ptr() as *const c_void;
    unsafe {
        std::ptr::copy_nonoverlapping(data_ptr, ptr, 3 * 4);
    }

    println!("a:{:?}", a.to_string());

    Ok(())
}
