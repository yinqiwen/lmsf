#[macro_export]
macro_rules! get_cuda_slice {
    ($obj:expr) => {{
        let (_storage, _) = $obj.storage_and_layout();
        match _storage.deref() {
            candle_core::Storage::Cuda(cuda_storage) => match $obj.dtype() {
                candle_core::DType::U8 => crate::cuda_ext::DeviceDataPtr::new(
                    *cuda_storage.as_cuda_slice::<u8>()?.device_ptr() as *mut std::ffi::c_void,
                ),
                candle_core::DType::F16 => crate::cuda_ext::DeviceDataPtr::new(
                    *cuda_storage.as_cuda_slice::<half::f16>()?.device_ptr()
                        as *mut std::ffi::c_void,
                ),
                candle_core::DType::BF16 => crate::cuda_ext::DeviceDataPtr::new(
                    *cuda_storage.as_cuda_slice::<half::bf16>()?.device_ptr()
                        as *mut std::ffi::c_void,
                ),
                candle_core::DType::F32 => crate::cuda_ext::DeviceDataPtr::new(
                    *cuda_storage.as_cuda_slice::<f32>()?.device_ptr() as *mut std::ffi::c_void,
                ),
                candle_core::DType::U32 => crate::cuda_ext::DeviceDataPtr::new(
                    *cuda_storage.as_cuda_slice::<u32>()?.device_ptr() as *mut std::ffi::c_void,
                ),
                candle_core::DType::F64 => crate::cuda_ext::DeviceDataPtr::new(
                    *cuda_storage.as_cuda_slice::<f64>()?.device_ptr() as *mut std::ffi::c_void,
                ),
                candle_core::DType::I64 => crate::cuda_ext::DeviceDataPtr::new(
                    cuda_storage.as_cuda_slice::<i64>()?.as_kernel_param(),
                ),
            },
            _ => unreachable!("unexpected storage type"),
        }
    }};
}

#[macro_export]
macro_rules! get_cuda_device_ptr {
    ($obj:expr) => {{
        let (_storage, _) = $obj.storage_and_layout();
        match _storage.deref() {
            candle_core::Storage::Cuda(cuda_storage) => match $obj.dtype() {
                candle_core::DType::U8 => {
                    *cuda_storage.as_cuda_slice::<u8>()?.device_ptr() as *mut std::ffi::c_void
                }
                candle_core::DType::F16 => *cuda_storage.as_cuda_slice::<half::f16>()?.device_ptr()
                    as *mut std::ffi::c_void,
                candle_core::DType::BF16 => {
                    *cuda_storage.as_cuda_slice::<half::bf16>()?.device_ptr()
                        as *mut std::ffi::c_void
                }
                candle_core::DType::F32 => {
                    *cuda_storage.as_cuda_slice::<f32>()?.device_ptr() as *mut std::ffi::c_void
                }
                candle_core::DType::U32 => {
                    *cuda_storage.as_cuda_slice::<u32>()?.device_ptr() as *mut std::ffi::c_void
                }
                candle_core::DType::F64 => {
                    *cuda_storage.as_cuda_slice::<f64>()?.device_ptr() as *mut std::ffi::c_void
                }
                candle_core::DType::I64 => {
                    *cuda_storage.as_cuda_slice::<i64>()?.device_ptr() as *mut std::ffi::c_void
                }
            },
            _ => unreachable!("unexpected storage type"),
        }
    }};
}

#[macro_export]
macro_rules! get_cuda_device {
    ($obj:expr) => {
        if let candle_core::Device::Cuda(cuda_dev) = $obj {
            cuda_dev
        } else {
            unimplemented!("unreach");
        };
    };
}

#[macro_export]
macro_rules! dispatch_cuda_func {
    ($funcs:expr,$dtype:expr,$cfg:expr,$params:expr) => {
        match $dtype {
            DType::F16 => unsafe { $funcs[0].clone().launch($cfg, $params) },
            DType::F32 => unsafe { $funcs[1].clone().launch($cfg, $params) },
            DType::BF16 => {
                todo!()
            }
            _ => {
                todo!()
            }
        }
    };
}
