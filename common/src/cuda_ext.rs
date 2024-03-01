use std::ops::Deref;

use candle_core::{
    cuda_backend::cudarc::driver::{sys, DevicePtr, DeviceRepr},
    Device, Storage, Tensor,
};

#[repr(C)]
pub struct CudaDevicePtr {
    ptr: sys::CUdeviceptr,
}

unsafe impl DeviceRepr for CudaDevicePtr {
    #[inline(always)]
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        (&self.ptr) as *const sys::CUdeviceptr as *mut std::ffi::c_void
    }
}

impl Clone for CudaDevicePtr {
    fn clone(&self) -> Self {
        CudaDevicePtr { ptr: self.ptr }
    }
}

impl CudaDevicePtr {
    pub fn as_ffi_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr as *mut std::ffi::c_void
    }
    pub fn as_ptr_int(&self) -> i64 {
        self.ptr as i64
    }
    pub fn null() -> Self {
        Self { ptr: 0 }
    }
}

pub fn get_tensor_cuda_device_ptr(tensor: &Tensor) -> candle_core::Result<CudaDevicePtr> {
    let (storage, _) = tensor.storage_and_layout();
    let start_offset = tensor.layout().start_offset();
    let data = match storage.deref() {
        Storage::Cuda(cuda_storage) => match tensor.dtype() {
            candle_core::DType::U8 => {
                // println!("slicelen:{}", cuda_storage.as_cuda_slice::<u8>()?.len());
                cuda_storage.as_cuda_slice::<u8>()?.device_ptr()
            }
            candle_core::DType::F16 => cuda_storage.as_cuda_slice::<half::f16>()?.device_ptr(),
            candle_core::DType::BF16 => cuda_storage.as_cuda_slice::<half::bf16>()?.device_ptr(),
            candle_core::DType::F32 => cuda_storage.as_cuda_slice::<f32>()?.device_ptr(),
            candle_core::DType::U32 => cuda_storage.as_cuda_slice::<u32>()?.device_ptr(),
            candle_core::DType::F64 => cuda_storage.as_cuda_slice::<f64>()?.device_ptr(),
            candle_core::DType::I64 => {
                // println!("slicelen:{}", cuda_storage.as_cuda_slice::<i64>()?.len());
                cuda_storage.as_cuda_slice::<i64>()?.device_ptr()
            }
        },
        _ => unreachable!("unexpected storage type"),
    };
    // println!("###start_offset:{}", start_offset);
    let ptr_int = if start_offset > 0 {
        *data + (start_offset * tensor.dtype().size_in_bytes()) as u64
    } else {
        *data
    };
    Ok(CudaDevicePtr { ptr: ptr_int })
}

pub fn cuda_profiler_start() {
    unsafe {
        sys::cuProfilerStart();
    }
}

pub fn cuda_profiler_stop() {
    unsafe {
        sys::cuProfilerStop();
    }
}
