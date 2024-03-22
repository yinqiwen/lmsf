use candle::cuda_backend::cudarc::driver::sys::CUstream;
use candle::{cuda_backend::cudarc::driver::DeviceRepr, Device, Tensor};
use common::cuda_ext::get_tensor_cuda_device_ptr;
use common::get_tensor_kernel_param;
use libc::c_int;

use std::os::raw::{c_longlong, c_void};

extern "C" {
    fn cuda_async_htod(dptr: *mut c_void, hptr: *const c_void, n: c_longlong, stream: CUstream);
    fn cuda_async_dtod(dptr: *mut c_void, hptr: *const c_void, n: c_longlong, stream: CUstream);
    fn cuda_async_set(dptr: *mut c_void, v: c_int, n: c_int, stream: CUstream);
}

pub fn unsafe_tensor_zero(t: &Tensor) -> candle::Result<()> {
    let n = t.dtype().size_in_bytes() * t.elem_count();
    match t.device() {
        Device::Cpu => todo!("cpu memset zero"),
        Device::Cuda(cuda_dev) => {
            let stream = cuda_dev.cu_stream();
            let dptr = get_tensor_cuda_device_ptr(t)?;
            unsafe {
                cuda_async_set(dptr.as_ffi_ptr(), 0_i32, n as i32, *stream);
            }
        }
        _ => {
            unreachable!("unexpected storage type")
        }
    }
    Ok(())
}

pub fn unsafe_tensor_htod_copy(host_tensor: &Tensor, device_tensor: &Tensor) -> candle::Result<()> {
    if host_tensor.dtype() != device_tensor.dtype() {
        return Err(candle::Error::DTypeMismatchBinaryOp {
            lhs: host_tensor.dtype(),
            rhs: device_tensor.dtype(),
            op: "unsafe_tensor_htod_copy",
        }
        .bt());
    }
    let n = host_tensor.dtype().size_in_bytes() * host_tensor.elem_count();
    // let dptr = get_tensor_kernel_param(device_tensor)?;
    let dptr = get_tensor_cuda_device_ptr(device_tensor)?;
    let hptr = get_tensor_kernel_param(host_tensor)?;
    // println!("copy {}", n);
    match device_tensor.device() {
        Device::Cpu => unsafe {
            std::ptr::copy_nonoverlapping(hptr, dptr.as_ffi_ptr(), n);
        },
        Device::Cuda(cuda_dev) => {
            let stream = cuda_dev.cu_stream();
            unsafe {
                cuda_async_htod(dptr.as_ffi_ptr(), hptr, n as i64, *stream);
            }
        }
        _ => {
            unreachable!("unexpected storage type")
        }
    }
    Ok(())
}

pub fn unsafe_tensor_dtod_copy(dst: &Tensor, src: &Tensor) -> candle::Result<()> {
    if dst.dtype() != src.dtype() {
        return Err(candle::Error::DTypeMismatchBinaryOp {
            lhs: dst.dtype(),
            rhs: src.dtype(),
            op: "unsafe_tensor_dtod_copy",
        }
        .bt());
    }
    let n = src.dtype().size_in_bytes() * src.elem_count();
    // let dptr = get_tensor_kernel_param(device_tensor)?;
    let mut dptr = get_tensor_cuda_device_ptr(dst)?;
    let mut sptr = get_tensor_cuda_device_ptr(src)?;

    if dst.is_contiguous() {
        match dst.device() {
            Device::Cpu => unsafe {
                std::ptr::copy_nonoverlapping(sptr.as_ffi_ptr(), dptr.as_ffi_ptr(), n);
            },
            Device::Cuda(cuda_dev) => {
                let stream = cuda_dev.cu_stream();
                unsafe {
                    cuda_async_dtod(dptr.as_ffi_ptr(), sptr.as_ffi_ptr(), n as i64, *stream);
                }
            }
            _ => {
                unreachable!("unexpected storage type")
            }
        }
    } else {
        let dst_strides = dst.stride();
        if dst_strides.len() != 2 {
            unimplemented!("unimplemented")
        } else {
            match dst.device() {
                Device::Cuda(cuda_dev) => {
                    let stream = cuda_dev.cu_stream();
                    let copy_n = src.dims()[1] * src.dtype().size_in_bytes();

                    for _ in 0..src.dims()[0] {
                        unsafe {
                            cuda_async_dtod(
                                dptr.as_ffi_ptr(),
                                sptr.as_ffi_ptr(),
                                copy_n as i64,
                                *stream,
                            );
                            dptr.advance(copy_n);
                            sptr.advance(copy_n);
                            //dptr.as_ffi_ptr()
                        }
                    }
                }
                _ => {
                    unreachable!("unexpected storage type")
                }
            }
        }
    }
    Ok(())
}

pub fn unsafe_tensor_write<T: DeviceRepr + Unpin + std::fmt::Debug>(
    t: &Tensor,
    v: Vec<T>,
) -> candle::Result<()> {
    let n = t.dtype().size_in_bytes() * v.len();
    //let dptr = get_tensor_kernel_param(t)?;
    let dptr = get_tensor_cuda_device_ptr(t)?;
    let hptr = v.as_ptr() as *const c_void;
    // println!("data:{:?}", v);

    match t.device() {
        Device::Cpu => unsafe {
            std::ptr::copy_nonoverlapping(hptr, dptr.as_ffi_ptr(), n);
        },
        Device::Cuda(cuda_dev) => {
            let stream = cuda_dev.cu_stream();
            unsafe {
                cuda_async_htod(dptr.as_ffi_ptr(), hptr, n as i64, *stream);
            }
        }
        _ => {
            unreachable!("unexpected storage type")
        }
    }
    Ok(())
}

#[test]
fn test_cuda_async_htod_copy() -> candle::Result<()> {
    use candle::{DType, Device};
    let cpu = Device::Cpu;
    let device = Device::new_cuda(0)?;
    let v = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let a = Tensor::new(v.clone(), &cpu)?;
    let b = Tensor::zeros(8, DType::F32, &device)?;
    // unsafe_tensor_write(&b, v)?;
    unsafe_tensor_htod_copy(&a, &b)?;
    // println!("Cost :{:?} to create tensor", start.elapsed());
    println!("out:{:?}", a.to_string());
    println!("out:{:?}", b.to_string());
    Ok(())
}

#[test]
fn test_cuda_zero() -> candle::Result<()> {
    use candle::{DType, Device};
    let device = Device::new_cuda(0)?;
    let b = Tensor::ones(8, DType::F32, &device)?;
    // unsafe_tensor_write(&b, v)?;
    unsafe_tensor_zero(&b)?;
    println!("out:{:?}", b.to_string());
    Ok(())
}
