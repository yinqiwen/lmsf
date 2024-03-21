use std::os::raw::{c_int, c_longlong, c_uint, c_void};

#[derive(Debug)]
#[repr(C)]
pub struct CShapeView {
    pub shape: [c_longlong; 4],
    pub ndim: c_int,
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct CTensorView {
    pub ptr: *mut c_void,
    pub shape: [c_longlong; 4],
    pub stride: [c_longlong; 4],
    pub dtype: c_uint,
    pub ndim: c_int,
}

#[derive(Debug)]
#[repr(C)]
pub enum ScalarType {
    DATA_U8 = 0,
    DATA_F16,
    DATA_BF16,
    DATA_F32,
    DATA_F64,
    DATA_U32,
    DATA_I64,

    DATA_UNSUPPORTED = 100,
}

#[repr(C)]
pub enum TopkType {
    AF_TOPK_MIN = 1,
    ///< Top k min values
    AF_TOPK_MAX = 2,
    ///< Top k max values
    AF_TOPK_STABLE = 4,
    ///< Preserve order of indices for equal values
    AF_TOPK_STABLE_MIN = 5,
    ///< Top k min with stable indices
    AF_TOPK_STABLE_MAX = 6,
    ///< Top k max with stable indices
    AF_TOPK_DEFAULT = 0, // Default option (max)
}

use candle::{cuda_backend::cudarc::driver::DeviceRepr, DType, Shape, Tensor};

use crate::cuda_ext::get_tensor_cuda_device_ptr;

pub fn get_scalar_type(dtype: DType) -> ScalarType {
    match dtype {
        DType::BF16 => ScalarType::DATA_BF16,
        DType::U8 => ScalarType::DATA_U8,
        DType::U32 => ScalarType::DATA_U32,
        DType::I64 => ScalarType::DATA_I64,
        DType::F16 => ScalarType::DATA_F16,
        DType::F32 => ScalarType::DATA_F32,
        DType::F64 => ScalarType::DATA_F64,
        _ => ScalarType::DATA_UNSUPPORTED,
    }
}

impl CTensorView {
    pub fn nil() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            shape: [1, 1, 1, 1],
            stride: [1, 1, 1, 1],
            ndim: 0,
            dtype: 0,
        }
    }
    pub fn from(t: &Tensor, column_major: bool) -> candle::Result<CTensorView> {
        let ptr = get_tensor_cuda_device_ptr(t)?.as_ffi_ptr();
        let mut dims = (1_i64, 1, 1, 1);
        if t.shape().dims().len() == 1 {
            dims.0 = t.shape().dims()[0] as i64;
        } else if t.shape().dims().len() == 2 {
            if column_major {
                dims.0 = t.shape().dims()[1] as i64;
                dims.1 = t.shape().dims()[0] as i64;
            } else {
                dims.0 = t.shape().dims()[0] as i64;
                dims.1 = t.shape().dims()[1] as i64;
            }
        } else if column_major {
            candle::bail!("Not supported shapes:{:?}", t.shape());
        } else if t.shape().dims().len() == 3 {
            dims.0 = t.shape().dims()[0] as i64;
            dims.1 = t.shape().dims()[1] as i64;
            dims.2 = t.shape().dims()[2] as i64;
        } else {
            dims.0 = t.shape().dims()[0] as i64;
            dims.1 = t.shape().dims()[1] as i64;
            dims.2 = t.shape().dims()[2] as i64;
            dims.3 = t.shape().dims()[3] as i64;
        }
        let mut strides = (1_i64, 1, 1, 1);
        if !column_major {
            strides.0 = t.stride()[0] as i64;
            if t.stride().len() > 1 {
                strides.1 = t.stride()[1] as i64;
            }
            if t.stride().len() > 2 {
                strides.2 = t.stride()[2] as i64;
            }
            if t.stride().len() > 3 {
                strides.3 = t.stride()[3] as i64;
            }
        }

        Ok(CTensorView {
            ptr,
            shape: [dims.0, dims.1, dims.2, dims.3],
            stride: [strides.0, strides.1, strides.2, strides.3],
            ndim: t.dims().len() as i32,
            dtype: get_scalar_type(t.dtype()) as u32,
        })
    }
}

impl CShapeView {
    pub fn new(s: &Shape) -> Self {
        let mut dims = (1_i64, 1, 1, 1);
        dims.0 = s.dims()[0] as i64;
        if s.dims().len() > 1 {
            dims.1 = s.dims()[1] as i64;
        }
        if s.dims().len() > 2 {
            dims.2 = s.dims()[2] as i64;
        }
        if s.dims().len() > 3 {
            dims.3 = s.dims()[3] as i64;
        }

        CShapeView {
            shape: [dims.0, dims.1, dims.2, dims.3],
            ndim: s.dims().len() as i32,
        }
    }
}
