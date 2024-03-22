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
    DataU8 = 0,
    DataF16,
    DataBF16,
    DataF32,
    DataF64,
    DataU32,
    DataI64,

    DataUnsupported = 100,
}

#[repr(C)]
pub enum TopkType {
    AfTopkMin = 1,
    ///< Top k min values
    AfTopkMax = 2,
    ///< Top k max values
    AfTopkStable = 4,
    ///< Preserve order of indices for equal values
    AfTopkStableMin = 5,
    ///< Top k min with stable indices
    AfTopkStableMax = 6,
    ///< Top k max with stable indices
    AfTopkDefault = 0, // Default option (max)
}

use candle::{DType, Shape, Tensor};

use crate::cuda_ext::get_tensor_cuda_device_ptr;

pub fn get_scalar_type(dtype: DType) -> ScalarType {
    match dtype {
        DType::BF16 => ScalarType::DataBF16,
        DType::U8 => ScalarType::DataU8,
        DType::U32 => ScalarType::DataU32,
        DType::I64 => ScalarType::DataI64,
        DType::F16 => ScalarType::DataF16,
        DType::F32 => ScalarType::DataF32,
        DType::F64 => ScalarType::DataF64,
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
