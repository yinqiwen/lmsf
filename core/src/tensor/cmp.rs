use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DeviceRepr, LaunchAsync};
use candle_core::cuda_backend::WrapErr;
use candle_core::{
    backend::BackendStorage, cuda_backend::cudarc::driver::LaunchConfig, shape::Dim, CpuStorage,
    CudaStorage, DType, Layout, Shape, Storage,
};
use candle_core::{
    scalar::{TensorOrScalar, TensorScalar},
    Device, Tensor,
};

use common::cuda_ext::get_tensor_cuda_device_ptr;
use common::{DefaultTensorCreator, TensorCreator};

#[derive(Clone, Copy, PartialEq, Eq)]
enum CmpOp {
    Eq,
    Ne,
    Le,
    Ge,
    Lt,
    Gt,
}

fn kernel_name(root: &str, dtype: DType) -> String {
    let dtype = dtype.as_str();
    format!("{root}_{dtype}")
}

fn cuda_cmp<T: TensorOrScalar>(
    lhs: &Tensor,
    rhs: T,
    op: CmpOp,
    dst: &Tensor,
) -> candle_core::Result<()> {
    if dst.dtype() != DType::U8 {
        return Err(candle_core::Error::UnexpectedDType {
            msg: "invalid dtype",
            expected: DType::U8,
            got: dst.dtype(),
        }
        .bt())?;
    }
    if dst.shape() != lhs.shape() {
        return Err(candle_core::Error::ShapeMismatchBinaryOp {
            op: "cuda_cmp (lhs,rhs,dst)",
            lhs: lhs.shape().clone(),
            rhs: dst.shape().clone(),
        }
        .bt())?;
    }
    let device = match lhs.device() {
        Device::Cuda(cuda_dev) => cuda_dev,
        _ => {
            candle_core::bail!("unexpected device")
        }
    };
    let rhs = match rhs.to_tensor_scalar()? {
        TensorScalar::Tensor(rhs) => rhs,
        TensorScalar::Scalar(rhs) => rhs
            .to_dtype(lhs.dtype())?
            .to_device(lhs.device())?
            .broadcast_as(lhs.shape())?,
    };
    let shape = lhs.shape();
    let dims = shape.dims();
    let elem_count = shape.elem_count();
    let cfg = LaunchConfig::for_num_elems(elem_count as u32);
    let dims_and_strides = device
        .htod_copy([dims, lhs.stride(), rhs.stride()].concat())
        .w()?;
    // let lhs = &lhs.slice(lhs_l.start_offset()..);
    // let rhs = &rhs.slice(rhs_l.start_offset()..);
    let name = match op {
        CmpOp::Eq => "eq",
        CmpOp::Ne => "ne",
        CmpOp::Lt => "lt",
        CmpOp::Le => "le",
        CmpOp::Gt => "gt",
        CmpOp::Ge => "ge",
    };
    let dtype = lhs.dtype();

    lhs.layout().start_offset();
    let lhs = get_tensor_cuda_device_ptr(lhs)?;
    let rhs = get_tensor_cuda_device_ptr(&rhs)?;
    let out = get_tensor_cuda_device_ptr(dst)?;
    let func = device.get_or_load_func(&kernel_name(name, dtype), candle_kernels::BINARY)?;
    let params = (elem_count, dims.len(), &dims_and_strides, lhs, rhs, out);
    // SAFETY: ffi
    unsafe { func.launch(cfg, params) }.w()?;
    Ok(())
}

// pub fn cuda_gt<T: TensorOrScalar>(lhs: &Tensor, rhs: T, dst: &Tensor) -> candle_core::Result<()> {
//     cuda_cmp(lhs, rhs, CmpOp::Gt, dst)
// }

pub fn cuda_gt_<T: TensorOrScalar, F: TensorCreator>(
    lhs: &Tensor,
    rhs: T,
    tensor_creator: &mut F,
) -> candle_core::Result<Tensor> {
    let dst = tensor_creator.new(lhs.shape(), DType::U8, lhs.device(), false)?;
    cuda_cmp(lhs, rhs, CmpOp::Gt, &dst)?;
    Ok(dst)
}

#[test]
fn test_cuda_gt() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let src = Tensor::ones((1, 128), DType::I64, &device)?;
    let dst = Tensor::ones((1, 128), DType::U8, &device)?;
    println!("stride:{:?}", src.stride());

    match 1_i64.to_tensor_scalar()? {
        TensorScalar::Scalar(v) => {
            let v = v.to_dtype(src.dtype())?.broadcast_as(src.shape())?;
            println!("stride:{:?}", v.stride());
        }
        _ => {}
    }

    cuda_gt(&src, 0_i64, &dst)?;
    println!("dst:{:?}", dst.to_string());
    Ok(())
}
