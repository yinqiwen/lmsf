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

pub fn cuda_copy(dst: &Tensor, src: &Tensor) -> candle_core::Result<()> {
    //dst.device().same_device(rhs)
    todo!("")
}
