use candle::cuda_backend::WrapErr;
use candle::{
    cuda_backend::cudarc::driver::{LaunchAsync, LaunchConfig},
    shape::Dim,
    DType,
};
use candle::{Device, Tensor};
use common::cuda_ext::get_tensor_cuda_device_ptr;

fn scatter_add_kernel_name(root: &str, dtype0: DType, dtype1: DType) -> String {
    let dtype0 = dtype0.as_str();
    let dtype1 = dtype1.as_str();
    format!("{root}_{dtype0}_{dtype1}")
}
pub fn cuda_scatter_add<D: Dim>(
    dst: &Tensor,
    index: &Tensor,
    src: &Tensor,
    dim: D,
) -> candle::Result<()> {
    let dim = dim.to_index(dst.shape(), "scatter-add")?;
    let source_dims = src.dims();
    let self_dims = dst.dims();
    let mismatch = if source_dims.len() != self_dims.len() {
        true
    } else {
        let mut mismatch = false;
        for (i, (&d1, &d2)) in self_dims.iter().zip(source_dims.iter()).enumerate() {
            if i != dim && d1 != d2 {
                mismatch = true;
                break;
            }
        }
        mismatch
    };
    if mismatch {
        return Err(candle::Error::ShapeMismatchBinaryOp {
            op: "cuda-scatter-add (self, src)",
            lhs: dst.shape().clone(),
            rhs: src.shape().clone(),
        }
        .bt())?;
    }
    if index.dims() != src.dims() {
        return Err(candle::Error::ShapeMismatchBinaryOp {
            op: "cuda-scatter-add (indexes, src)",
            lhs: index.shape().clone(),
            rhs: src.shape().clone(),
        }
        .bt())?;
    }

    let device = match dst.device() {
        Device::Cuda(cuda_dev) => cuda_dev,
        _ => {
            candle::bail!("unexpected device")
        }
    };
    let (_ids_o1, _ids_o2) = match index.layout().contiguous_offsets() {
        Some(o12) => o12,
        None => return Err(candle::Error::RequiresContiguous { op: "scatter-add" }.bt())?,
    };
    let (_src_o1, _src_o2) = match src.layout().contiguous_offsets() {
        Some(o12) => o12,
        None => return Err(candle::Error::RequiresContiguous { op: "scatter-add" }.bt())?,
    };

    let func = device.get_or_load_func(
        &scatter_add_kernel_name("sa", index.dtype(), dst.dtype()),
        candle_kernels::INDEXING,
    )?;
    let left_sz: usize = src.dims()[..dim].iter().product();
    let right_sz: usize = src.dims()[dim + 1..].iter().product();
    let src_dim_sz = src.dims()[dim];
    let dst_dim_sz = dst.dims()[dim];
    let cfg = LaunchConfig::for_num_elems((left_sz * right_sz) as u32);
    let dst = get_tensor_cuda_device_ptr(dst)?;
    let ids = get_tensor_cuda_device_ptr(index)?;
    let src = get_tensor_cuda_device_ptr(src)?;

    let params = (ids, src, dst, left_sz, src_dim_sz, dst_dim_sz, right_sz);
    // SAFETY: ffi.
    unsafe { func.launch(cfg, params) }.w()?;
    Ok(())
}

#[test]
fn test_inplace_scatter_add() -> candle::Result<()> {
    let device = Device::new_cuda(0)?;
    let src = Tensor::ones((2, 5), DType::I64, &device)?;
    let index = Tensor::new(vec![vec![0_i64, 1, 2, 0, 0], vec![0, 1, 2, 2, 2]], &device)?;
    let dst = Tensor::zeros((3, 5), DType::I64, &device)?;
    println!("afer inplace scatter add:{}", dst.to_string());
    let _x = dst.scatter_add(&index, &src, 0)?;
    // println!("afer inplace scatter add:{}", x.to_string());
    cuda_scatter_add(&dst, &index, &src, 0)?;
    println!("afer inplace scatter add:{}", dst.to_string());

    Ok(())
}
