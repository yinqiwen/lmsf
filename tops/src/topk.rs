use candle::cuda_backend::cudarc::driver::sys::CUstream;
use candle::{shape::Dim, DType, Shape, Tensor};
use common::ffi::CTensorView;
use common::{DefaultTensorCreator, TensorCreator};
use libc::c_int;

extern "C" {
    // fn cuda_topk_tensor(
    //     input: CTensorView,
    //     topk: c_int,
    //     dim: c_int,
    //     topk_type: c_int,
    //     stream: CUstream,
    //     output: CTensorView,
    //     indices: CTensorView,
    // );
    fn cuda_topk_indices(
        input: CTensorView,
        k: c_int,
        descend: bool,
        stream: CUstream,
        indices: CTensorView,
    );
    // fn cuda_dim_gather_tensor(
    //     input: CTensorView,
    //     dim: c_uint,
    //     indices: CTensorView,
    //     stream: CUstream,
    //     output: CTensorView,
    // );
}

pub fn cuda_topk(
    t: &Tensor,
    topk: usize,
    dim: usize,
    _stream: CUstream,
) -> candle::Result<(Tensor, Tensor)> {
    // let dim = get_column_major_dim(t.shape(), dim)?;

    // let mut indices_dims = Vec::from(t.dims());
    // *indices_dims.last_mut().unwrap() = topk;
    // let indices = Tensor::zeros(indices_dims.clone(), DType::U32, t.device())?;
    // let out = Tensor::zeros(indices_dims, t.dtype(), t.device())?;

    // let input_view = CTensorView::from(t, true)?;
    // let output_view = CTensorView::from(&out, true)?;
    // let indices_view = CTensorView::from(&indices, true)?;
    // unsafe {
    //     cuda_topk_tensor(
    //         input_view,
    //         topk as i32,
    //         dim as i32,
    //         0,
    //         stream,
    //         output_view,
    //         indices_view,
    //     );
    // };
    // Ok((out, indices))
    cuda_topk2(t, topk, dim)
}

pub fn cuda_topk2<D: Dim>(t: &Tensor, k: usize, dim: D) -> candle::Result<(Tensor, Tensor)> {
    let mut default_creator = DefaultTensorCreator {};
    cuda_topk2_(t, k, dim, &mut default_creator, std::ptr::null_mut())
}

pub fn cuda_topk2_<F: TensorCreator, D: Dim>(
    t: &Tensor,
    k: usize,
    dim: D,
    tensor_creator: &mut F,
    stream: CUstream,
) -> candle::Result<(Tensor, Tensor)> {
    let dim = if t.dims().len() == 1 {
        0
    } else {
        dim.to_index(t.shape(), "cuda_topk")?
    };
    let mut indices_shape = Vec::from(t.shape().dims());
    let last_idx = indices_shape.len() - 1;
    let last_dim = indices_shape[last_idx];
    indices_shape[last_idx] = std::cmp::min(last_dim, k);
    let indices = tensor_creator.new(
        Shape::from_dims(&indices_shape),
        DType::U32,
        t.device(),
        false,
    )?;
    // let output = tensor_creator.new(
    //     Shape::from_dims(&indices_shape),
    //     t.dtype(),
    //     t.device(),
    //     false,
    // )?;
    let input_view = CTensorView::from(t, false)?;
    let indices_view = CTensorView::from(&indices, false)?;
    // let output_view = CTensorView::from(&output, false)?;
    if dim == t.dims().len() - 1 {
        unsafe {
            cuda_topk_indices(
                input_view.clone(),
                k as i32,
                true,
                stream,
                indices_view.clone(),
            );

            //cuda_dim_gather_tensor(input_view, dim as c_uint, indices_view, stream, output_view);
        };

        let output = t.gather(&indices, dim)?;
        Ok((output, indices))
    } else {
        unimplemented!("Unsupported non last dim topk")
    }
    // if (axis == input->ndim() - 1) {
    //     if (largest) {
    //       indices = JUST(OpInterpUtil::Dispatch<Tensor>(*op_, {input}, attrs));
    //     } else {
    //       auto neg_input = JUST(ScalarMul(input, -1, false));
    //       indices = JUST(OpInterpUtil::Dispatch<Tensor>(*op_, {neg_input}, attrs));
    //     }
    //     values = JUST(DimGather(input, axis, indices, false));

    //   } else {
    //     auto perm = JUST(GetPermWhenTransposeAxisToLastDim(input->ndim(), dim_value));
    //     auto x = JUST(Transpose(input, *perm));
    //     if (largest) {
    //       indices = JUST(OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs));
    //     } else {
    //       auto neg_input = JUST(ScalarMul(x, -1, false));
    //       indices = JUST(OpInterpUtil::Dispatch<Tensor>(*op_, {neg_input}, attrs));
    //     }
    //     auto inversed_perm = JUST(GetInversedPerm(*perm));
    //     indices = JUST(Transpose(indices, *inversed_perm));
    //     values = JUST(DimGather(input, axis, indices, false));
    //   }
}
#[test]
fn test_topk() -> candle::Result<()> {
    use candle::{Device, D};
    let device = Device::new_cuda(0)?;
    let test = Tensor::rand(1_f32, 10.0, (2, 10), &device)?;
    println!("{}", test.to_string());

    let (out, indices) = cuda_topk2(&test, 2, D::Minus1)?;
    println!("out:{}", out.to_string());
    println!("indices:{}", indices.to_string());
    Ok(())
}
