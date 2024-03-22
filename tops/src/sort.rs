use candle::cuda_backend::cudarc::driver::sys::CUstream;
use candle::{DType, Tensor};
use common::ffi::CTensorView;
use common::{DefaultTensorCreator, TensorCreator};

extern "C" {
    // fn cuda_sort_tensor(
    //     input: CTensorView,
    //     dim: c_uint,
    //     ascend: bool,
    //     stream: CUstream,
    //     output: CTensorView,
    //     indices: CTensorView,
    // );

    fn cuda_argsort_tensor(
        input: CTensorView,
        ascend: bool,
        indices: CTensorView,
        stream: CUstream,
    );

    // fn cuda_dim_gather_tensor(
    //     input: CTensorView,
    //     dim: c_uint,
    //     indices: CTensorView,
    //     stream: CUstream,
    //     output: CTensorView,
    // );
}
pub fn cuda_sort2(
    t: Tensor,
    dim: usize,
    descending: bool,
    stream: CUstream,
) -> candle::Result<(Tensor, Tensor)> {
    let mut default_creator = DefaultTensorCreator {};
    cuda_sort2_(t, dim, descending, &mut default_creator, stream)
}
fn cuda_sort2_<F: TensorCreator>(
    t: Tensor,
    dim: usize,
    descending: bool,
    tensor_creator: &mut F,
    stream: CUstream,
) -> candle::Result<(Tensor, Tensor)> {
    //let out = tensor_creator.new(t.shape(), t.dtype(), t.device(), false)?;
    let indices = tensor_creator.new(t.shape(), DType::U32, t.device(), false)?;
    let input_view = common::ffi::CTensorView::from(&t, false)?;
    //let output_view = common::ffi::CTensorView::from(&out, false)?;
    let indices_view = common::ffi::CTensorView::from(&indices, false)?;
    if dim == t.shape().dims().len() - 1 {
        unsafe {
            cuda_argsort_tensor(
                input_view.clone(),
                !descending,
                indices_view.clone(),
                stream,
            );
            //cuda_dim_gather_tensor(input_view, dim as c_uint, indices_view, stream, output_view);
        };
        let output = t.gather(&indices, dim)?;
        Ok((output, indices))
    } else {
        unimplemented!("Unsupported non last dim sort")
    }
}

pub fn cuda_sort(
    t: Tensor,
    dim: usize,
    descending: bool,
    stream: CUstream,
) -> candle::Result<(Tensor, Tensor)> {
    let mut default_creator = DefaultTensorCreator {};
    cuda_sort_(t, dim, descending, &mut default_creator, stream)
}

pub fn cuda_sort_<F: TensorCreator>(
    t: Tensor,
    dim: usize,
    descending: bool,
    tensor_creator: &mut F,
    stream: CUstream,
) -> candle::Result<(Tensor, Tensor)> {
    cuda_sort2_(t, dim, descending, tensor_creator, stream)

    // let out = tensor_creator.new(t.shape(), t.dtype(), t.device(), false)?;
    // unsafe_tensor_dtod_copy(&out, &t)?;
    // let indices = tensor_creator.new(out.shape(), DType::U32, t.device(), true)?;
    // //let out = t.copy()?;
    // // let indices = Tensor::zeros(out.shape(), DType::U32, t.device())?;
    // let dim = get_column_major_dim(t.shape(), dim)?;

    // let input_view = common::ffi::CTensorView::from(&t, true)?;
    // let output_view = common::ffi::CTensorView::from(&out, true)?;
    // let indices_view = common::ffi::CTensorView::from(&indices, true)?;
    // unsafe {
    //     cuda_sort_tensor(
    //         input_view,
    //         dim as u32,
    //         !descending,
    //         stream,
    //         output_view,
    //         indices_view,
    //     );
    // };
    // Ok((out, indices))
}

#[test]
fn test_sort() -> candle::Result<()> {
    use candle::Device;
    let device = Device::new_cuda(0)?;
    let a = Tensor::new(
        &[
            0.6719, -0.2162, 2.3332, 0.0608, 0.9497, -0.5793, 0.6058, 0.0061, 0.3343, -0.5071,
            1.0960, 0.9553,
        ],
        &device,
    )?;
    let a = a.reshape((3, 4))?;

    let stream: CUstream = std::ptr::null_mut();
    let (out, indices) = cuda_sort(a, 1, false, stream)?;
    println!("out:{:?}", out.to_string());
    println!("indices:{:?}", indices.to_string());

    let a = Tensor::new(
        &[
            0.6719_f32, -0.2162, 2.3332, 0.0608, 0.9497, -0.5793, 0.6058, 0.0061, 0.3343, -0.5071,
            1.0960, 0.9553,
        ],
        &device,
    )?;
    println!("{:?}", a.dtype());
    let a = a.reshape((3, 4))?;
    let (out, indices) = cuda_sort2(a, 1, false, stream)?;
    println!("out:{:?}", out.to_string());
    println!("indices:{:?}", indices.to_string());

    Ok(())
}
