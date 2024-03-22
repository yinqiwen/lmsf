use candle::cuda_backend::cudarc::driver::sys::CUstream;

use candle::{Shape, Tensor};
use common::ffi::{CShapeView, CTensorView};
use common::{DefaultTensorCreator, TensorCreator};
use libc::c_float;

extern "C" {
    // void cuda_oneflow_rms_norm(CTensorView x, CTensorView weight, CShapeView normalized_shape, float epsilon,
    //     cudaStream_t stream, CTensorView inv_rms, CTensorView y)
    fn cuda_oneflow_rms_norm(
        x: CTensorView,
        weight: CTensorView,
        normalized_shape: CShapeView,
        epsilon: c_float,
        stream: CUstream,
        inv_rms: CTensorView,
        y: CTensorView,
    );
}

#[derive(Clone, Debug)]
pub struct RmsNorm {
    weight: Option<Tensor>,
    normalized_shape: Shape,
    eps: f64,
    _elementwise_affine: bool,
}

impl RmsNorm {
    pub fn new<S: Into<Shape>>(s: S, eps: f64, elementwise_affine: bool) -> Self {
        Self {
            weight: None,
            normalized_shape: s.into(),
            eps,
            _elementwise_affine: elementwise_affine,
        }
    }
    pub fn load<S: Into<Shape>>(s: S, eps: f64, vb: candle_nn::VarBuilder) -> candle::Result<Self> {
        let normalized_shape = s.into();
        let weight = vb.get_with_hints(
            normalized_shape.clone(),
            "weight",
            candle_nn::Init::Const(1.),
        )?;
        //let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        // let inner = candle_nn::rms_norm(size, eps, vb)?;
        Ok(Self {
            weight: Some(weight),
            normalized_shape,
            eps,
            _elementwise_affine: true,
        })
    }
    pub fn forward_<F: TensorCreator>(
        &self,
        xs: &Tensor,
        tensor_creator: &mut F,
    ) -> candle::Result<Tensor> {
        let y = tensor_creator.new(xs.shape(), xs.dtype(), xs.device(), false)?;
        let y_view = common::ffi::CTensorView::from(&y, false)?;
        let x_view = common::ffi::CTensorView::from(xs, false)?;
        let weight_view = match &self.weight {
            Some(w) => common::ffi::CTensorView::from(w, false)?,
            None => CTensorView::nil(),
        };
        let normalized_shape = CShapeView::new(&self.normalized_shape);
        let batch_ndim = xs.dims().len() - self.normalized_shape.dims().len();
        let mut batch_dims = Vec::new();
        for i in 0..batch_ndim {
            batch_dims.push(xs.dims()[i]);
        }
        let inv_rms_shape = Shape::from_dims(&batch_dims);
        let inv_rms = tensor_creator.new(inv_rms_shape, xs.dtype(), xs.device(), false)?;
        let inv_rms_view = common::ffi::CTensorView::from(&inv_rms, false)?;

        unsafe {
            cuda_oneflow_rms_norm(
                x_view,
                weight_view,
                normalized_shape,
                self.eps as f32,
                std::ptr::null_mut(),
                inv_rms_view,
                y_view,
            );
        }
        Ok(y)
    }
}

impl candle::Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> candle::Result<Tensor> {
        let mut default_creator = DefaultTensorCreator {};
        self.forward_(xs, &mut default_creator)
    }
}

#[test]
fn test_rms_norm() -> candle::Result<()> {
    use candle::{Device, Module};
    let device = Device::new_cuda(0)?;
    let a = Tensor::new(
        &[
            -0.16046895,
            -1.03667831,
            -0.34974465,
            0.26505867,
            -1.24111986,
            -0.53806001,
            1.72426331,
            0.43572459,
            -0.77390957,
            -0.42610624,
            0.16398858,
            -1.35760343,
            1.07541728,
            0.11008703,
            0.26361224,
            -0.48663723,
        ],
        &device,
    )?
    .reshape((2, 2, 2, 2))?;
    let rms_norm = RmsNorm::new(2, 1e-5, true);
    let r = rms_norm.forward(&a)?;
    println!("r:{}", r.to_string());

    Ok(())
}
