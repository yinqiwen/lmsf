use candle_core::{DType, Device, Shape, Tensor};

pub trait TensorCreator {
    fn new<S: Into<Shape>>(
        &mut self,
        shape: S,
        dtype: DType,
        device: &Device,
        zero: bool,
    ) -> candle_core::Result<Tensor>;

    fn like(&mut self, t: &Tensor, device: &Device) -> candle_core::Result<Tensor> {
        self.new(t.shape(), t.dtype(), device, false)
    }

    fn default() -> DefaultTensorCreator {
        DefaultTensorCreator {}
    }
}

pub struct DefaultTensorCreator {}

impl TensorCreator for DefaultTensorCreator {
    fn new<S: Into<Shape>>(
        &mut self,
        shape: S,
        dtype: DType,
        device: &Device,
        _zero: bool,
    ) -> candle_core::Result<Tensor> {
        Tensor::zeros(shape, dtype, device)
    }
}
