use std::collections::HashMap;

use candle::{DType, Tensor};
use candle_nn::VarBuilder;
use common::{DefaultTensorCreator, TensorCreator};

use crate::model_executor::parallel::ParallelState;

use super::WeightRegistry;

pub type WeightDesc = (&'static str, usize, usize);
pub trait Layer: Sized {
    // fn get_weights(input_size: usize, output_size: usize, dtype: DType) -> Vec<WeightRegistry>;

    // fn weight_loader(
    //     tensor_reg: &WeightRegistry,
    //     loaded_weight: Tensor,
    //     parallel_state: &ParallelState,
    // ) -> candle_core::Result<Tensor>;

    // fn from_weights(weights: HashMap<&'static str, Tensor>) -> candle_core::Result<Self>;

    fn forward_<C: TensorCreator>(
        &self,
        x: &Tensor,
        tensor_creator: &mut C,
    ) -> candle::Result<Tensor>;

    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        let mut default_creator = DefaultTensorCreator {};
        self.forward_(x, &mut default_creator)
    }

    // fn load(
    //     vb: &VarBuilder,
    //     input_size: usize,
    //     output_size: usize,
    //     parallel_state: &ParallelState,
    // ) -> candle_core::Result<Self> {
    //     let weights = Self::get_weights(input_size, output_size, vb.dtype());
    //     let mut loaded_weights = HashMap::new();
    //     for weight in &weights {
    //         let loaded_weight = vb.get(weight.shape.clone(), weight.name)?;

    //         let loaded_weight = Self::weight_loader(&weight, loaded_weight, parallel_state)?;
    //         loaded_weights.insert(weight.name, loaded_weight);
    //     }
    //     Self::from_weights(loaded_weights)
    // }
}
