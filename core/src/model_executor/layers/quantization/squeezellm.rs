use super::QuantizationConfig;
use crate::model_executor::layers::linear::LinearWeights;
use crate::model_executor::layers::WeightRegistry;
use candle::{DType, Device, Shape, Tensor, D};
use common::{DefaultTensorCreator, TensorCreator};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SqueezeLLMConfig {
    weight_bits: usize,
    pack_factor: usize,
}

impl QuantizationConfig for SqueezeLLMConfig {
    fn get_name() -> &'static str {
        "squeezellm"
    }

    fn get_supported_act_dtypes() -> Vec<DType> {
        vec![DType::F16]
    }

    fn get_config_filenames() -> Vec<&'static str> {
        vec!["quant_config.json"]
    }

    fn get_min_capability() -> u32 {
        70
    }

    fn get_scaled_act_names() -> Vec<&'static str> {
        vec![]
    }

    fn from_json(mut val: serde_json::Value) -> candle::Result<Self> {
        let cfg_dict = val
            .as_object()
            .ok_or(candle::Error::Msg("invalid json".to_string()))?;

        let wbits = if let Some(wbits) = cfg_dict.get("wbits") {
            wbits
                .as_u64()
                .ok_or(candle::Error::Msg("invalid json with 'wbits'".to_string()))?
        } else {
            return candle::bail!("no wbits found in json");
        } as usize;

        Self::new(wbits)
    }
}

impl SqueezeLLMConfig {
    pub fn new(weight_bits: usize) -> candle::Result<Self> {
        if weight_bits != 4 {
            return candle::bail!("Currently, only 4-bit weight quantization is supported for SqueezeLLM, but got {weight_bits} bits.");
        }
        let pack_factor = 32 / weight_bits;
        Ok(Self {
            weight_bits,
            pack_factor,
        })
    }
}

pub struct SqueezeLLMLinearWeights {
    quant_config: SqueezeLLMConfig,
    qweight: Tensor,
    lookup_table: Tensor,
    bias: Option<Tensor>,
}

impl LinearWeights for SqueezeLLMLinearWeights {
    type Config = SqueezeLLMConfig;
    fn from(
        mut weights: HashMap<&'static str, Tensor>,
        config: SqueezeLLMConfig,
    ) -> candle::Result<Self> {
        let qweight = if let Some(weight) = weights.remove("qweight") {
            weight
        } else {
            candle::bail!("missing 'qweight' in weights")
        };
        let lookup_table = if let Some(weight) = weights.remove("lookup_table") {
            weight
        } else {
            candle::bail!("missing 'lookup_table' in weights")
        };

        let bias = if let Some(weight) = weights.remove("bias") {
            Some(weight)
        } else {
            None
        };

        Ok(Self {
            quant_config: config,
            qweight,
            lookup_table,
            bias,
        })
    }

    fn get_descs(
        input_size: usize,
        output_size: usize,
        params_dtype: DType,
        config: &SqueezeLLMConfig,
    ) -> Vec<WeightRegistry> {
        let pack_factor = config.pack_factor;

        let qweight_shape = Shape::from_dims(&[input_size, output_size]);
        let qweight_attrs = HashMap::from([
            ("input_dim", 0_usize),
            ("output_dim", 1),
            ("packed_dim", 0),
            ("pack_factor", pack_factor),
        ]);

        let lookup_table_shape =
            Shape::from_dims(&[output_size, config.weight_bits * config.weight_bits]);
        let lookup_table_attrs = HashMap::from([("output_dim", 0)]);

        vec![
            WeightRegistry::new("qweight", qweight_shape, DType::U32, qweight_attrs),
            WeightRegistry::new(
                "lookup_table",
                lookup_table_shape,
                params_dtype,
                lookup_table_attrs,
            ),
        ]
    }

    fn apply<C: TensorCreator>(
        &self,
        tensor_creator: &mut C,
        x: &Tensor,
    ) -> candle::Result<Tensor> {
        let last_qweight_shape = self.qweight.dims()[self.qweight.dims().len() - 1];
        let mut out_shape = Vec::from(x.dims());
        out_shape[x.dims().len() - 1] = last_qweight_shape;
        let out_shape = Shape::from_dims(&out_shape);
        let last_x_shape = x.dims()[x.dims().len() - 1];
        let reshaped_x = x.reshape((x.elem_count() / last_x_shape, last_x_shape))?;

        // NOTE: The output tensor should be zero-initialized.
        let out = tensor_creator.new(out_shape.clone(), DType::F16, x.device(), true)?;
        vllm::squeezellm_gemm(&reshaped_x, &self.qweight, &out, &self.lookup_table)?;

        let out = match self.bias.as_ref() {
            None => out,
            Some(bias) => out.broadcast_add(bias)?,
        };

        out.reshape(out_shape)
    }
}
