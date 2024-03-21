use super::QuantizationConfig;
use crate::model_executor::layers::linear::LinearWeights;
use crate::model_executor::layers::WeightRegistry;
use candle::{DType, Device, IndexOp, Shape, Tensor, D};
use common::{DefaultTensorCreator, TensorCreator};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct AWQConfig {
    weight_bits: usize,
    group_size: usize,
    zero_point: bool,
    pub(crate) pack_factor: usize,
}

impl QuantizationConfig for AWQConfig {
    fn get_name() -> &'static str {
        "awq"
    }

    fn get_supported_act_dtypes() -> Vec<DType> {
        vec![DType::F16]
    }

    fn get_config_filenames() -> Vec<&'static str> {
        vec!["quant_config.json", "quantize_config.json"]
    }

    fn get_min_capability() -> u32 {
        75
    }

    fn get_scaled_act_names() -> Vec<&'static str> {
        vec!["gelu", "gelu_fast", "gelu_new", "gelu_pytorch_tanh"]
    }

    fn from_json(mut val: serde_json::Value) -> candle::Result<Self> {
        let cfg_dict = val
            .as_object()
            .ok_or(candle::Error::Msg("invalid json".to_string()))?;
        let weight_bits = if let Some(w_bit) = cfg_dict.get("w_bit") {
            w_bit
                .as_u64()
                .ok_or(candle::Error::Msg("invalid json with 'w_bit'".to_string()))?
        } else {
            if let Some(bits) = cfg_dict.get("bits") {
                bits.as_u64()
                    .ok_or(candle::Error::Msg("invalid json with 'bits'".to_string()))?
            } else {
                return candle::bail!("no w_bit/bits found in json");
            }
        } as usize;

        let group_size = if let Some(q_group_size) = cfg_dict.get("q_group_size") {
            q_group_size.as_u64().ok_or(candle::Error::Msg(
                "invalid json with 'q_group_size'".to_string(),
            ))?
        } else {
            if let Some(group_size) = cfg_dict.get("group_size") {
                group_size.as_u64().ok_or(candle::Error::Msg(
                    "invalid json with 'group_size'".to_string(),
                ))?
            } else {
                return candle::bail!("no q_group_size/group_size found in json");
            }
        } as usize;

        let zero_point = if let Some(zero_point) = cfg_dict.get("zero_point") {
            zero_point.as_bool().ok_or(candle::Error::Msg(
                "invalid json with 'zero_point'".to_string(),
            ))?
        } else {
            return candle::bail!("no q_group_size/group_size found in json");
        };

        Self::new(weight_bits, group_size, zero_point)
    }
}

impl AWQConfig {
    pub fn new(weight_bits: usize, group_size: usize, zero_point: bool) -> candle::Result<Self> {
        if weight_bits != 4 {
            return candle::bail!("Currently, only 4-bit weight quantization is supported for AWQ, but got {weight_bits} bits.");
        }
        let pack_factor = 32 / weight_bits;
        Ok(Self {
            weight_bits,
            group_size,
            zero_point,
            pack_factor,
        })
    }
}

pub struct AWQLinearWeights {
    quant_config: AWQConfig,
    qweight: Tensor,
    scales: Tensor,
    qzeros: Tensor,
    bias: Option<Tensor>,
}

impl LinearWeights for AWQLinearWeights {
    type Config = AWQConfig;
    fn from(mut weights: HashMap<&'static str, Tensor>, config: AWQConfig) -> candle::Result<Self> {
        let qweight = if let Some(weight) = weights.remove("qweight") {
            weight
        } else {
            candle::bail!("missing 'qweight' in weights")
        };
        let scales = if let Some(weight) = weights.remove("scales") {
            weight
        } else {
            candle::bail!("missing 'scales' in weights")
        };
        let qzeros = if let Some(weight) = weights.remove("qzeros") {
            weight
        } else {
            candle::bail!("missing 'qzeros' in weights")
        };

        let bias = if let Some(weight) = weights.remove("bias") {
            Some(weight)
        } else {
            None
        };

        Ok(Self {
            quant_config: config,
            qweight,
            scales,
            qzeros,
            bias,
        })
    }

    fn get_descs(
        input_size: usize,
        output_size: usize,
        params_dtype: DType,
        config: &AWQConfig,
    ) -> Vec<WeightRegistry> {
        let pack_factor = config.pack_factor;
        let group_size = config.group_size;

        let qweight_shape = Shape::from_dims(&[input_size, output_size / pack_factor]);
        let qweight_attrs = HashMap::from([
            ("input_dim", 0_usize),
            ("output_dim", 1),
            ("packed_dim", 1),
            ("pack_factor", pack_factor),
        ]);

        let qzero_shape = Shape::from_dims(&[input_size / group_size, output_size / pack_factor]);
        let qzero_attrs = HashMap::from([
            ("input_dim", 0_usize),
            ("output_dim", 1),
            ("packed_dim", 1),
            ("pack_factor", pack_factor),
        ]);

        let scales_shape = Shape::from_dims(&[input_size / group_size, output_size]);
        let scales_attrs = HashMap::from([
            ("input_dim", 0_usize),
            ("output_dim", 1),
            // ("packed_dim", 1),
            // ("pack_factor", pack_factor),
        ]);
        vec![
            WeightRegistry::new("qweight", qweight_shape, DType::U32, qweight_attrs),
            WeightRegistry::new("qzeros", qzero_shape, DType::U32, qzero_attrs),
            WeightRegistry::new("scales", scales_shape, params_dtype, scales_attrs),
        ]
    }

    fn apply<C: TensorCreator>(
        &self,
        tensor_creator: &mut C,
        x: &Tensor,
    ) -> candle::Result<Tensor> {
        let pack_factor = self.quant_config.pack_factor;
        let last_x_shape = x.dims()[x.dims().len() - 1];

        let last_qweight_shape = self.qweight.dims()[self.qweight.dims().len() - 1];
        let reshaped_x = x.reshape((x.elem_count() / last_x_shape, last_x_shape))?;
        let mut out_shape = Vec::from(x.dims());
        out_shape[x.dims().len() - 1] = last_qweight_shape * pack_factor;
        let out_shape = Shape::from_dims(&out_shape);

        let fp16_matmul_heuristic_condition = (x.elem_count() / last_x_shape) >= 256;

        let out = if fp16_matmul_heuristic_condition {
            let out = vllm::awq_dequantize(
                &self.qweight,
                &self.scales,
                &self.qzeros,
                0,
                0,
                tensor_creator,
            )?;
            reshaped_x.matmul(&out)?
        } else {
            let out = vllm::awq_gemm(
                &reshaped_x,
                &self.qweight,
                &self.scales,
                &self.qzeros,
                pack_factor as i32,
                tensor_creator,
            )?;

            out
        };

        let out = match self.bias.as_ref() {
            None => out,
            Some(bias) => out.broadcast_add(bias)?,
        };
        out.reshape(out_shape)
    }
}
