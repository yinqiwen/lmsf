use candle::{DType, Device, Tensor};
use candle::{IndexOp, Shape};

use candle_nn::VarBuilder;
use common::{DefaultTensorCreator, TensorCreator};
use std::collections::HashMap;

use crate::model_executor::parallel::ParallelState;
use crate::tensor::cuda_copy;

use super::{Layer, WeightRegistry};

fn adjust_marlin_shard(
    tensor_reg: &WeightRegistry,
    shard_size: usize,
    shard_offset: usize,
) -> (usize, usize) {
    if let Some(marlin_tile_size) = tensor_reg.getattr("marlin_tile_size") {
        (
            shard_size * marlin_tile_size,
            shard_offset * marlin_tile_size,
        )
    } else {
        (shard_size, shard_offset)
    }
}

pub trait LinearWeights: Sized {
    type Config: Clone;
    fn from(weights: HashMap<&'static str, Tensor>, config: Self::Config) -> candle::Result<Self>;
    fn get_descs(
        input_size_per_partition: usize,
        output_size_per_partition: usize,
        input_size: usize,
        output_size: usize,
        params_dtype: DType,
        config: &Self::Config,
    ) -> Vec<WeightRegistry>;

    fn apply<C: TensorCreator>(&self, tensor_creator: &mut C, x: &Tensor)
        -> candle::Result<Tensor>;
}

pub struct UnquantizedLinearWeights {
    cublas: tops::CublasWrapper,
    pub(crate) weight: Tensor,
    bias: Option<Tensor>,
}

impl LinearWeights for UnquantizedLinearWeights {
    type Config = Option<String>;
    fn from(
        mut weights: HashMap<&'static str, Tensor>,
        _config: Option<String>,
    ) -> candle::Result<Self> {
        if let Some(weight) = weights.remove("weight") {
            let bias = weights.remove("bias");
            let stream = match weight.device() {
                Device::Cuda(cuda_dev) => *cuda_dev.cu_stream(),
                _ => {
                    candle::bail!("no cuda dev")
                }
            };

            let cublas = tops::CublasWrapper::new(weight.device(), weight.dtype(), stream)?;
            Ok(Self {
                cublas,
                weight,
                bias,
            })
        } else {
            candle::bail!("missing 'weight' in weights")
        }
    }

    fn get_descs(
        input_size_per_partition: usize,
        output_size_per_partition: usize,
        input_size: usize,
        output_size: usize,
        params_dtype: DType,
        _config: &Option<String>,
    ) -> Vec<WeightRegistry> {
        let full_shape = Shape::from_dims(&[output_size, input_size]);
        let shard_shape = Shape::from_dims(&[output_size_per_partition, input_size_per_partition]);
        let attrs = HashMap::from([("input_dim", 1_usize), ("output_dim", 0)]);
        vec![WeightRegistry::new(
            "weight",
            shard_shape,
            full_shape,
            params_dtype,
            attrs,
        )]
    }
    fn apply<C: TensorCreator>(
        &self,
        tensor_creator: &mut C,
        x: &Tensor,
    ) -> candle::Result<Tensor> {
        let x1 = self.cublas.linear_(x, &self.weight, tensor_creator)?;
        match self.bias.as_ref() {
            None => Ok(x1),
            Some(bias) => x1.broadcast_add(bias),
        }
    }
}

pub struct ReplicatedLinear<W: LinearWeights> {
    weights: Option<W>,
    input_size: usize,
    output_size: usize,
    bias: bool,
    skip_bias_add: bool,
}
impl<W: LinearWeights> ReplicatedLinear<W> {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            weights: None,
            input_size,
            output_size,
            bias: false,
            skip_bias_add: false,
        }
    }
    pub fn with_bias(mut self, v: bool) -> Self {
        self.bias = v;
        self
    }
    pub fn with_skip_bias_add(mut self, v: bool) -> Self {
        self.skip_bias_add = v;
        self
    }
    pub fn load(
        mut self,
        vb: &VarBuilder,
        _parallel_state: &ParallelState,
        config: W::Config,
    ) -> candle::Result<Self> {
        let mut regs = W::get_descs(
            self.input_size,
            self.output_size,
            self.input_size,
            self.output_size,
            vb.dtype(),
            &config,
        );
        if self.bias {
            let bias_shape = Shape::from_dims(&[self.output_size]);
            let bias_attrs = HashMap::from([("output_dim", 0_usize)]);
            let bias_reg = WeightRegistry::new(
                "bias",
                bias_shape.clone(),
                bias_shape,
                vb.dtype(),
                bias_attrs,
            );
            regs.push(bias_reg);
        }
        let mut loaded_weights = HashMap::new();
        for reg in &regs {
            let loaded_weight = vb.get_with_hints_dtype(
                reg.full_shape.clone(),
                reg.name,
                Default::default(),
                reg.dtype,
            )?;
            loaded_weights.insert(reg.name, loaded_weight);
        }
        self.weights = Some(W::from(loaded_weights, config)?);
        Ok(self)
    }

    pub fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        let mut default_creator = DefaultTensorCreator {};
        self.forward_(x, &mut default_creator)
    }
    pub fn forward_<F: TensorCreator>(
        &self,
        x: &Tensor,
        tensor_creator: &mut F,
    ) -> candle::Result<Tensor> {
        match &self.weights {
            Some(weights) => weights.apply(tensor_creator, x),
            None => {
                candle::bail!("ReplicatedLinear is not inited!")
            }
        }
    }
}

pub struct ColumnParallelLinear<W: LinearWeights> {
    weights: Option<W>,
    input_size: usize,
    output_size: usize,
    output_size_per_partition: usize,
    tp_size: usize,
    tp_rank: usize,
    bias: bool,
    skip_bias_add: bool,
    gather_output: bool,
}

impl<W: LinearWeights> ColumnParallelLinear<W> {
    pub fn new(input_size: usize, output_size: usize, parallel_state: &ParallelState) -> Self {
        let tp = parallel_state.get_tensor_model_parallel_world_size();
        Self {
            weights: None,
            tp_size: parallel_state.get_tensor_model_parallel_world_size(),
            tp_rank: parallel_state.get_tensor_model_parallel_rank(),
            output_size_per_partition: output_size / tp,
            input_size,
            output_size,
            bias: false,
            gather_output: false,
            skip_bias_add: false,
        }
    }
    pub fn with_bias(mut self, v: bool) -> Self {
        self.bias = v;
        self
    }
    pub fn with_gather_output(mut self, v: bool) -> Self {
        self.gather_output = v;
        self
    }
    pub fn with_skip_bias_add(mut self, v: bool) -> Self {
        self.skip_bias_add = v;
        self
    }

    fn weight_loader(
        &self,
        tensor_reg: &WeightRegistry,
        loaded_weight: Tensor,
    ) -> candle::Result<Tensor> {
        if let Some(output_dim) = tensor_reg.getattr("output_dim") {
            if self.tp_size == 1 {
                Ok(loaded_weight)
            } else {
                let shard_size = tensor_reg.shard_shape.dims()[output_dim];
                let start_idx = self.tp_rank * shard_size;
                let narrow_loaded_weight =
                    loaded_weight.narrow(output_dim, start_idx, shard_size)?;
                drop(loaded_weight);
                let t = Tensor::zeros_like(&narrow_loaded_weight)?;
                cuda_copy(&t, &narrow_loaded_weight)?;
                drop(narrow_loaded_weight);
                common::cuda_synchronize(t.device());
                Ok(t)
            }
        } else {
            Ok(loaded_weight)
        }
    }

    fn get_weight_desc(&self, config: &W::Config, params_dtype: DType) -> Vec<WeightRegistry> {
        let mut regs = W::get_descs(
            self.input_size,
            self.output_size_per_partition,
            self.input_size,
            self.output_size,
            params_dtype,
            config,
        );
        if self.bias {
            let bias_shape = Shape::from_dims(&[self.output_size]);
            let shard_bias_shape = Shape::from_dims(&[self.output_size_per_partition]);
            let bias_attrs = HashMap::from([("output_dim", 0_usize)]);
            let bias_reg = WeightRegistry::new(
                "bias",
                shard_bias_shape,
                bias_shape,
                params_dtype,
                bias_attrs,
            );
            regs.push(bias_reg);
        }
        regs
    }

    pub fn load(mut self, vb: &VarBuilder, config: W::Config) -> candle::Result<Self> {
        let regs = self.get_weight_desc(&config, vb.dtype());
        let mut loaded_weights = HashMap::new();
        for reg in &regs {
            let loaded_weight = vb.get_with_hints_dtype(
                reg.full_shape.clone(),
                reg.name,
                Default::default(),
                reg.dtype,
            )?;
            let loaded_weight = self.weight_loader(reg, loaded_weight)?;
            loaded_weights.insert(reg.name, loaded_weight);
        }
        self.weights = Some(W::from(loaded_weights, config)?);
        Ok(self)
    }

    pub fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        let mut default_creator = DefaultTensorCreator {};
        self.forward_(x, &mut default_creator)
    }
    pub fn forward_<F: TensorCreator>(
        &self,
        x: &Tensor,
        tensor_creator: &mut F,
    ) -> candle::Result<Tensor> {
        match &self.weights {
            Some(weights) => {
                let output_parallel = weights.apply(tensor_creator, x)?;
                let output_parallel = if self.gather_output {
                    // output = tensor_model_parallel_all_gather(output_parallel)
                    todo!("tensor_model_parallel_all_gather")
                } else {
                    output_parallel
                };
                Ok(output_parallel)
            }
            None => {
                candle::bail!("RowParallelLinear is not inited!")
            }
        }
    }
}

pub struct MergedColumnParallelLinear<W: LinearWeights> {
    linear: ColumnParallelLinear<W>,
    output_sizes: Vec<usize>,
}
impl<W: LinearWeights> MergedColumnParallelLinear<W> {
    pub fn new(input_size: usize, output_sizes: &[usize], parallel_state: &ParallelState) -> Self {
        let sum_output_size = output_sizes.iter().sum::<usize>();
        let linear = ColumnParallelLinear::<W>::new(input_size, sum_output_size, parallel_state);
        Self {
            linear,
            output_sizes: Vec::from(output_sizes),
        }
    }
    pub fn with_bias(mut self, v: bool) -> Self {
        self.linear.bias = v;
        self
    }
    pub fn with_gather_output(mut self, v: bool) -> Self {
        self.linear.gather_output = v;
        self
    }
    pub fn with_skip_bias_add(mut self, v: bool) -> Self {
        self.linear.skip_bias_add = v;
        self
    }

    fn weight_loader(
        &self,
        tensor_reg: &WeightRegistry,
        param: &Tensor,
        loaded_weight: Tensor,
        loaded_shard_id: Option<usize>,
    ) -> candle::Result<()> {
        let tp_rank = self.linear.tp_rank;
        let tp_size = self.linear.tp_size;
        if let Some(shard_id) = loaded_shard_id {
            let (param_data, loaded_weight) = if let Some(output_dim) =
                tensor_reg.getattr("output_dim")
            {
                let mut shard_offset =
                    self.output_sizes[..shard_id].iter().sum::<usize>() / tp_size;
                let mut shard_size = self.output_sizes[shard_id] / tp_size;
                if let Some(packed_dim) = tensor_reg.getattr("packed_dim") {
                    let pack_factor = tensor_reg.getattr("pack_factor").unwrap();
                    if packed_dim == output_dim {
                        shard_size = shard_size / pack_factor;
                        shard_offset = shard_offset / pack_factor;

                        (shard_size, shard_offset) =
                            adjust_marlin_shard(tensor_reg, shard_size, shard_offset);
                    }
                }
                let param_data = param.narrow(output_dim, shard_offset, shard_size)?;
                if tp_size == 1 {
                    (param_data, loaded_weight)
                } else {
                    let start_idx = tp_rank * shard_size;
                    let narrow_loaded_weight =
                        loaded_weight.narrow(output_dim, start_idx, shard_size)?;
                    drop(loaded_weight);
                    (param_data, narrow_loaded_weight)
                }
            } else {
                let ignore_warning =
                    if let Some(ignore_warning) = tensor_reg.getattr("ignore_warning") {
                        ignore_warning == 1
                    } else {
                        false
                    };
                if !ignore_warning {
                    tracing::warn!("Loading a weight without `output_dim` attribute in MergedColumnParallelLinear, assume the weight is the same for all partitions.");
                }
                let param_data = param.i(..)?;
                (param_data, loaded_weight)
            };
            cuda_copy(&param_data, &loaded_weight)?;
            drop(loaded_weight);
        } else {
            if let Some(output_dim) = tensor_reg.getattr("output_dim") {
                let mut current_shard_offset: usize = 0;
                let mut shard_offsets = Vec::new();
                for (i, output_size) in self.output_sizes.iter().enumerate() {
                    shard_offsets.push((i, current_shard_offset, *output_size));
                    current_shard_offset += *output_size;
                }
                let packed_dim = tensor_reg.getattr("packed_dim");
                for (shard_id, mut shard_offset, mut shard_size) in shard_offsets {
                    if packed_dim.is_some() && packed_dim.unwrap() == output_dim {
                        let pack_factor = tensor_reg.getattr("pack_factor").unwrap();
                        shard_size = shard_size / pack_factor;
                        shard_offset = shard_offset / pack_factor;

                        (shard_size, shard_offset) =
                            adjust_marlin_shard(tensor_reg, shard_size, shard_offset);
                    }
                    let loaded_weight_shard =
                        loaded_weight.narrow(output_dim, shard_offset, shard_size)?;
                    self.weight_loader(tensor_reg, param, loaded_weight_shard, Some(shard_id))?;
                }
            } else {
                cuda_copy(&param, &loaded_weight)?;
                drop(loaded_weight);
            }
        }
        common::cuda_synchronize(param.device());
        Ok(())
    }
    pub fn load(
        mut self,
        vb: &VarBuilder,
        prefixes: &[&str],
        config: W::Config,
        with_shard: bool,
    ) -> candle::Result<Self> {
        if prefixes.len() != self.output_sizes.len() {
            candle::bail!("expected same len with out_dims & prefixes");
        }
        let regs = self.linear.get_weight_desc(&config, vb.dtype());
        let mut loaded_weights = HashMap::new();
        for reg in &regs {
            let merged_weight = Tensor::zeros(reg.shard_shape.clone(), reg.dtype, vb.device())?;
            loaded_weights.insert(reg.name, merged_weight);
        }
        let mut merge_regs = Vec::new();
        for (i, _prefix) in prefixes.iter().enumerate() {
            let regs = W::get_descs(
                self.linear.input_size,
                self.output_sizes[i],
                self.linear.input_size,
                self.output_sizes[i],
                vb.dtype(),
                &config,
            );
            let mut reg_map = HashMap::new();
            for reg in regs {
                reg_map.insert(reg.name, reg);
            }
            merge_regs.push(reg_map);
        }

        let mut shard_id: usize = 0;
        for (i, (_output_size, prefix)) in self.output_sizes.iter().zip(prefixes.iter()).enumerate()
        {
            for reg in &regs {
                let param = loaded_weights.get(reg.name).unwrap();
                let act_reg = merge_regs[i].get(reg.name).unwrap();
                let loaded_weight = vb.pp(*prefix).get_with_hints_dtype(
                    act_reg.full_shape.clone(),
                    act_reg.name,
                    Default::default(),
                    act_reg.dtype,
                )?;
                let loaded_shard_id = if with_shard { Some(shard_id) } else { None };
                self.weight_loader(reg, param, loaded_weight, loaded_shard_id)?;
            }
            shard_id += 1;
        }
        self.linear.weights = Some(W::from(loaded_weights, config)?);
        Ok(self)
    }

    pub fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        let mut default_creator = DefaultTensorCreator {};
        self.forward_(x, &mut default_creator)
    }
    pub fn forward_<F: TensorCreator>(
        &self,
        x: &Tensor,
        tensor_creator: &mut F,
    ) -> candle::Result<Tensor> {
        self.linear.forward_(x, tensor_creator)
    }
}

pub struct QKVParallelLinear<W: LinearWeights> {
    linear: ColumnParallelLinear<W>,
    num_heads: usize,
    hidden_size: usize,
    head_size: usize,
    total_num_heads: usize,
    total_num_kv_heads: usize,
    num_kv_heads: usize,
    num_kv_head_replicas: usize,
    q_size: usize,
    kv_size: usize,
}

impl<W: LinearWeights> QKVParallelLinear<W> {
    pub fn new(
        hidden_size: usize,
        head_size: usize,
        total_num_heads: usize,
        total_num_kv_heads: Option<usize>,
        parallel_state: &ParallelState,
    ) -> Self {
        let total_num_kv_heads = if let Some(v) = total_num_kv_heads {
            v
        } else {
            total_num_heads
        };
        let tp_size = parallel_state.get_tensor_model_parallel_world_size();
        let num_heads = total_num_heads / tp_size;
        let (num_kv_heads, num_kv_head_replicas) = if tp_size >= total_num_kv_heads {
            (1_usize, tp_size / total_num_kv_heads)
        } else {
            (total_num_kv_heads / tp_size, 1_usize)
        };

        let input_size = hidden_size;
        let output_size = (num_heads + 2 * num_kv_heads) * tp_size * head_size;

        let head_dim = hidden_size / total_num_heads;
        let q_size = num_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        let linear = ColumnParallelLinear::<W>::new(input_size, output_size, parallel_state);
        Self {
            linear,
            num_heads,
            hidden_size,
            head_size,
            total_num_heads,
            total_num_kv_heads,
            num_kv_heads,
            num_kv_head_replicas,
            q_size,
            kv_size,
        }
    }
    pub fn with_bias(mut self, v: bool) -> Self {
        self.linear.bias = v;
        self
    }

    pub fn with_skip_bias_add(mut self, v: bool) -> Self {
        self.linear.skip_bias_add = v;
        self
    }

    fn weight_loader(
        &self,
        tensor_reg: &WeightRegistry,
        param: &Tensor,
        loaded_weight: Tensor,
        loaded_shard_id: Option<usize>,
    ) -> candle::Result<()> {
        let tp_rank = self.linear.tp_rank;
        if let Some(loaded_shard_id) = loaded_shard_id {
            let shard_offset_sizes = [
                (0, self.num_heads * self.head_size),
                (
                    self.num_heads * self.head_size,
                    self.num_kv_heads * self.head_size,
                ),
                (
                    (self.num_heads + self.num_kv_heads) * self.head_size,
                    self.num_kv_heads * self.head_size,
                ),
            ];
            let (param_data, loaded_weight) = if let Some(output_dim) =
                tensor_reg.getattr("output_dim")
            {
                let (mut shard_offset, mut shard_size) = shard_offset_sizes[loaded_shard_id];
                let packed_dim = tensor_reg.getattr("packed_dim");
                if packed_dim.is_some() && packed_dim.unwrap() == output_dim {
                    let pack_factor = tensor_reg.getattr("pack_factor").unwrap();
                    shard_size = shard_size / pack_factor;
                    shard_offset = shard_offset / pack_factor;

                    (shard_size, shard_offset) =
                        adjust_marlin_shard(tensor_reg, shard_size, shard_offset);
                }
                let shard_id = if loaded_shard_id == 0 {
                    tp_rank
                } else {
                    tp_rank / self.num_kv_head_replicas
                };

                let param_data = param.narrow(output_dim, shard_offset, shard_size)?;
                let start_idx = shard_id * shard_size;
                let loaded_shard_weight =
                    loaded_weight.narrow(output_dim, start_idx, shard_size)?;
                drop(loaded_weight);
                (param_data, loaded_shard_weight)
            } else {
                let ignore_warning =
                    if let Some(ignore_warning) = tensor_reg.getattr("ignore_warning") {
                        ignore_warning == 1
                    } else {
                        false
                    };
                if !ignore_warning {
                    tracing::warn!("Loading a weight without `output_dim` attribute in QKVParallelLinear, assume the weight is the same for all partitions.");
                }
                let param_data = param.i(..)?;
                (param_data, loaded_weight)
            };
            cuda_copy(&param_data, &loaded_weight)?;
            drop(loaded_weight);
        } else {
            if let Some(output_dim) = tensor_reg.getattr("output_dim") {
                let shard_offsets = [
                    (0_usize, self.total_num_heads * self.head_size),
                    (
                        self.total_num_heads * self.head_size,
                        self.total_num_kv_heads * self.head_size,
                    ),
                    (
                        (self.total_num_heads + self.total_num_kv_heads) * self.head_size,
                        self.total_num_kv_heads * self.head_size,
                    ),
                ];
                let packed_dim = tensor_reg.getattr("packed_dim");
                for (idx, (mut shard_offset, mut shard_size)) in shard_offsets.iter().enumerate() {
                    if packed_dim.is_some() && packed_dim.unwrap() == output_dim {
                        let pack_factor = tensor_reg.getattr("pack_factor").unwrap();
                        shard_size = shard_size / pack_factor;
                        shard_offset = shard_offset / pack_factor;

                        (shard_size, shard_offset) =
                            adjust_marlin_shard(tensor_reg, shard_size, shard_offset);
                    }
                    let loaded_weight_shard =
                        loaded_weight.narrow(output_dim, shard_offset, shard_size)?;
                    self.weight_loader(tensor_reg, param, loaded_weight_shard, Some(idx))?
                }
            } else {
                cuda_copy(&param, &loaded_weight)?;
                drop(loaded_weight);
            }
        }
        common::cuda_synchronize(param.device());
        Ok(())
    }

    pub fn load(
        mut self,
        vb: &VarBuilder,
        prefixes: &[&str],
        config: W::Config,
        with_shard: bool,
    ) -> candle::Result<Self> {
        // if prefixes.len() != 3 {
        //     candle::bail!("expected 3 prefixes for qkv weight load");
        // }
        let regs = self.linear.get_weight_desc(&config, vb.dtype());
        let mut loaded_weights = HashMap::new();
        for reg in &regs {
            let merged_weight = Tensor::zeros(reg.shard_shape.clone(), reg.dtype, vb.device())?;
            loaded_weights.insert(reg.name, merged_weight);
        }
        let qkv_sizes = [self.q_size, self.kv_size, self.kv_size];
        let mut qkv_regs = Vec::new();
        for (i, _prefix) in prefixes.iter().enumerate() {
            let regs = W::get_descs(
                self.hidden_size,
                qkv_sizes[i],
                self.hidden_size,
                qkv_sizes[i],
                vb.dtype(),
                &config,
            );
            let mut reg_map = HashMap::new();
            for reg in regs {
                reg_map.insert(reg.name, reg);
            }
            qkv_regs.push(reg_map);
        }

        for (i, prefix) in prefixes.iter().enumerate() {
            for reg in &regs {
                let param = loaded_weights.get(reg.name).unwrap();
                let act_reg = qkv_regs[i].get(reg.name).unwrap();
                let loaded_weight = vb.pp(*prefix).get_with_hints_dtype(
                    act_reg.full_shape.clone(),
                    act_reg.name,
                    Default::default(),
                    act_reg.dtype,
                )?;
                let loaded_shard_id = if with_shard { Some(i) } else { None };
                self.weight_loader(reg, param, loaded_weight, loaded_shard_id)?;
            }
        }
        self.linear.weights = Some(W::from(loaded_weights, config)?);
        Ok(self)
    }

    pub fn forward(&self, x: &Tensor) -> candle::Result<(Tensor, Tensor, Tensor)> {
        let mut default_creator = DefaultTensorCreator {};
        self.forward_(x, &mut default_creator)
    }

    pub fn forward_<F: TensorCreator>(
        &self,
        x: &Tensor,
        tensor_creator: &mut F,
    ) -> candle::Result<(Tensor, Tensor, Tensor)> {
        let result = self.linear.forward_(x, tensor_creator)?;
        let q = result.i((.., .., 0..self.q_size))?;
        let k = result.i((.., .., self.q_size..(self.q_size + self.kv_size)))?;
        let v = result.i((.., .., (self.q_size + self.kv_size)..))?;
        Ok((q, k, v))
    }
}

pub struct RowParallelLinear<W: LinearWeights> {
    weights: Option<W>,
    tp_size: usize,
    tp_rank: usize,
    input_size: usize,
    output_size: usize,
    input_size_per_partition: usize,
    bias: bool,
    input_is_parallel: bool,
    skip_bias_add: bool,
    reduce_results: bool,
}

impl<W: LinearWeights> RowParallelLinear<W> {
    pub fn new(input_size: usize, output_size: usize, parallel_state: &ParallelState) -> Self {
        let tp = parallel_state.get_tensor_model_parallel_world_size();
        Self {
            weights: None,
            tp_size: parallel_state.get_tensor_model_parallel_world_size(),
            tp_rank: parallel_state.get_tensor_model_parallel_rank(),
            input_size_per_partition: input_size / tp,
            input_size,
            output_size,
            bias: false,
            input_is_parallel: false,
            skip_bias_add: false,
            reduce_results: false,
        }
    }
    pub fn with_bias(mut self, v: bool) -> Self {
        self.bias = v;
        self
    }
    pub fn with_input_is_parallel(mut self, v: bool) -> Self {
        self.input_is_parallel = v;
        self
    }
    pub fn with_reduce_results(mut self, v: bool) -> Self {
        self.reduce_results = v;
        self
    }
    pub fn with_skip_bias_add(mut self, v: bool) -> Self {
        self.skip_bias_add = v;
        self
    }

    fn weight_loader(
        &self,
        tensor_reg: &WeightRegistry,
        loaded_weight: Tensor,
    ) -> candle::Result<Tensor> {
        let loaded_weight = if let Some(input_dim) = tensor_reg.getattr("input_dim") {
            if self.tp_size == 1 {
                loaded_weight
            } else {
                let shard_size = tensor_reg.shard_shape.dims()[input_dim];
                let start_idx = self.tp_rank * shard_size;
                let t = loaded_weight.narrow(input_dim, start_idx, shard_size)?;
                let copy_t = Tensor::zeros_like(&t)?;
                cuda_copy(&copy_t, &t)?;
                drop(t);
                drop(loaded_weight);
                common::cuda_synchronize(copy_t.device());
                copy_t
            }
        } else {
            loaded_weight
        };
        Ok(loaded_weight)
    }

    pub fn load(mut self, vb: &VarBuilder, config: W::Config) -> candle::Result<Self> {
        let mut regs = W::get_descs(
            self.input_size_per_partition,
            self.output_size,
            self.input_size,
            self.output_size,
            vb.dtype(),
            &config,
        );
        if self.bias {
            let bias_shape = Shape::from_dims(&[self.output_size]);
            let bias_attrs = HashMap::from([("output_dim", 0_usize)]);
            let bias_reg = WeightRegistry::new(
                "bias",
                bias_shape.clone(),
                bias_shape,
                vb.dtype(),
                bias_attrs,
            );
            regs.push(bias_reg);
        }
        let mut loaded_weights = HashMap::new();
        for reg in &regs {
            let loaded_weight = vb.get_with_hints_dtype(
                reg.full_shape.clone(),
                reg.name,
                Default::default(),
                reg.dtype,
            )?;
            let loaded_weight = self.weight_loader(reg, loaded_weight)?;
            loaded_weights.insert(reg.name, loaded_weight);
        }
        self.weights = Some(W::from(loaded_weights, config)?);
        Ok(self)
    }

    pub fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        let mut default_creator = DefaultTensorCreator {};
        self.forward_(x, &mut default_creator)
    }
    pub fn forward_<F: TensorCreator>(
        &self,
        x: &Tensor,
        tensor_creator: &mut F,
    ) -> candle::Result<Tensor> {
        match &self.weights {
            Some(weights) => {
                let input_parallel = if self.input_is_parallel {
                    x
                } else {
                    //     tp_rank = get_tensor_model_parallel_rank()
                    //     splitted_input = split_tensor_along_last_dim(
                    //         input_, num_partitions=self.tp_size)
                    //     input_parallel = splitted_input[tp_rank].contiguous()
                    x
                };
                let output_parallel = weights.apply(tensor_creator, input_parallel)?;
                let output_parallel = if self.reduce_results && self.tp_size > 1 {
                    //output_ = tensor_model_parallel_all_reduce(output_parallel)
                    todo!("parallel_all_reduce")
                } else {
                    output_parallel
                };
                Ok(output_parallel)
            }
            None => {
                candle::bail!("RowParallelLinear is not inited!")
            }
        }
    }
}
