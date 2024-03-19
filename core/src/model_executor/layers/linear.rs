use std::sync::Arc;

use candle_core::{CudaDevice, DType, Device, Tensor, D};
use candle_core::{IndexOp, Shape};
use candle_nn::Module;
use candle_nn::VarBuilder;
use common::{DefaultTensorCreator, TensorCreator};
use std::collections::HashMap;
use std::rc::Rc;

use crate::model_executor::parallel::ParallelState;

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
    type Config;
    fn from(weights: HashMap<&'static str, Tensor>) -> candle_core::Result<Self>;
    fn get_descs(input_size: usize, output_size: usize, params_dtype: DType)
        -> Vec<WeightRegistry>;

    fn apply<C: TensorCreator>(
        &self,
        tensor_creator: &mut C,
        x: &Tensor,
    ) -> candle_core::Result<Tensor>;
}

pub struct UnquantizedLinearWeights {
    cublas: tops::CublasWrapper,
    pub(crate) weight: Tensor,
    bias: Option<Tensor>,
}

impl LinearWeights for UnquantizedLinearWeights {
    type Config = Option<String>;
    fn from(mut weights: HashMap<&'static str, Tensor>) -> candle_core::Result<Self> {
        if let Some(weight) = weights.remove("weight") {
            let bias = weights.remove("bias");
            let stream = match weight.device() {
                Device::Cuda(cuda_dev) => *cuda_dev.cu_stream(),
                _ => {
                    candle_core::bail!("no cuda dev")
                }
            };
            let cublas = tops::CublasWrapper::new(weight.device(), weight.dtype(), stream)?;
            Ok(Self {
                cublas,
                weight,
                bias,
            })
        } else {
            candle_core::bail!("missing 'weight' in weights")
        }
    }
    fn get_descs(
        input_size: usize,
        output_size: usize,
        params_dtype: DType,
    ) -> Vec<WeightRegistry> {
        let shape = Shape::from_dims(&[output_size, input_size]);
        let attrs = HashMap::from([("input_dim", 1_usize), ("output_dim", 0)]);
        vec![WeightRegistry::new("weight", shape, params_dtype, attrs)]
    }
    fn apply<C: TensorCreator>(
        &self,
        tensor_creator: &mut C,
        x: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let x1 = self.cublas.linear_(x, &self.weight, tensor_creator)?;
        match self.bias.as_ref() {
            None => Ok(x1),
            Some(bias) => x1.broadcast_add(bias),
        }
    }
}

pub struct ColumnParallelLinear<W: LinearWeights> {
    weights: W,
}

impl<W: LinearWeights> Layer for ColumnParallelLinear<W> {
    fn forward_<C: TensorCreator>(
        &self,
        x: &Tensor,
        tensor_creator: &mut C,
    ) -> candle_core::Result<candle_core::Tensor> {
        self.weights.apply(tensor_creator, x)
    }
}

impl<W: LinearWeights> ColumnParallelLinear<W> {
    fn merged_weight_loader(
        device: &Device,
        tensor_reg: &WeightRegistry,
        param: &Tensor,
        loaded_weight: Tensor,
        output_sizes: &[usize],
        shard_id: Option<usize>,
        parallel_state: &ParallelState,
    ) -> candle_core::Result<()> {
        let cuda_dev = match device {
            Device::Cuda(c) => c,
            _ => {
                candle_core::bail!("unexpected!")
            }
        };
        if let Some(shard_id) = shard_id {
            let tp_rank = parallel_state.get_tensor_model_parallel_rank();
            let tp_size = parallel_state.get_tensor_model_parallel_world_size();
            let (param_data, loaded_weight) = if let Some(output_dim) =
                tensor_reg.getattr("output_dim")
            {
                let mut shard_offset: usize = output_sizes[..shard_id].iter().sum();
                let mut shard_size = output_sizes[shard_id];
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
            tops::unsafe_tensor_dtod_copy(&param_data, &loaded_weight)?;
            drop(loaded_weight);
        } else {
            if let Some(output_dim) = tensor_reg.getattr("output_dim") {
                let mut current_shard_offset: usize = 0;
                let mut shard_offsets = Vec::new();
                for (i, output_size) in output_sizes.iter().enumerate() {
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
                    return Self::merged_weight_loader(
                        device,
                        tensor_reg,
                        param,
                        loaded_weight_shard,
                        output_sizes,
                        Some(shard_id),
                        parallel_state,
                    );
                }
            } else {
                tops::unsafe_tensor_dtod_copy(&param, &loaded_weight)?;
                drop(loaded_weight);
            }
        }
        cuda_dev.synchronize();
        Ok(())
    }
    pub fn merge_load(
        vb: &VarBuilder,
        in_size: usize,
        out_sizes: &[usize],
        prefixes: &[&str],
        parallel_state: &ParallelState,
    ) -> candle_core::Result<Self> {
        let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
        if prefixes.len() != out_sizes.len() {
            return candle_core::bail!("expected same len with out_dims & prefixes");
        }
        let sum_out_sizes: usize = out_sizes.iter().sum();

        let mut loaded_weights = HashMap::new();
        let regs = W::get_descs(in_size, sum_out_sizes, vb.dtype());
        for reg in &regs {
            let merged_weight = Tensor::zeros(reg.shape.clone(), reg.dtype, vb.device())?;
            //println!("create {:?}", reg.shape.clone());
            loaded_weights.insert(reg.name, merged_weight);
        }

        let mut shard_id: usize = 0;
        for (output_size, prefix) in out_sizes.iter().zip(prefixes.iter()) {
            for reg in &regs {
                let param = loaded_weights.get(reg.name).unwrap();
                let loaded_weight =
                    vb.pp(*prefix)
                        .get_with_hints((*output_size, in_size), reg.name, init_ws)?;
                Self::merged_weight_loader(
                    vb.device(),
                    reg,
                    param,
                    loaded_weight,
                    out_sizes,
                    Some(shard_id),
                    parallel_state,
                )?;
            }
            shard_id += 1;
        }
        let weights = W::from(loaded_weights)?;
        Ok(Self { weights })
    }

    fn weight_loader(
        device: &Device,
        tensor_reg: &WeightRegistry,
        loaded_weight: Tensor,
        parallel_state: &ParallelState,
    ) -> candle_core::Result<Tensor> {
        let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
        let tp_rank = parallel_state.get_tensor_model_parallel_rank();
        if let Some(output_dim) = tensor_reg.getattr("output_dim") {
            if parallel_state.get_tensor_model_parallel_world_size() == 1 {
                Ok(loaded_weight)
            } else {
                let shard_size = tensor_reg.shape.dims()[output_dim]
                    / parallel_state.get_tensor_model_parallel_world_size();
                let start_idx = tp_rank * shard_size;

                let cuda_dev = match device {
                    Device::Cuda(c) => c,
                    _ => {
                        candle_core::bail!("unexpected!")
                    }
                };
                let narrow_loaded_weight =
                    loaded_weight.narrow(output_dim, start_idx, shard_size)?;
                drop(loaded_weight);
                let t = Tensor::zeros(
                    narrow_loaded_weight.shape().clone(),
                    narrow_loaded_weight.dtype(),
                    device,
                )?;
                tops::unsafe_tensor_dtod_copy(&t, &narrow_loaded_weight)?;
                drop(narrow_loaded_weight);
                cuda_dev.synchronize();
                Ok(t)
            }
        } else {
            Ok(loaded_weight)
        }
    }
    pub fn load(
        vb: VarBuilder,
        in_dim: usize,
        out_dim: usize,
        parallel_state: &ParallelState,
    ) -> candle_core::Result<Self> {
        let regs = W::get_descs(in_dim, out_dim, vb.dtype());
        let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
        let mut loaded_weights = HashMap::new();
        for reg in &regs {
            let loaded_weight = vb.get_with_hints(reg.shape.clone(), reg.name, init_ws)?;
            let loaded_weight =
                Self::weight_loader(vb.device(), reg, loaded_weight, parallel_state)?;
            loaded_weights.insert(reg.name, loaded_weight);
        }
        let weights = W::from(loaded_weights)?;
        Ok(Self { weights })
    }
}

struct QKVParallelLinearMeta {
    total_num_heads: usize,
    head_size: usize,
    total_num_kv_heads: usize,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_head_replicas: usize,
    shard_offsets: [(usize, usize); 3],
}
pub struct QKVParallelLinear<W: LinearWeights> {
    weights: W,
    q_size: usize,
    k_size: usize,
    v_size: usize,
}

impl<W: LinearWeights> QKVParallelLinear<W> {
    fn qkv_weight_loader(
        device: &Device,
        tensor_reg: &WeightRegistry,
        meta: &QKVParallelLinearMeta,
        param: &Tensor,
        loaded_weight: Tensor,
        loaded_shard_id: Option<usize>,
        parallel_state: &ParallelState,
    ) -> candle_core::Result<()> {
        if let Some(loaded_shard_id) = loaded_shard_id {
            let tp_rank = parallel_state.get_tensor_model_parallel_rank();
            let (param_data, loaded_weight) = if let Some(output_dim) =
                tensor_reg.getattr("output_dim")
            {
                let (mut shard_offset, mut shard_size) = meta.shard_offsets[loaded_shard_id];

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
                    tp_rank / meta.num_kv_head_replicas
                };
                let param_data = param.narrow(output_dim, shard_offset, shard_size)?;
                let start_idx = shard_id * shard_size;
                //let loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)?;
                (param_data, loaded_weight)
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
            tops::unsafe_tensor_dtod_copy(&param_data, &loaded_weight)?;
            drop(loaded_weight);
        } else {
            if let Some(output_dim) = tensor_reg.getattr("output_dim") {
                let shard_offsets = [
                    (0_usize, meta.total_num_heads * meta.head_size),
                    (
                        meta.total_num_heads * meta.head_size,
                        meta.total_num_kv_heads * meta.head_size,
                    ),
                    (
                        (meta.total_num_heads + meta.total_num_kv_heads) * meta.head_size,
                        meta.total_num_kv_heads * meta.head_size,
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
                    drop(loaded_weight);
                    return Self::qkv_weight_loader(
                        device,
                        tensor_reg,
                        meta,
                        param,
                        loaded_weight_shard,
                        Some(idx),
                        parallel_state,
                    );
                }
            } else {
                tops::unsafe_tensor_dtod_copy(&param, &loaded_weight)?;
                drop(loaded_weight);
            }
        }
        let cuda_dev = match device {
            Device::Cuda(c) => c,
            _ => {
                candle_core::bail!("unexpected!")
            }
        };
        cuda_dev.synchronize();
        Ok(())
    }
    pub fn load(
        vb: &VarBuilder,
        hidden_size: usize,
        head_size: usize,
        total_num_heads: usize,
        total_num_kv_heads: Option<usize>,
        prefixes: &[&str],
        parallel_state: &ParallelState,
    ) -> candle_core::Result<Self> {
        let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
        if prefixes.len() != 3 {
            return candle_core::bail!("expected 3 prefixes for qkv");
        }
        let total_num_kv_heads = if let Some(total_num_kv_heads) = total_num_kv_heads {
            total_num_kv_heads
        } else {
            total_num_heads
        };
        let tp_size = parallel_state.get_tensor_model_parallel_world_size();
        let num_heads = total_num_heads / tp_size;
        let (num_kv_heads, num_kv_head_replicas) = if tp_size >= total_num_kv_heads {
            (1, tp_size / total_num_kv_heads)
        } else {
            let num_kv_heads = total_num_kv_heads / tp_size;
            let num_kv_head_replicas = 1;
            (num_kv_heads, num_kv_head_replicas)
        };

        let shard_offsets = [
            (0_usize, num_heads * head_size),
            (num_heads * head_size, total_num_kv_heads * head_size),
            (
                (num_heads + total_num_kv_heads) * head_size,
                total_num_kv_heads * head_size,
            ),
        ];

        let meta = QKVParallelLinearMeta {
            total_num_heads,
            head_size,
            total_num_kv_heads,
            num_heads,
            num_kv_heads,
            num_kv_head_replicas,
            shard_offsets,
        };

        let qkv_sizes = [
            meta.total_num_heads * meta.head_size,
            meta.total_num_kv_heads * meta.head_size,
            meta.total_num_kv_heads * meta.head_size,
        ];
        let input_size = hidden_size;
        let output_size = (num_heads + 2 * num_kv_heads) * tp_size * head_size;
        let mut loaded_weights = HashMap::new();
        let regs = W::get_descs(input_size, output_size, vb.dtype());
        for reg in &regs {
            let merged_weight = Tensor::zeros(reg.shape.clone(), reg.dtype, vb.device())?;
            loaded_weights.insert(reg.name, merged_weight);
        }

        for (i, prefix) in prefixes.iter().enumerate() {
            for reg in &regs {
                let param = loaded_weights.get(reg.name).unwrap();
                let loaded_weight =
                    vb.pp(*prefix)
                        .get_with_hints((input_size, qkv_sizes[i]), reg.name, init_ws)?;
                Self::qkv_weight_loader(
                    vb.device(),
                    reg,
                    &meta,
                    param,
                    loaded_weight,
                    Some(i),
                    parallel_state,
                )?;
            }
        }
        let weights = W::from(loaded_weights)?;
        Ok(Self {
            weights,
            q_size: meta.total_num_heads * meta.head_size,
            k_size: meta.total_num_kv_heads * meta.head_size,
            v_size: meta.total_num_kv_heads * meta.head_size,
        })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let mut default_creator = DefaultTensorCreator {};
        self.forward_(x, &mut default_creator)
    }

    pub fn forward_<F: TensorCreator>(
        &self,
        x: &Tensor,
        tensor_creator: &mut F,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let result = self.weights.apply(tensor_creator, x)?;
        let q = result.i((.., .., 0..self.q_size))?;
        let k = result.i((.., .., self.q_size..(self.q_size + self.k_size)))?;
        let v = result.i((.., .., (self.q_size + self.k_size)..))?;
        Ok((q, k, v))
    }
}

pub struct Linear {
    // cublas: Arc<tops::CublasWrapper>,
    pub(crate) weight: Tensor,
    bias: Option<Tensor>,
    cublas: tops::CublasWrapper,
}
impl Linear {
    pub fn load_multi(
        vb: &VarBuilder,
        in_dim: usize,
        out_dim: usize,
        prefixes: &[&str],
    ) -> candle_core::Result<Self> {
        let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
        let first_tensor =
            vb.pp(prefixes[0])
                .get_with_hints((out_dim, in_dim), "weight", init_ws)?;
        let mut weight = Tensor::zeros((out_dim * 2, in_dim), vb.dtype(), vb.device())?;
        tops::unsafe_tensor_dtod_copy(&weight, &first_tensor)?;
        drop(first_tensor);
        for (i, prefix) in prefixes.into_iter().enumerate() {
            if i > 0 {
                let t = vb
                    .pp(prefix)
                    .get_with_hints((out_dim, in_dim), "weight", init_ws)?;
                let next_w = weight.i((i * out_dim.., ..))?;
                tops::unsafe_tensor_dtod_copy(&next_w, &t)?;
                drop(t);
            }
        }

        let cublas =
            tops::CublasWrapper::new(weight.device(), weight.dtype(), std::ptr::null_mut())?;
        Ok(Linear {
            weight,
            bias: None,
            cublas,
        })
    }

    pub fn forward_<F: TensorCreator>(
        &self,
        x: &Tensor,
        tensor_creator: &mut F,
    ) -> candle_core::Result<Tensor> {
        let x1 = self.cublas.linear_(x, &self.weight, tensor_creator)?;

        match &self.bias {
            None => Ok(x1),
            Some(bias) => {
                //println!("####with bias");
                x1.broadcast_add(bias)
            }
        }
    }
}

pub fn linear(
    // cublas: Arc<tops::CublasWrapper>,
    in_dim: usize,
    out_dim: usize,
    vs: candle_nn::VarBuilder,
) -> candle_core::Result<Linear> {
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let ws = vs.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    let bound = 1. / (in_dim as f64).sqrt();
    let init_bs = candle_nn::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bs = vs.get_with_hints(out_dim, "bias", init_bs)?;
    let cublas = tops::CublasWrapper::new(ws.device(), ws.dtype(), std::ptr::null_mut())?;
    Ok(Linear {
        // cublas,
        weight: ws,
        bias: Some(bs),
        cublas,
    })
}

/// Create or initialize a new linear layer without biases.
pub fn linear_no_bias(
    in_dim: usize,
    out_dim: usize,
    vs: candle_nn::VarBuilder,
) -> candle_core::Result<Linear> {
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let ws = vs.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    let cublas = tops::CublasWrapper::new(ws.device(), ws.dtype(), std::ptr::null_mut())?;
    // println!("###Load ws shape:{:?}", ws.shape());

    Ok(Linear {
        // cublas,
        weight: ws,
        bias: None,
        cublas,
    })
}

impl candle_core::Module for Linear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // let w = match *x.dims() {
        //     [b1, b2, _, _] => self.weight.broadcast_left((b1, b2))?.t()?,
        //     [bsize, _, _] => self.weight.broadcast_left(bsize)?.t()?,
        //     _ => self.weight.t()?,
        // };
        // let x1 = x.matmul(&w)?;
        let x1 = self.cublas.linear(x, &self.weight)?;
        // println!(
        //     "####x:{:?}, w:{:?}, x1:{:?}",
        //     x.shape(),
        //     self.weight.shape(),
        //     x1.shape()
        // );
        match &self.bias {
            None => Ok(x1),
            Some(bias) => x1.broadcast_add(bias),
        }
    }
}

pub struct QKVLinear {
    linear: Linear,
    q_size: usize,
    k_size: usize,
    v_size: usize,
}

impl QKVLinear {
    pub fn load_qkv(
        vb: &VarBuilder,
        in_dim: usize,
        q_out_dim: usize,
        q_name: &str,
        k_out_dim: usize,
        k_name: &str,
        v_out_dim: usize,
        v_name: &str,
    ) -> candle_core::Result<Self> {
        let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
        let cuda_dev = match vb.device() {
            Device::Cuda(c) => c,
            _ => {
                candle_core::bail!("unexpected!")
            }
        };
        let mut weight = Tensor::zeros(
            (q_out_dim + k_out_dim + v_out_dim, in_dim),
            vb.dtype(),
            vb.device(),
        )?;
        let q_tensor = vb
            .pp(q_name)
            .get_with_hints((q_out_dim, in_dim), "weight", init_ws)?;
        tops::unsafe_tensor_dtod_copy(&weight, &q_tensor)?;
        drop(q_tensor);

        let k_tensor = vb
            .pp(k_name)
            .get_with_hints((k_out_dim, in_dim), "weight", init_ws)?;
        let k_view = weight.i((q_out_dim.., ..))?;
        tops::unsafe_tensor_dtod_copy(&k_view, &k_tensor)?;
        drop(k_tensor);

        let v_tensor = vb
            .pp(v_name)
            .get_with_hints((v_out_dim, in_dim), "weight", init_ws)?;
        let v_view = weight.i((q_out_dim + k_out_dim.., ..))?;
        tops::unsafe_tensor_dtod_copy(&v_view, &v_tensor)?;
        drop(v_tensor);

        // cuda_dev.synchronize();

        let cublas =
            tops::CublasWrapper::new(weight.device(), weight.dtype(), std::ptr::null_mut())?;
        let linear = Linear {
            weight,
            bias: None,
            cublas,
        };
        Ok(QKVLinear {
            linear,
            q_size: q_out_dim,
            k_size: k_out_dim,
            v_size: v_out_dim,
        })
    }
}

impl QKVLinear {
    pub fn forward(
        &self,
        x: &Tensor,
        // debug_file: Option<&str>,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let mut default_creator = DefaultTensorCreator {};
        self.forward_(x, &mut default_creator)
    }

    pub fn forward_<F: TensorCreator>(
        &self,
        x: &Tensor,
        tensor_creator: &mut F,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let result = self.linear.forward_(x, tensor_creator)?;
        let q = result.i((.., .., 0..self.q_size))?;
        let k = result.i((.., .., self.q_size..(self.q_size + self.k_size)))?;
        let v = result.i((.., .., (self.q_size + self.k_size)..))?;
        Ok((q, k, v))
    }
}
