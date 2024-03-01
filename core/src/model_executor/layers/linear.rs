use std::sync::Arc;

use candle_core::IndexOp;
use candle_core::{CudaDevice, DType, Device, Tensor, D};
use candle_nn::Module;
use candle_nn::VarBuilder;
use common::{DefaultTensorCreator, TensorCreator};
use std::rc::Rc;

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

struct TensorParallelColumnLinear {
    linear: Linear,
}

impl TensorParallelColumnLinear {
    fn new(linear: Linear) -> Self {
        Self { linear }
    }
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.linear.forward(x)
    }
    // fn load(vb: VarBuilder, comm: Rc<Comm>) -> Result<Self> {
    //     let rank = comm.rank();
    //     let size = comm.world_size();
    //     let weight = vb.get_with_hints((), "weight", shard(0, rank, size))?;
    //     Ok(Self::new(candle_nn::Linear::new(weight, None)))
    // }

    // fn load_multi(vb: VarBuilder, prefixes: &[&str], comm: Rc<Comm>) -> candle_core::Result<Self> {
    //     let rank = comm.rank();
    //     let size = comm.world_size();
    //     let weights: Vec<_> = prefixes
    //         .iter()
    //         .map(|p| vb.pp(p).get_with_hints((), "weight", shard(0, rank, size)))
    //         .collect::<candle_core::Result<Vec<_>>>()?;
    //     let weight = Tensor::cat(&weights, 0)?;
    //     Ok(Self::new(candle_nn::Linear::new(weight, None)))
    // }
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
        let result = self.linear.forward(x)?;
        // if debug_file.is_some() {
        //     result.save_safetensors("qkv", debug_file.unwrap());
        // }
        let q = result.i((.., .., 0..self.q_size))?;
        let k = result.i((.., .., self.q_size..(self.q_size + self.k_size)))?;
        let v = result.i((.., .., (self.q_size + self.k_size)..))?;
        Ok((q, k, v))
    }

    pub fn forward_<F: TensorCreator>(
        &self,
        x: &Tensor,
        tensor_creator: &mut F,
        // debug_file: Option<&str>,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let result = self.linear.forward_(x, tensor_creator)?;
        // if debug_file.is_some() {
        //     result.save_safetensors("qkv", debug_file.unwrap());
        // }
        let q = result.i((.., .., 0..self.q_size))?;
        let k = result.i((.., .., self.q_size..(self.q_size + self.k_size)))?;
        let v = result.i((.., .., (self.q_size + self.k_size)..))?;
        Ok((q, k, v))
    }
}
