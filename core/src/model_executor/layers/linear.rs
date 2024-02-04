use std::sync::Arc;

use candle_core::{CudaDevice, DType, Device, Tensor, D};
use candle_nn::Module;
// use cudarc::nccl::safe::{Comm, ReduceOp};
use std::rc::Rc;

pub struct Linear {
    // cublas: Arc<tops::CublasWrapper>,
    weight: Tensor,
    bias: Option<Tensor>,
    cublas: tops::CublasWrapper,
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
    linear: candle_nn::Linear,
}

impl TensorParallelColumnLinear {
    fn new(linear: candle_nn::Linear) -> Self {
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
