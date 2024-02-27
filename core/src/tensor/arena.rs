use std::sync::atomic::AtomicUsize;

use candle_core::{DType, Device, IndexOp, Shape, Tensor};
use common::TensorCreator;
use metrics::atomics::AtomicU64;

const DEFAULT_MIN_BLOCK_SIZE: usize = 4096;
const ALLIGNMENT: usize = 64;
struct TensorArenaUnit {
    cache: Tensor,
    curosr: AtomicUsize,
    capacity: usize,
}

impl TensorArenaUnit {
    pub fn new(size: usize, dtype: DType, device: &Device) -> candle_core::Result<Self> {
        let cache = Tensor::zeros(size, dtype, device)?;
        Ok(Self {
            cache,
            curosr: AtomicUsize::new(0),
            capacity: size,
        })
    }

    pub fn reset(&self) {
        self.curosr.store(0, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn get(&self, dtype: DType, shape: &Shape, zero: bool) -> candle_core::Result<Tensor> {
        // let shape = shape.into();
        if dtype != self.cache.dtype() {
            return Err(candle_core::Error::DTypeMismatchBinaryOp {
                lhs: dtype,
                rhs: self.cache.dtype(),
                op: "arena_get",
            });
        }
        let n = shape.elem_count();
        let allign_n = (n + ALLIGNMENT - 1) / ALLIGNMENT * ALLIGNMENT;
        let curosr = self.curosr.load(std::sync::atomic::Ordering::SeqCst);
        if curosr + allign_n > self.capacity {
            return candle_core::bail!(
                "arena buffer overflow with requestd:{}, while rest:{}",
                allign_n,
                self.capacity - curosr
            );
        }
        // println!("cursor:{}, n:{}", curosr, n);
        let t = self.cache.i(curosr..curosr + n)?.reshape(shape.clone())?;
        self.curosr
            .fetch_add(allign_n, std::sync::atomic::Ordering::SeqCst);
        if zero {
            tops::unsafe_tensor_zero(&t)?;
        }
        Ok(t)
    }
}

struct TensorArenaUnitGroup {
    group: Vec<TensorArenaUnit>,
    min_block_size: usize,
    cursor: AtomicUsize,
}
impl TensorArenaUnitGroup {
    pub fn new(min_block_size: usize) -> Self {
        Self {
            group: Vec::new(),
            min_block_size,
            cursor: AtomicUsize::new(0),
        }
    }
    pub fn reset(&self) {
        for arena in &self.group {
            arena.reset();
        }
        self.cursor.store(0, std::sync::atomic::Ordering::SeqCst);
    }
    pub fn len(&self) -> usize {
        self.group.len()
    }
    pub fn cursor(&self) -> usize {
        match self.group.last() {
            Some(c) => c.curosr.load(std::sync::atomic::Ordering::SeqCst) as usize,
            None => 0,
        }
    }

    fn do_get(&mut self, dtype: DType, shape: &Shape, zero: bool) -> candle_core::Result<Tensor> {
        let curosr = self.cursor.load(std::sync::atomic::Ordering::SeqCst);

        if curosr < self.group.len() {
            match self.group[curosr].get(dtype, shape, zero) {
                Ok(s) => {
                    return Ok(s);
                }
                Err(_) => {}
            }
        }
        Err(candle_core::Error::Msg("no space".to_owned()))
        //candle_core::bail!("no space")
    }
    pub fn get<S: Into<Shape>>(
        &mut self,
        dtype: DType,
        shape: S,
        zero: bool,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let shape = shape.into();
        match self.do_get(dtype, &shape, zero) {
            Ok(s) => {
                return Ok(s);
            }
            Err(_) => {}
        }
        //create new block

        let n = shape.elem_count();

        let allign_n = (n + ALLIGNMENT - 1) / ALLIGNMENT * ALLIGNMENT;
        let new_block_len = if allign_n > self.min_block_size {
            allign_n
        } else {
            self.min_block_size
        };
        //println!("create with n:{}/{}/{}", new_block_len, allign_n, n);
        let segment = TensorArenaUnit::new(new_block_len, dtype, device)?;
        self.group.push(segment);
        if self.group.len() > 1 {
            self.cursor
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }

        self.do_get(dtype, &shape, zero)
    }
}

pub struct TensorArena {
    cache: Vec<TensorArenaUnitGroup>,
    device: Device,
}

impl TensorArena {
    pub fn new(device: &Device) -> Self {
        TensorArena::with_min_block_size(DEFAULT_MIN_BLOCK_SIZE, device)
    }
    pub fn with_min_block_size(min_block_size: usize, device: &Device) -> Self {
        Self {
            cache: vec![
                TensorArenaUnitGroup::new(min_block_size),
                TensorArenaUnitGroup::new(min_block_size),
                TensorArenaUnitGroup::new(min_block_size),
                TensorArenaUnitGroup::new(min_block_size),
                TensorArenaUnitGroup::new(min_block_size),
                TensorArenaUnitGroup::new(min_block_size),
                TensorArenaUnitGroup::new(min_block_size),
            ],
            device: device.clone(),
        }
    }
    pub fn stat(&self, dtype: DType) -> (usize, usize) {
        let idx = dtype as usize;
        let cache = &self.cache[idx];
        (cache.len(), cache.cursor())
    }

    pub fn reset(&self) {
        for c in &self.cache {
            c.reset();
        }
    }

    pub fn get<S: Into<Shape>>(
        &mut self,
        dtype: DType,
        shape: S,
        zero: bool,
    ) -> candle_core::Result<Tensor> {
        let idx = dtype as usize;
        let cache = &self.cache[idx];
        self.cache[idx].get(dtype, shape, zero, &self.device)
    }
}

impl TensorCreator for TensorArena {
    fn new<S: Into<Shape>>(
        &mut self,
        shape: S,
        dtype: DType,
        device: &Device,
        zero: bool,
    ) -> candle_core::Result<Tensor> {
        if !self.device.same_device(device) {
            return Err(candle_core::Error::DeviceMismatchBinaryOp {
                lhs: self.device.location(),
                rhs: device.location(),
                op: "new_tensor",
            });
        }
        self.get(dtype, shape, zero)
    }
}

#[test]
fn test_arena() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let mut arena = TensorArena::new(&device);

    let t0 = arena.get(DType::F16, (1, 5), false)?;
    let t1 = arena.get(DType::F16, (2, 5), false)?;
    println!("fp16 arena stat: {:?}", arena.stat(DType::F16));
    arena.reset();
    println!("after reset fp16 arena stat: {:?}", arena.stat(DType::F16));
    let t0 = arena.get(DType::F16, (1, 5), false)?;
    let t1 = arena.get(DType::F16, (2, 5), false)?;
    let t1 = arena.get(DType::F16, (1, 32000), false)?;
    println!(" fp16 arena stat: {:?}", arena.stat(DType::F16));
    Ok(())
}
