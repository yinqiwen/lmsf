use std::sync::atomic::AtomicUsize;

use candle::{DType, Device, IndexOp, Shape, Tensor};
use common::TensorCreator;

const DEFAULT_MIN_BLOCK_SIZE: usize = 1024 * 1024;
const ALLIGNMENT: usize = 64;

#[derive(Debug)]
struct TensorArenaUnit {
    cache: Tensor,
    curosr: AtomicUsize,
    capacity: usize,
}

impl TensorArenaUnit {
    pub fn new(size: usize, dtype: DType, device: &Device) -> candle::Result<Self> {
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

    pub fn get(&self, dtype: DType, shape: &Shape, zero: bool) -> candle::Result<Tensor> {
        // let shape = shape.into();
        if dtype != self.cache.dtype() {
            return Err(candle::Error::DTypeMismatchBinaryOp {
                lhs: dtype,
                rhs: self.cache.dtype(),
                op: "arena_get",
            });
        }
        let n = shape.elem_count();
        let allign_n = (n + ALLIGNMENT - 1) / ALLIGNMENT * ALLIGNMENT;
        let curosr = self.curosr.load(std::sync::atomic::Ordering::SeqCst);
        if curosr + allign_n > self.capacity {
            candle::bail!(
                "arena buffer overflow with requestd:{}, while rest:{}",
                allign_n,
                self.capacity - curosr
            )
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

#[allow(dead_code)]
#[derive(Debug)]
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

    pub fn capacity(&self) -> usize {
        self.group.iter().map(|x| x.capacity).sum()
    }
    pub fn cursor(&self) -> usize {
        match self.group.last() {
            Some(c) => c.curosr.load(std::sync::atomic::Ordering::SeqCst) as usize,
            None => 0,
        }
    }

    fn do_get(
        &mut self,
        curosr: usize,
        dtype: DType,
        shape: &Shape,
        zero: bool,
    ) -> candle::Result<Tensor> {
        // println!("####{} {}", curosr, self.group.len());
        if curosr < self.group.len() {
            match self.group[curosr].get(dtype, shape, zero) {
                Ok(s) => {
                    return Ok(s);
                }
                Err(_) => {}
            }
        }
        Err(candle::Error::Msg("no space".to_owned()))
        //candle_core::bail!("no space")
    }
    pub fn get<S: Into<Shape>>(
        &mut self,
        dtype: DType,
        shape: S,
        zero: bool,
        device: &Device,
    ) -> candle::Result<Tensor> {
        let shape = shape.into();
        let mut curosr = self.cursor.load(std::sync::atomic::Ordering::SeqCst);
        while curosr < self.group.len() {
            match self.do_get(curosr, dtype, &shape, zero) {
                Ok(s) => {
                    return Ok(s);
                }
                Err(_) => {
                    curosr += 1;
                }
            }
        }
        if curosr > self.group.len() {
            curosr = self.group.len();
        }
        self.cursor
            .store(curosr, std::sync::atomic::Ordering::SeqCst);

        let n = shape.elem_count();
        let allign_n = (n + ALLIGNMENT - 1) / ALLIGNMENT * ALLIGNMENT;
        let new_block_len = if allign_n > self.min_block_size {
            allign_n
        } else {
            self.min_block_size
        };
        // println!(
        //     "###need new arena for {:?} with n:{}, allign_n:{}, new_block_len:{}, at size:{}, cursor:{}",
        //     dtype, n, allign_n,new_block_len, self.group.len(), self.cursor.load(std::sync::atomic::Ordering::SeqCst)
        // );
        //println!("create with n:{}/{}/{}", new_block_len, allign_n, n);
        let segment = TensorArenaUnit::new(new_block_len, dtype, device)?;
        self.group.push(segment);
        // self.cursor
        //     .store(curosr + 1, std::sync::atomic::Ordering::SeqCst);
        self.do_get(curosr, dtype, &shape, zero)
    }
}

#[allow(dead_code)]
#[derive(Debug)]
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

    pub fn print_stat(&self) {
        tracing::info!("TensorArena stat:");
        for cache in &self.cache {
            tracing::info!(
                "CacheGroup Curosr:{}, Size:{}, capacity:{}",
                cache.cursor.load(std::sync::atomic::Ordering::SeqCst),
                cache.group.len(),
                cache.capacity()
            );

            //tracing::info!("TensorArena stat:);
        }
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
    ) -> candle::Result<Tensor> {
        let idx = dtype as usize;
        let _cache = &self.cache[idx];
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
    ) -> candle::Result<Tensor> {
        if !self.device.same_device(device) {
            return Err(candle::Error::DeviceMismatchBinaryOp {
                lhs: self.device.location(),
                rhs: device.location(),
                op: "new_tensor",
            });
        }
        self.get(dtype, shape, zero)
    }
}

#[test]
fn test_arena() -> candle::Result<()> {
    let device = Device::new_cuda(0)?;
    let mut arena = TensorArena::new(&device);

    let _t1 = arena.get(DType::F16, (1, 32000), false)?;
    let _t1 = arena.get(DType::F16, (1, 32000), false)?;
    arena.reset();
    arena.print_stat();

    let _t1 = arena.get(DType::F16, (1, 32000), false)?;
    let _t1 = arena.get(DType::F16, (1, 32000), false)?;
    arena.reset();
    arena.print_stat();
    Ok(())
}
