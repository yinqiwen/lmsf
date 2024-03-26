use std::collections::HashMap;

use candle::{DType, Shape};

#[derive(Debug)]
pub struct WeightRegistry {
    pub(crate) name: &'static str,
    pub(crate) full_shape: Shape,
    pub(crate) shard_shape: Shape,
    pub(crate) dtype: DType,
    attrs: HashMap<&'static str, usize>,
}

impl WeightRegistry {
    pub fn new(
        name: &'static str,
        shard_shape: Shape,
        full_shape: Shape,
        dtype: DType,
        attrs: HashMap<&'static str, usize>,
    ) -> Self {
        // let full_shape = if let Some(s) = full_shape {
        //     s
        // } else {
        //     shard_shape.clone()
        // };
        Self {
            name,
            shard_shape,
            full_shape,
            dtype,
            attrs,
        }
    }

    pub fn getattr(&self, name: &'static str) -> Option<usize> {
        self.attrs.get(&name).map(|x| *x)
    }
}
