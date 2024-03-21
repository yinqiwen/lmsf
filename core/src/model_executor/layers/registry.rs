use std::collections::HashMap;

use candle::{DType, Shape};

#[derive(Debug)]
pub struct WeightRegistry {
    pub(crate) name: &'static str,
    pub(crate) shape: Shape,
    pub(crate) dtype: DType,
    attrs: HashMap<&'static str, usize>,
}

impl WeightRegistry {
    pub fn new(
        name: &'static str,
        shape: Shape,
        dtype: DType,
        attrs: HashMap<&'static str, usize>,
    ) -> Self {
        Self {
            name,
            shape,
            dtype,
            attrs,
        }
    }

    pub fn getattr(&self, name: &'static str) -> Option<usize> {
        self.attrs.get(&name).map(|x| *x)
    }
}
