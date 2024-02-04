use candle_core::Shape;

pub(crate) fn get_column_major_dim(s: &Shape, dim: usize) -> candle_core::Result<usize> {
    if s.dims().len() == 1 {
        Ok(0)
    } else if s.dims().len() == 2 {
        if dim == 1 {
            Ok(0_usize)
        } else if dim == 0 {
            Ok(1_usize)
        } else {
            candle_core::bail!("Invalid dim:{:?}", dim)
        }
    } else {
        candle_core::bail!("Not supported shapes:{:?}", s)
    }
}
