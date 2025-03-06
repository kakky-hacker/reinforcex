use candle_core::{Result, Tensor};

pub trait BaseQFunction {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
    fn is_cuda(&self) -> bool;
    fn clone(&self) -> Box<dyn BaseQFunction>;
}
