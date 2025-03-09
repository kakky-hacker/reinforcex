use candle_core::{Device, Result, Tensor};

pub trait BaseQFunction {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
    fn is_cuda(&self) -> bool;
    fn get_device(&self) -> &Device;
    fn clone(&self) -> Box<dyn BaseQFunction>;
}
