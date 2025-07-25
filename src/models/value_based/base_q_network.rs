use tch::{Device, Tensor};

pub trait BaseQFunction {
    fn forward(&self, x: &Tensor) -> Tensor;
    fn device(&self) -> Device;
    fn clone(&self) -> Box<dyn BaseQFunction>;
}
