use tch::{Device, Tensor};

pub trait BaseCuriousityModel {
    fn forward(&self, x: &Tensor) -> Tensor;
    fn update(&mut self, x: &Tensor) -> Tensor;
    fn device(&self) -> Device;
}
