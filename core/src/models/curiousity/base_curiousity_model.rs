use tch::{Device, Tensor};

pub trait BaseCuriousityModel {
    fn forward(&self, x: &Tensor) -> Tensor;
    fn device(&self) -> Device;
    fn save(&self, path: &str);
    fn load(&mut self, path: &str);
}
