use tch::{Device, Tensor};

pub trait BasecuriosityModel {
    fn forward(&self, x: &Tensor) -> Tensor;
    fn device(&self) -> Device;
    fn save(&self, path: &str);
    fn load(&mut self, path: &str);
}
