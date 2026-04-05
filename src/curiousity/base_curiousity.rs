use crate::memory::Experience;
use std::sync::Arc;
use tch::{Device, Tensor};

pub trait BaseCuriousity {
    fn calc_reward(&self, experience: Arc<Experience>) -> Tensor;
    fn observe(&mut self, experience: Arc<Experience>) {
        self.experiences.push_back(experience.clone());
        if self.update_interval <= self.experiences.len() {
            self._update();
        }
    }
}

pub trait BaseCuriousityModel {
    fn forward(&self, x: &Tensor) -> Tensor;
    fn device(&self) -> Device;
}
