use crate::memory::Experience;
use std::sync::Arc;
use tch::Tensor;

pub trait Basecuriosity {
    fn calc_reward(&self, experience: Arc<Experience>) -> Tensor;
    fn observe(&mut self, experience: Arc<Experience>);
    fn save(&self);
    fn load(&mut self);
}
