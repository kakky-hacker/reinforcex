use crate::memory::Experience;
use std::sync::Arc;

pub trait BaseCuriousity {
    fn calc_reward(&self, experience: Arc<Experience>);
    fn observe(&mut self, experience: Arc<Experience>);
    fn update(&mut self);
}
