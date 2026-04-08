use crate::memory::Experience;
use std::sync::Arc;

pub trait BaseCuriousity {
    fn observe(&mut self, experience: Arc<Experience>, record: bool) -> f32;
    fn update(&mut self);
}