use super::base_curiousity::BaseCuriousity;
use crate::memory::Experience;
use std::sync::Arc;

pub struct RND {}

impl RND {
    pub fn new() -> Self {
        RND {}
    }
}

impl BaseCuriousity for RND {
    fn calc_reward(&self, experience: Arc<Experience>) {}

    fn observe(&mut self, experience: Arc<Experience>) {}

    fn update(&mut self) {}
}
