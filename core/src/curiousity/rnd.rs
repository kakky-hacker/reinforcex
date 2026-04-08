use super::base_curiousity::BaseCuriousity;
use crate::memory::Experience;
use std::sync::Arc;

pub struct RND {

}

impl RND {
    pub fn new() -> Self {
        RND {

        }
    }
}

impl BaseCuriousity for RND {
    fn observe(&mut self, experience: Arc<Experience>, record: bool) -> f32 {
        1.0
    }

    fn update(&mut self) {
        
    }
}
