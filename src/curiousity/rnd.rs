use tch::Tensor;

use super::base_curiousity::{BaseCuriousity, BaseCuriousityModel};
use crate::misc::bounded_vec_deque::BoundedVecDeque;
use crate::memory::Experience;
use std::sync::Arc;

pub struct RND {
    model: Box<dyn BaseCuriousityModel>,
    experiences: BoundedVecDeque<Arc<Experience>>,
    update_interval: usize,
}

impl RND {
    pub fn new(model: Box<dyn BaseCuriousityModel>, update_interval: usize) -> Self {
        RND {
            model,
            experiences: BoundedVecDeque::new(update_interval),
            update_interval,
        }
    }

    fn _update(&mut self) {

    }
}

impl BaseCuriousity for RND {
    fn calc_reward(&self, experience: Arc<Experience>) -> Tensor {
        self.model.forward()
    }

    fn observe(&mut self, experience: Arc<Experience>) {
        self.experiences.push_back(experience.clone());
        if self.update_interval <= self.experiences.len() {
            self._update();
        }
    }
}
