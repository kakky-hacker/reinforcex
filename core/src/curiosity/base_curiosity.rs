use crate::memory::Experience;
use std::sync::{Arc, Mutex};
use tch::Tensor;

pub trait Basecuriosity {
    fn calc_internal_reward(&self, experiences: &[Arc<Experience>]) -> Tensor;
    fn update(&mut self, experiences: &[Arc<Experience>]);
    fn save(&self);
    fn load(&mut self);
}

impl<T: Basecuriosity + ?Sized> Basecuriosity for Box<T> {
    fn calc_internal_reward(&self, experiences: &[Arc<Experience>]) -> Tensor {
        (**self).calc_internal_reward(experiences)
    }

    fn update(&mut self, experiences: &[Arc<Experience>]) {
        (**self).update(experiences)
    }

    fn save(&self) {
        (**self).save()
    }

    fn load(&mut self) {
        (**self).load()
    }
}

impl<T: Basecuriosity + ?Sized> Basecuriosity for Arc<Mutex<T>> {
    fn calc_internal_reward(&self, experiences: &[Arc<Experience>]) -> Tensor {
        self.lock().unwrap().calc_internal_reward(experiences)
    }

    fn update(&mut self, experiences: &[Arc<Experience>]) {
        self.lock().unwrap().update(experiences)
    }

    fn save(&self) {
        self.lock().unwrap().save()
    }

    fn load(&mut self) {
        self.lock().unwrap().load()
    }
}
