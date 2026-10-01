use crate::memory::Experience;
use std::sync::{Arc, Mutex};
use tch::Tensor;

/// Curiosity modules may move between workers; shared modules must synchronize
/// their mutable predictor state (for example with `Arc<Mutex<_>>`).
pub trait Basecuriosity: Send {
    fn calc_internal_reward(&self, experiences: &[Arc<Experience>]) -> Tensor;
    fn update(&mut self, experiences: &[Arc<Experience>]);
    /// Return novelty before training on this rollout. Shared implementations
    /// should hold their lock across both operations.
    fn calc_internal_reward_and_update(&mut self, experiences: &[Arc<Experience>]) -> Tensor {
        let rewards = self.calc_internal_reward(experiences);
        self.update(experiences);
        rewards
    }
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

    fn calc_internal_reward_and_update(&mut self, experiences: &[Arc<Experience>]) -> Tensor {
        (**self).calc_internal_reward_and_update(experiences)
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

    fn calc_internal_reward_and_update(&mut self, experiences: &[Arc<Experience>]) -> Tensor {
        self.lock()
            .unwrap()
            .calc_internal_reward_and_update(experiences)
    }

    fn save(&self) {
        self.lock().unwrap().save()
    }

    fn load(&mut self) {
        self.lock().unwrap().load()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CountingCuriosity {
        updates: usize,
    }
    impl Basecuriosity for CountingCuriosity {
        fn calc_internal_reward(&self, _: &[Arc<Experience>]) -> Tensor {
            std::thread::yield_now();
            Tensor::from_slice(&[self.updates as f32])
        }
        fn update(&mut self, _: &[Arc<Experience>]) {
            self.updates += 1;
        }
        fn save(&self) {}
        fn load(&mut self) {}
    }

    #[test]
    fn test_shared_curiosity_calculates_and_updates_under_one_lock() {
        let shared = Arc::new(Mutex::new(CountingCuriosity { updates: 0 }));
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let rewards = std::thread::scope(|scope| {
            let handles = (0..8)
                .map(|_| {
                    let mut curiosity = Arc::clone(&shared);
                    let barrier = Arc::clone(&barrier);
                    scope.spawn(move || {
                        barrier.wait();
                        curiosity
                            .calc_internal_reward_and_update(&[])
                            .double_value(&[0]) as usize
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        let mut rewards = rewards;
        rewards.sort_unstable();
        assert_eq!(rewards, (0..8).collect::<Vec<_>>());
        assert_eq!(shared.lock().unwrap().updates, 8);
    }
}
