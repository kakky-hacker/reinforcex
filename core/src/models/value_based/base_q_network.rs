use crate::misc::autograd::no_grad;
use tch::{Device, Tensor};

/// Q-functions may move with their owning agent to a worker thread.
pub trait BaseQFunction: Send {
    fn forward(&self, x: &Tensor) -> Tensor;
    fn device(&self) -> Device;
    fn clone(&self) -> Box<dyn BaseQFunction>;
    fn save(&self, path: &str);
    fn load(&mut self, path: &str);
    fn trainable_variables(&self) -> Vec<Tensor>;

    /// Opt in only when `copy_from` synchronizes all state used by `forward`,
    /// including non-trainable buffers, without replacing tensor storage or
    /// consuming random numbers. DQN otherwise retains clone-based syncing.
    fn supports_in_place_target_sync(&self) -> bool {
        false
    }

    fn copy_from(&mut self, source: &dyn BaseQFunction) {
        let mut target_variables = self.trainable_variables();
        let source_variables = source.trainable_variables();
        assert_eq!(target_variables.len(), source_variables.len());

        no_grad(|| {
            for (target, source) in target_variables.iter_mut().zip(source_variables.iter()) {
                target.copy_(source);
            }
        });
    }

    fn soft_update_from(&mut self, source: &dyn BaseQFunction, tau: f64) {
        assert!((0.0..=1.0).contains(&tau));

        let mut target_variables = self.trainable_variables();
        let source_variables = source.trainable_variables();
        assert_eq!(target_variables.len(), source_variables.len());

        no_grad(|| {
            for (target, source) in target_variables.iter_mut().zip(source_variables.iter()) {
                let updated = target.shallow_clone() * (1.0 - tau) + source * tau;
                target.copy_(&updated);
            }
        });
    }
}
