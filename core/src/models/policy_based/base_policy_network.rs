use crate::prob_distributions::BaseDistribution;
use tch::{Device, Tensor};

/// Policies may move with their owning agent to a worker thread.
pub trait BasePolicy: Send {
    fn forward(&self, x: &Tensor) -> (Box<dyn BaseDistribution>, Option<Tensor>);
    fn device(&self) -> Device;
    fn save(&self, path: &str);
    fn load(&mut self, path: &str);
}
