use crate::prob_distributions::BaseDistribution;
use tch::{Device, Tensor};

pub trait BasePolicy {
    fn forward(&self, x: &Tensor) -> (Box<dyn BaseDistribution>, Option<Tensor>);
    fn device(&self) -> Device;
}
