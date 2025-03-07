use crate::prob_distributions::BaseDistribution;
use candle_core::{Result, Tensor};

pub trait BasePolicy {
    // return (action, value)
    fn forward(&self, x: &Tensor) -> Result<(Box<dyn BaseDistribution>, Option<Tensor>)>;
    fn is_cuda(&self) -> bool;
}
