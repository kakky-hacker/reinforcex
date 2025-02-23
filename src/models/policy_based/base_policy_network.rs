use tch::Tensor;
use crate::prob_distributions::BaseDistribution;


pub trait BasePolicy{
    // return (action, value)
    fn forward(&self, x: &Tensor) -> (Box<dyn BaseDistribution>, Option<Tensor>);
    fn is_cuda(&self) -> bool;
}