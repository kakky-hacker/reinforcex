use tch::Tensor;
use crate::prob_distributions::BaseDistribution;


pub trait BasePolicy{
    // return (action, value)
    fn forward(&self, x: &Tensor) -> (Box<dyn BaseDistribution>, Option<Tensor>);
    fn is_recurrent(&self) -> bool;
    fn reset_state(&mut self);
    fn is_cuda(&self) -> bool;
}