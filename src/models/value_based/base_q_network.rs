use tch::Tensor;


pub trait BaseQFunction{
    fn forward(&self, x: &Tensor) -> Tensor;
    fn is_recurrent(&self) -> bool;
    fn reset_state(&mut self);
    fn is_cuda(&self) -> bool;
    fn clone(&self) -> Box<dyn BaseQFunction>;
}