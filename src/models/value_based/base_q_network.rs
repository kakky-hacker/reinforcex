use tch::Tensor;


pub trait BaseQFunction{
    fn forward(&self, x: &Tensor) -> Tensor;
    fn is_cuda(&self) -> bool;
    fn clone(&self) -> Box<dyn BaseQFunction>;
}