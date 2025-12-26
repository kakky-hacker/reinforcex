use tch::Tensor;

pub trait BaseDistribution: Send + Sync {
    fn params(&self) -> (&Tensor, &Tensor);
    fn kl(&self, q: &Box<dyn BaseDistribution>) -> Tensor;
    fn entropy(&self) -> Tensor;
    fn sample(&self) -> Tensor;
    fn prob(&self, x: &Tensor) -> Tensor;
    fn log_prob(&self, x: &Tensor) -> Tensor;
    fn all_prob(&self) -> Tensor;
    fn all_log_prob(&self) -> Tensor;
    fn copy(&self) -> Box<dyn BaseDistribution>;
    fn most_probable(&self) -> Tensor;
    fn concat(&mut self, others: Vec<Box<dyn BaseDistribution>>);
    fn detach(&mut self);
}
