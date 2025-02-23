use tch::Tensor;

pub trait BaseDistribution {
    fn params(&self) -> (&Tensor, &Tensor);
    fn kl(&self, q: Box<dyn BaseDistribution>) -> Tensor;
    fn entropy(&self) -> Tensor;
    fn sample(&self) -> Tensor;
    fn prob(&self, x: &Tensor) -> Tensor;
    fn log_prob(&self, x: &Tensor) -> Tensor;
    fn copy(&self) -> Box<dyn BaseDistribution>;
    fn most_probable(&self) -> Tensor;
}
