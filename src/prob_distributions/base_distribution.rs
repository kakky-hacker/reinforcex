use candle_core::{Result, Tensor};

pub trait BaseDistribution {
    fn params(&self) -> (&Tensor, &Tensor);
    fn kl(&self, q: Box<dyn BaseDistribution>) -> Result<Tensor>;
    fn entropy(&self) -> Result<Tensor>;
    fn sample(&self) -> Result<Tensor>;
    fn prob(&self, x: &Tensor) -> Result<Tensor>;
    fn log_prob(&self, x: &Tensor) -> Result<Tensor>;
    fn copy(&self) -> Box<dyn BaseDistribution>;
    fn most_probable(&self) -> Result<Tensor>;
}
