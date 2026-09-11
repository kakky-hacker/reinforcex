use tch::Tensor;

pub trait BaseDistribution: Send + Sync {
    fn is_discrete(&self) -> bool;
    fn params(&self) -> (&Tensor, &Tensor);
    fn kl(&self, q: &Box<dyn BaseDistribution>) -> Tensor;
    fn entropy(&self) -> Tensor;
    fn sample(&self) -> Tensor;
    /// Map a policy-space action to the environment without changing the value
    /// used by `log_prob`. On-policy algorithms must retain the original action.
    fn to_env_action(&self, action: &Tensor) -> Tensor {
        action.shallow_clone()
    }
    fn prob(&self, x: &Tensor) -> Tensor;
    fn log_prob(&self, x: &Tensor) -> Tensor;
    fn all_prob(&self) -> Tensor;
    fn all_log_prob(&self) -> Tensor;
    fn copy(&self) -> Box<dyn BaseDistribution>;
    fn most_probable(&self) -> Tensor;
    fn concat(&mut self, others: Vec<Box<dyn BaseDistribution>>);
    fn detach(&mut self);
}
