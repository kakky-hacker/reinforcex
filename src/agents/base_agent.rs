use std::collections::HashSet;

use tch::Tensor;

pub trait BaseAgent {
    fn act_and_train(&mut self, obs: &Tensor, reward: f64) -> Tensor;
    fn act(&self, obs: &Tensor) -> Tensor;
    fn stop_episode_and_train(&mut self, obs: &Tensor, reward: f64);
    fn get_statistics(&self) -> Vec<(String, f64)>;
    fn save(&self, dirname: &str, ancestors: HashSet<String>);
    fn load(&self, dirname: &str, ancestors: HashSet<String>);
}
