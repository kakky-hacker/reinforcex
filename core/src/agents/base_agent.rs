use std::path::Path;

use tch::Tensor;

use ulid::Ulid;

pub trait BaseAgent {
    fn act_and_train(&mut self, obs: &Tensor, reward: f64) -> Tensor;
    fn act(&self, obs: &Tensor) -> Tensor;
    fn stop_episode_and_train(&mut self, obs: &Tensor, reward: f64);
    fn get_statistics(&self) -> Vec<(String, f64)>;
    fn get_agent_id(&self) -> &Ulid;
    fn save(&self);
    fn load(&mut self);
}

pub(crate) fn ensure_parent_dir(path: &str) {
    let path = Path::new(path);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .unwrap_or_else(|e| panic!("failed to create model directory {:?}: {}", parent, e));
        }
    }
}
