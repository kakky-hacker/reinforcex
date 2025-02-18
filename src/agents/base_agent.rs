use std::fs;
use std::collections::HashSet;

use tch::Tensor;


pub trait BaseAgent {
    fn act_and_train(&mut self, obs: &Tensor, reward: f64) -> Tensor;

    fn act(&self, obs: &Tensor) -> Tensor;

    fn stop_episode_and_train(&mut self, obs: &Tensor, reward: f64) {
        // Implementation specific logic
        println!("Stop episode and train: state = {}, reward = {}", obs, reward);
    }

    fn stop_episode(&mut self) {
        // Implementation specific logic
        println!("Stop episode");
    }

    fn get_statistics(&self) -> Vec<(String, f64)> {
        // Implement logic for getting statistics
        vec![("average_loss".to_string(), 0.0), ("average_value".to_string(), 1.0)]
    }

    fn saved_attributes(&self) -> Vec<String> {
        vec!["attribute1".to_string(), "attribute2".to_string()]
    }

    fn save(&self, dirname: &str, ancestors: HashSet<String>) {
        fs::create_dir_all(dirname).unwrap();
        let mut ancestors = ancestors;
        ancestors.insert("agent".to_string());

        for attr in self.saved_attributes() {
            if ancestors.contains(&attr) {
                continue;
            }

            println!("Saving attribute: {}", attr);
        }
    }

    fn load(&self, dirname: &str, ancestors: HashSet<String>) {
        let mut ancestors = ancestors;
        ancestors.insert("agent".to_string());

        for attr in self.saved_attributes() {
            if ancestors.contains(&attr) {
                continue;
            }

            println!("Loading attribute: {}", attr);
        }
    }
}
