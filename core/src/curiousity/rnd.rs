use super::base_curiousity::BaseCuriousity;
use crate::memory::Experience;
use crate::misc::batch_states::batch_states;
use crate::misc::bounded_vec_deque::BoundedVecDeque;
use crate::models::BaseCuriousityModel;
use std::sync::Arc;
use tch::{nn, no_grad, Kind, Tensor};

pub struct RND {
    model: Box<dyn BaseCuriousityModel + Send>,
    optimizer: nn::Optimizer,
    experiences: BoundedVecDeque<Arc<Experience>>,
    update_interval: usize,
    save_path: Option<String>,
    load_path: Option<String>,
}

impl RND {
    pub fn new(
        model: Box<dyn BaseCuriousityModel + Send>,
        optimizer: nn::Optimizer,
        update_interval: usize,
        save_path: Option<String>,
        load_path: Option<String>,
    ) -> Self {
        assert!(update_interval > 0);

        let mut rnd = RND {
            model,
            optimizer,
            experiences: BoundedVecDeque::new(update_interval),
            update_interval,
            save_path,
            load_path,
        };
        rnd.load();
        rnd
    }

    fn _update(&mut self) {
        if self.experiences.is_empty() {
            return;
        }

        let states = self
            .experiences
            .to_vec()
            .iter()
            .map(|experience| experience.state.shallow_clone())
            .collect::<Vec<Tensor>>();
        let states = batch_states(&states, self.model.device());

        let loss = self.model.forward(&states).mean(Kind::Float);
        self.optimizer.zero_grad();
        loss.backward();
        self.optimizer.step();
        self.experiences.empty();
    }
}

impl BaseCuriousity for RND {
    fn calc_reward(&self, experience: Arc<Experience>) -> Tensor {
        no_grad(|| {
            self.model
                .forward(&experience.state.to_device(self.model.device()))
        })
        .detach()
    }

    fn observe(&mut self, experience: Arc<Experience>) {
        self.experiences.push_back(experience.clone());
        if self.update_interval <= self.experiences.len() {
            self._update();
        }
    }

    fn save(&self) {
        if let Some(path) = &self.save_path {
            if path.is_empty() {
                return;
            }
            self.model.save(path);
        }
    }

    fn load(&mut self) {
        if let Some(path) = self.load_path.clone() {
            if path.is_empty() {
                return;
            }
            self.model.load(&path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::FCRNDModel;
    use std::sync::Mutex;
    use tch::{nn, nn::OptimizerConfig, Device, Kind};
    use ulid::Ulid;

    fn experience(state: Tensor) -> Arc<Experience> {
        Arc::new(Experience::new(
            Ulid::new(),
            Ulid::new(),
            state,
            None,
            None,
            0.0,
            false,
            Mutex::new(None),
            Mutex::new(None),
        ))
    }

    #[test]
    fn test_rnd_calc_reward() {
        let predictor_vs = nn::VarStore::new(Device::Cpu);
        let target_vs = nn::VarStore::new(Device::Cpu);
        let model = FCRNDModel::new(predictor_vs, target_vs, 4, 8, 1, 16);
        let optimizer = nn::Adam::default()
            .build(model.predictor_var_store(), 1e-3)
            .unwrap();
        let rnd = RND::new(Box::new(model), optimizer, 2, None, None);
        let exp = experience(Tensor::randn([1, 4], (Kind::Float, Device::Cpu)));

        let reward = rnd.calc_reward(exp);

        assert_eq!(reward.size(), vec![1]);
        assert!(reward.isfinite().all().int64_value(&[]) == 1);
    }

    #[test]
    fn test_rnd_observe_updates_periodically() {
        let predictor_vs = nn::VarStore::new(Device::Cpu);
        let target_vs = nn::VarStore::new(Device::Cpu);
        let model = FCRNDModel::new(predictor_vs, target_vs, 4, 8, 1, 16);
        let optimizer = nn::Adam::default()
            .build(model.predictor_var_store(), 1e-3)
            .unwrap();
        let mut rnd = RND::new(Box::new(model), optimizer, 2, None, None);

        rnd.observe(experience(Tensor::randn(
            [1, 4],
            (Kind::Float, Device::Cpu),
        )));
        assert_eq!(rnd.experiences.len(), 1);

        rnd.observe(experience(Tensor::randn(
            [1, 4],
            (Kind::Float, Device::Cpu),
        )));
        assert_eq!(rnd.experiences.len(), 0);
    }

    #[test]
    fn test_rnd_save_and_load() {
        let dirname = std::env::temp_dir().join(format!("reinforcex-rnd-{}", Ulid::new()));
        let dirname = dirname.to_string_lossy().into_owned();
        let state = Tensor::randn([1, 4], (Kind::Float, Device::Cpu));
        let exp = experience(state.shallow_clone());

        let predictor_vs = nn::VarStore::new(Device::Cpu);
        let target_vs = nn::VarStore::new(Device::Cpu);
        let model = FCRNDModel::new(predictor_vs, target_vs, 4, 8, 1, 16);
        let optimizer = nn::Adam::default()
            .build(model.predictor_var_store(), 1e-3)
            .unwrap();
        let rnd = RND::new(Box::new(model), optimizer, 2, Some(dirname.clone()), None);
        let expected_reward = rnd.calc_reward(Arc::clone(&exp));
        rnd.save();

        let predictor_vs = nn::VarStore::new(Device::Cpu);
        let target_vs = nn::VarStore::new(Device::Cpu);
        let model = FCRNDModel::new(predictor_vs, target_vs, 4, 8, 1, 16);
        let optimizer = nn::Adam::default()
            .build(model.predictor_var_store(), 1e-3)
            .unwrap();
        let loaded_rnd = RND::new(Box::new(model), optimizer, 2, None, Some(dirname.clone()));
        let actual_reward = loaded_rnd.calc_reward(exp);

        assert!(expected_reward.allclose(&actual_reward, 1e-6, 1e-6, false));

        let _ = std::fs::remove_dir_all(dirname);
    }
}
