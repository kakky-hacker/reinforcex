use super::base_curiosity::Basecuriosity;
use crate::memory::Experience;
use crate::misc::batch_states::batch_states;
use crate::models::BasecuriosityModel;
use std::sync::Arc;
use tch::{nn, no_grad, Kind, Tensor};

pub struct RND {
    model: Box<dyn BasecuriosityModel + Send>,
    optimizer: nn::Optimizer,
    minibatch_size: usize,
    save_path: Option<String>,
    load_path: Option<String>,
}

impl RND {
    pub fn new(
        model: Box<dyn BasecuriosityModel + Send>,
        optimizer: nn::Optimizer,
        minibatch_size: usize,
        save_path: Option<String>,
        load_path: Option<String>,
    ) -> Self {
        assert!(minibatch_size > 0);

        let mut rnd = RND {
            model,
            optimizer,
            minibatch_size,
            save_path,
            load_path,
        };
        rnd.load();
        rnd
    }
}

impl Basecuriosity for RND {
    fn calc_internal_reward(&self, experiences: &[Arc<Experience>]) -> Tensor {
        if experiences.is_empty() {
            return Tensor::zeros([0], (Kind::Float, self.model.device()));
        }

        let states = experiences
            .iter()
            .map(|experience| experience.state.shallow_clone())
            .collect::<Vec<Tensor>>();
        let states = batch_states(&states, self.model.device());
        no_grad(|| self.model.forward(&states)).detach()
    }

    fn update(&mut self, experiences: &[Arc<Experience>]) {
        for minibatch in experiences.chunks(self.minibatch_size) {
            let states = minibatch
                .iter()
                .map(|experience| experience.state.shallow_clone())
                .collect::<Vec<Tensor>>();
            let states = batch_states(&states, self.model.device());

            let loss = self.model.forward(&states).mean(Kind::Float);
            self.optimizer.zero_grad();
            loss.backward();
            self.optimizer.step();
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
    use tch::{nn, nn::OptimizerConfig, Device, Kind};
    use ulid::Ulid;

    struct ScalarErrorModel {
        var_store: nn::VarStore,
        predictor: Tensor,
    }

    impl ScalarErrorModel {
        fn new() -> Self {
            let var_store = nn::VarStore::new(Device::Cpu);
            let predictor = var_store.root().var("predictor", &[], nn::Init::Const(1.0));
            Self {
                var_store,
                predictor,
            }
        }
    }

    impl BasecuriosityModel for ScalarErrorModel {
        fn forward(&self, x: &Tensor) -> Tensor {
            (x.view([-1, 1]) * &self.predictor)
                .square()
                .mean_dim(&[1i64][..], false, Kind::Float)
        }

        fn device(&self) -> Device {
            self.var_store.device()
        }

        fn save(&self, _path: &str) {}

        fn load(&mut self, _path: &str) {}
    }

    fn experience(state: Tensor) -> Arc<Experience> {
        Arc::new(Experience::new(
            Ulid::new(),
            Ulid::new(),
            state,
            None,
            None,
            0.0,
            false,
        ))
    }

    #[test]
    fn test_rnd_calc_internal_reward_in_batch() {
        let predictor_vs = nn::VarStore::new(Device::Cpu);
        let target_vs = nn::VarStore::new(Device::Cpu);
        let model = FCRNDModel::new(predictor_vs, target_vs, 4, 8, 1, 16);
        let optimizer = nn::Adam::default()
            .build(model.predictor_var_store(), 1e-3)
            .unwrap();
        let rnd = RND::new(Box::new(model), optimizer, 2, None, None);
        let experiences = (0..3)
            .map(|_| experience(Tensor::randn([1, 4], (Kind::Float, Device::Cpu))))
            .collect::<Vec<Arc<Experience>>>();

        let reward = rnd.calc_internal_reward(&experiences);

        assert_eq!(reward.size(), vec![3]);
        assert!(reward.isfinite().all().int64_value(&[]) == 1);
    }

    #[test]
    fn test_rnd_update_reduces_prediction_error() {
        let model = ScalarErrorModel::new();
        let optimizer = nn::Sgd::default().build(&model.var_store, 0.1).unwrap();
        let mut rnd = RND::new(Box::new(model), optimizer, 2, None, None);
        let experiences = (0..4)
            .map(|_| experience(Tensor::ones([1, 1], (Kind::Float, Device::Cpu))))
            .collect::<Vec<_>>();
        let reward_before = rnd
            .calc_internal_reward(&experiences)
            .mean(Kind::Float)
            .double_value(&[]);

        rnd.update(&experiences);

        let reward_after = rnd
            .calc_internal_reward(&experiences)
            .mean(Kind::Float)
            .double_value(&[]);
        assert!(
            reward_after < reward_before,
            "RND update should reduce predictor error: before={reward_before}, after={reward_after}"
        );
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
        let expected_reward = rnd.calc_internal_reward(&[Arc::clone(&exp)]);
        rnd.save();

        let predictor_vs = nn::VarStore::new(Device::Cpu);
        let target_vs = nn::VarStore::new(Device::Cpu);
        let model = FCRNDModel::new(predictor_vs, target_vs, 4, 8, 1, 16);
        let optimizer = nn::Adam::default()
            .build(model.predictor_var_store(), 1e-3)
            .unwrap();
        let loaded_rnd = RND::new(Box::new(model), optimizer, 2, None, Some(dirname.clone()));
        let actual_reward = loaded_rnd.calc_internal_reward(&[exp]);

        assert!(expected_reward.allclose(&actual_reward, 1e-6, 1e-6, false));

        let _ = std::fs::remove_dir_all(dirname);
    }
}
