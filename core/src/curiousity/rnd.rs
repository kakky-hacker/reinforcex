use super::base_curiousity::BaseCuriousity;
use crate::memory::Experience;
use crate::misc::batch_states::batch_states;
use crate::misc::bounded_vec_deque::BoundedVecDeque;
use crate::models::BaseCuriousityModel;
use std::sync::Arc;
use tch::{no_grad, Tensor};

pub struct RND {
    model: Box<dyn BaseCuriousityModel>,
    experiences: BoundedVecDeque<Arc<Experience>>,
    update_interval: usize,
}

impl RND {
    pub fn new(model: Box<dyn BaseCuriousityModel>, update_interval: usize) -> Self {
        assert!(update_interval > 0);

        RND {
            model,
            experiences: BoundedVecDeque::new(update_interval),
            update_interval,
        }
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

        let _ = self.model.update(&states);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::FCRNDModel;
    use std::sync::Mutex;
    use tch::{nn, Device, Kind};
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
        let model = FCRNDModel::new(predictor_vs, target_vs, 4, 8, 1, 16, 1e-3);
        let rnd = RND::new(Box::new(model), 2);
        let exp = experience(Tensor::randn([1, 4], (Kind::Float, Device::Cpu)));

        let reward = rnd.calc_reward(exp);

        assert_eq!(reward.size(), vec![1]);
        assert!(reward.isfinite().all().int64_value(&[]) == 1);
    }

    #[test]
    fn test_rnd_observe_updates_periodically() {
        let predictor_vs = nn::VarStore::new(Device::Cpu);
        let target_vs = nn::VarStore::new(Device::Cpu);
        let model = FCRNDModel::new(predictor_vs, target_vs, 4, 8, 1, 16, 1e-3);
        let mut rnd = RND::new(Box::new(model), 2);

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
}
