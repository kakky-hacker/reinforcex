use super::base_agent::BaseAgent;
use crate::misc::batch_states::batch_states;
use crate::misc::cumsum;
use crate::models::BasePolicy;
use crate::prob_distributions::BaseDistribution;
use candle_core::{Device, Result, Tensor};
use candle_nn::optim::Optimizer;
use std::fs;
use std::{collections::HashSet, ops::Deref};

pub struct REINFORCE<T: Optimizer> {
    model: Box<dyn BasePolicy>,
    optimizer: T,
    gamma: f64,
    beta: f64,
    batchsize: usize,
    act_deterministically: bool,
    average_entropy_decay: f64,

    // Statistics
    average_entropy: f64,

    // State management
    t: usize,
    reward_sequences: Vec<Vec<f64>>,
    log_prob_sequences: Vec<Vec<Tensor>>,
    entropy_sequences: Vec<Vec<Tensor>>,
    value_sequences: Vec<Vec<Option<Tensor>>>,
}

impl<T: Optimizer> REINFORCE<T> {
    pub fn new(
        model: Box<dyn BasePolicy>,
        optimizer: T,
        gamma: f64,
        beta: f64,
        batchsize: usize,
        act_deterministically: bool,
        average_entropy_decay: f64,
    ) -> Self {
        REINFORCE {
            model,
            optimizer,
            gamma,
            beta,
            batchsize,
            act_deterministically,
            average_entropy_decay,
            average_entropy: 0.0,
            t: 0,
            reward_sequences: vec![vec![]],
            log_prob_sequences: vec![vec![]],
            entropy_sequences: vec![vec![]],
            value_sequences: vec![vec![]],
        }
    }

    fn _compute_loss(&mut self) -> Result<Tensor> {
        let mut loss = Tensor::from_slice(&[0.0], &[], &Device::Cpu)?;

        // Iterate over the reward, log probabilities, and entropy sequences
        for (r_seq, log_prob_seq, ent_seq, v_seq) in self
            .reward_sequences
            .iter()
            .zip(self.log_prob_sequences.iter())
            .zip(self.entropy_sequences.iter())
            .zip(self.value_sequences.iter())
            .map(|(((r, log_prob), ent), v)| (r, log_prob, ent, v))
        {
            assert_eq!(r_seq.len() - 1, log_prob_seq.len());
            assert_eq!(log_prob_seq.len(), ent_seq.len());

            // Calculate returns (sum of future rewards)
            let g_seq = cumsum::cumsum_rev(&r_seq[1..], self.gamma);

            assert_eq!(g_seq.len(), log_prob_seq.len());

            // Compute the losses based on rewards, log probabilities, and entropy
            for (((g, log_prob), entropy), v) in g_seq
                .iter()
                .zip(log_prob_seq.iter())
                .zip(ent_seq.iter())
                .zip(v_seq.iter())
            {
                let _loss;
                let _g = *g;
                let _log_prob = log_prob;
                let _entropy = entropy;
                let _v = v;
                if v.is_none() {
                    _loss = (((-_g * _log_prob)? - (self.beta * _entropy)?)? / r_seq.len() as f64)?;
                } else {
                    let v_tensor = _v.as_ref().unwrap().detach();
                    let advantage = (_g - &v_tensor)?;
                    let actor_loss = ((-1.0 * advantage * _log_prob)? - self.beta * _entropy)?;
                    let critic_loss = (&v_tensor - _g)?.powf(2.0)?;
                    _loss = ((actor_loss + critic_loss)? / r_seq.len() as f64)?;
                }
                loss = (loss + _loss / self.batchsize as f64)?;
            }
        }

        // Reset the reward, log probability, and entropy sequences for the next episode
        self.reward_sequences = vec![vec![]];
        self.log_prob_sequences = vec![vec![]];
        self.entropy_sequences = vec![vec![]];
        self.value_sequences = vec![vec![]];

        Ok(loss)
    }

    // Perform a batch update by accumulating gradients for all episodes in the batch
    fn _update(&mut self) -> Result<()> {
        assert_eq!(self.reward_sequences.len(), self.batchsize);
        assert_eq!(self.log_prob_sequences.len(), self.batchsize);
        assert_eq!(self.entropy_sequences.len(), self.batchsize);
        assert_eq!(self.value_sequences.len(), self.batchsize);

        // First accumulate gradients for the batch
        let loss = self._compute_loss()?;

        // Perform the optimizer update
        self.optimizer.backward_step(&loss);
        Ok(())
    }
}

impl<T: Optimizer> BaseAgent for REINFORCE<T> {
    fn act_and_train(&mut self, obs: &Tensor, reward: f64) -> Result<Tensor> {
        let state = batch_states(&vec![obs.clone()], self.model.get_device())?;

        // Get action distribution from the model
        let (action_distrib, value) = self.model.forward(&state)?;

        // Sample an action from the distribution
        let action = action_distrib.sample()?.to_device(&Device::Cpu)?;

        // Save values used to compute losses
        if let Some(last_vec) = self.reward_sequences.last_mut() {
            last_vec.push(reward);
        }
        if let Some(last_vec) = self.log_prob_sequences.last_mut() {
            last_vec.push(action_distrib.log_prob(&action)?);
        }
        if let Some(last_vec) = self.entropy_sequences.last_mut() {
            last_vec.push(action_distrib.entropy()?);
        }
        if let Some(last_vec) = self.value_sequences.last_mut() {
            last_vec.push(value);
        }

        // Update stats for entropy
        self.average_entropy += (1.0 - self.average_entropy_decay)
            * (action_distrib.entropy()?.to_vec0::<f64>()? - self.average_entropy);

        // Increment the time step
        self.t += 1;

        // Return the action
        Ok(action)
    }

    fn act(&self, obs: &Tensor) -> Result<Tensor> {
        let state = batch_states(&vec![obs.clone()], self.model.get_device())?.detach();

        // Get action distribution from the model
        let (action_distrib, _) = self.model.forward(&state)?;

        let action;

        if self.act_deterministically {
            // Choose the most probable action
            action = action_distrib.most_probable()?;
        } else {
            // Sample an action from the distribution
            action = action_distrib.sample()?;
        }

        action.detach().to_device(&Device::Cpu)
    }

    fn stop_episode_and_train(&mut self, obs: &Tensor, reward: f64) -> Result<()> {
        // Add reward to the sequences
        if let Some(last_vec) = self.reward_sequences.last_mut() {
            last_vec.push(reward);
        }

        if self.reward_sequences.len() == self.batchsize {
            self._update();
        } else {
            // Prepare for the next episode
            self.reward_sequences.push(vec![]);
            self.log_prob_sequences.push(vec![]);
            self.entropy_sequences.push(vec![]);
            self.value_sequences.push(vec![]);
        }

        Ok(())
    }

    fn get_statistics(&self) -> Vec<(String, f64)> {
        vec![("average_entropy".to_string(), self.average_entropy as f64)]
    }

    fn saved_attributes(&self) -> Vec<String> {
        vec!["model".to_string(), "optimizer".to_string()]
    }

    fn save(&self, dirname: &str, ancestors: HashSet<String>) {
        fs::create_dir_all(dirname).unwrap();
        let mut ancestors: HashSet<String> = ancestors;
        ancestors.insert("agent".to_string());

        for attr in self.saved_attributes() {
            if ancestors.contains(&attr) {
                continue;
            }

            println!("Saving attribute: {}", attr);
        }
    }

    fn load(&self, dirname: &str, ancestors: HashSet<String>) {
        let mut ancestors: HashSet<String> = ancestors;
        ancestors.insert("agent".to_string());

        for attr in self.saved_attributes() {
            if ancestors.contains(&attr) {
                continue;
            }

            println!("Loading attribute: {}", attr);
        }
    }
}
