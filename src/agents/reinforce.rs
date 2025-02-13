use std::collections::HashSet;
use std::fs;
use tch::{nn, Tensor, Device, no_grad};
use super::base_agent::BaseAgent;
use crate::models::BasePolicy;
use crate::prob_distributions::BaseDistribution;
use crate::misc::batch_states::batch_states;
use crate::misc::cumsum;

pub struct REINFORCE {
    model: Box<dyn BasePolicy>,
    optimizer: nn::Optimizer,
    gamma: f64,
    beta: f64,
    batchsize: usize,
    act_deterministically: bool,
    average_entropy_decay: f64,
    backward_separately: bool,

    // Statistics
    average_entropy: f64,

    // State management
    t: usize,
    reward_sequences: Vec<Vec<f64>>,
    log_prob_sequences: Vec<Vec<Tensor>>,
    entropy_sequences: Vec<Vec<Tensor>>,
    value_sequences: Vec<Vec<Option<Tensor>>>,
    n_backward: usize,
}

impl REINFORCE {
    pub fn new(
        model: Box<dyn BasePolicy>,
        optimizer: nn::Optimizer,
        gamma: f64,
        beta: f64,
        batchsize: usize,
        act_deterministically: bool,
        average_entropy_decay: f64,
        backward_separately: bool,
    ) -> Self {
        REINFORCE {
            model,
            optimizer,
            gamma,
            beta,
            batchsize,
            act_deterministically,
            average_entropy_decay,
            backward_separately,
            average_entropy: 0.0,
            t: 0,
            reward_sequences: vec![vec![]],
            log_prob_sequences: vec![vec![]],
            entropy_sequences: vec![vec![]],
            value_sequences: vec![vec![]],
            n_backward: 0,
        }
    }

    fn accumulate_grad(&mut self) {
        if self.n_backward == 0 {
            self.optimizer.zero_grad();
        }
    
        let mut losses = vec![];
    
        // Iterate over the reward, log probabilities, and entropy sequences
        for (r_seq, log_prob_seq, ent_seq, v_seq) in self.reward_sequences.iter()
            .zip(self.log_prob_sequences.iter())
            .zip(self.entropy_sequences.iter())
            .zip(self.value_sequences.iter())
            .map(|(((r, log_prob), ent), v)| (r, log_prob, ent, v)) {
            
            assert_eq!(r_seq.len() - 1, log_prob_seq.len());
            assert_eq!(log_prob_seq.len(), ent_seq.len());
    
            // Calculate returns (sum of future rewards)
            let g_seq = cumsum::cumsum_rev(&r_seq[1..], self.gamma);

            assert_eq!(g_seq.len(), log_prob_seq.len());
    
            // Compute the losses based on rewards, log probabilities, and entropy
            for (((g, log_prob), entropy), v) in g_seq.iter().zip(log_prob_seq.iter()).zip(ent_seq.iter()).zip(v_seq.iter()) {
                let loss;
                let g_tensor = tch::no_grad(|| Tensor::from(*g));
                if v.is_none() {
                    loss = (-g_tensor * log_prob - self.beta * entropy) / r_seq.len() as f64;
                } else {
                    let v_tensor = v.as_ref().unwrap().copy();
                    let advantage = &g_tensor - v_tensor.detach();
                    let actor_loss = -advantage * log_prob - self.beta * entropy;
                    let critic_loss = (v_tensor - &g_tensor).pow_tensor_scalar(2.0);
                    loss = (actor_loss + critic_loss) / r_seq.len() as f64;
                }
                losses.push(loss)
            }
        }
    
        // Sum the losses and divide by batch size, then backward.
        // TODO:don't user into_iter().sum when tensor is on GPU.
        (losses.into_iter().sum::<Tensor>() / self.batchsize as f64).squeeze().backward();
    
        // Reset the reward, log probability, and entropy sequences for the next episode
        self.reward_sequences = vec![vec![]];
        self.log_prob_sequences = vec![vec![]];
        self.entropy_sequences = vec![vec![]];
        self.value_sequences = vec![vec![]];
    
        self.n_backward += 1;
    }

    // Perform the update with the accumulated gradients
    fn update_with_accumulated_grad(&mut self) {
        assert_eq!(self.n_backward, self.batchsize);
        self.optimizer.step();
        self.n_backward = 0;
    }

    // Perform a batch update by accumulating gradients for all episodes in the batch
    fn batch_update(&mut self) {
        assert_eq!(self.reward_sequences.len(), self.batchsize);
        assert_eq!(self.log_prob_sequences.len(), self.batchsize);
        assert_eq!(self.entropy_sequences.len(), self.batchsize);
        assert_eq!(self.value_sequences.len(), self.batchsize);
        assert_eq!(self.n_backward, 0);

        // First accumulate gradients for the batch
        self.accumulate_grad();

        // Perform the optimizer update
        self.optimizer.step();
        self.n_backward = 0;
    }
}

impl BaseAgent for REINFORCE {
    fn act_and_train(&mut self, obs: &Tensor, reward: f64) -> Tensor {
        let state = batch_states(vec![obs.shallow_clone()], self.model.is_cuda());

        // Get action distribution from the model
        let (action_distrib, value) = self.model.forward(&state);

        // Sample an action from the distribution
        let batch_action = action_distrib.sample().detach();
        let action = batch_action.to_device(Device::Cpu);

        // Save values used to compute losses
        if let Some(last_vec) = self.reward_sequences.last_mut(){
            last_vec.push(reward);
        }
        if let Some(last_vec) = self.log_prob_sequences.last_mut(){
            last_vec.push(action_distrib.log_prob(&batch_action));
        }
        if let Some(last_vec) = self.entropy_sequences.last_mut(){
            last_vec.push(action_distrib.entropy());
        }
        if let Some(last_vec) = self.value_sequences.last_mut(){
            last_vec.push(value);
        }

        // Update stats for entropy
        self.average_entropy += (1.0 - self.average_entropy_decay) * (action_distrib.entropy().double_value(&[]) - self.average_entropy);

        // Increment the time step
        self.t += 1;

        // Return the action
        action
    }

    fn act(&self, obs: &Tensor) -> Tensor {
        let mut action: Option<Tensor> = None;
        no_grad(|| {
            let state = batch_states(vec![obs.shallow_clone()], self.model.is_cuda());

            // Get action distribution from the model
            let action_distrib: Box<dyn BaseDistribution> = self.model.forward(&state).0;

            let batch_action;

            if self.act_deterministically {
                // Choose the most probable action
                batch_action = action_distrib.most_probable();
            } else {
                // Sample an action from the distribution
                batch_action = action_distrib.sample();
            }

            action = Some(batch_action.to_device(Device::Cpu));
        });
        action.unwrap()
    }

    fn stop_episode_and_train(&mut self, obs: &Tensor, reward:f64 , done: bool) {
        // Add reward to the sequences
        if let Some(last_vec) = self.reward_sequences.last_mut(){
            last_vec.push(reward);
        }
        
        if done {
            if self.backward_separately {
                // Perform backprop for each episode and accumulate gradients
                self.accumulate_grad();
                if self.n_backward == self.batchsize {
                    self.update_with_accumulated_grad();
                }
            } else {
                if self.reward_sequences.len() == self.batchsize {
                    self.batch_update();
                } else {
                    // Prepare for the next episode
                    self.reward_sequences.push(vec![]);
                    self.log_prob_sequences.push(vec![]);
                    self.entropy_sequences.push(vec![]);
                    self.value_sequences.push(vec![]);
                }
            }
        } else {
            panic!("Since REINFORCE supports episodic environments only, must be done=True.")
        }

        // Reset model state if it's a recurrent model
        self.stop_episode();
    }

    fn stop_episode(&mut self) {
        if self.model.is_recurrent() {
            self.model.reset_state();
        }
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
