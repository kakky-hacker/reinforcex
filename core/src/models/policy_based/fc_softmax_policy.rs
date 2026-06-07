use super::base_policy_network::BasePolicy;

use crate::misc::weight_initializer::he_init;
use crate::prob_distributions::BaseDistribution;
use crate::prob_distributions::MultiSoftmaxDistribution;
use crate::prob_distributions::SoftmaxDistribution;
use tch::nn::{linear, Init, Linear, LinearConfig, Module, VarStore};
use tch::{Device, Tensor};

pub struct FCSoftmaxPolicy {
    vs: VarStore,
    layers: Vec<Linear>,
    logits_layer: Linear,
    n_input_channels: i64,
    action_branch_sizes: Vec<i64>,
    min_prob: f64,
}

pub struct FCSoftmaxPolicyWithValue {
    base_policy: FCSoftmaxPolicy,
    value_layer: Linear,
}

impl FCSoftmaxPolicy {
    pub fn new(
        vs: VarStore,
        n_input_channels: i64,
        n_actions: i64,
        n_hidden_layers: usize,
        n_hidden_channels: i64,
        min_prob: f64,
    ) -> Self {
        Self::new_multi(
            vs,
            n_input_channels,
            vec![n_actions],
            n_hidden_layers,
            n_hidden_channels,
            min_prob,
        )
    }

    pub fn new_multi(
        vs: VarStore,
        n_input_channels: i64,
        action_branch_sizes: Vec<i64>,
        n_hidden_layers: usize,
        n_hidden_channels: i64,
        min_prob: f64,
    ) -> Self {
        assert!(!action_branch_sizes.is_empty());
        assert!(action_branch_sizes.iter().all(|&n_actions| n_actions > 0));

        let root = (&vs).root();
        let mut layers = Vec::new();

        layers.push(linear(
            &root,
            n_input_channels,
            n_hidden_channels,
            LinearConfig {
                ws_init: he_init(n_input_channels),
                bs_init: Some(Init::Const(0.0)),
                bias: true,
            },
        ));
        for _ in 0..n_hidden_layers {
            layers.push(linear(
                &root,
                n_hidden_channels,
                n_hidden_channels,
                LinearConfig {
                    ws_init: he_init(n_hidden_channels),
                    bs_init: Some(Init::Const(0.0)),
                    bias: true,
                },
            ));
        }

        let n_actions = action_branch_sizes.iter().sum();
        let logits_layer = linear(
            &root,
            n_hidden_channels,
            n_actions,
            LinearConfig {
                ws_init: he_init(n_hidden_channels),
                bs_init: Some(Init::Const(0.0)),
                bias: true,
            },
        );

        FCSoftmaxPolicy {
            vs,
            layers,
            logits_layer,
            n_input_channels,
            action_branch_sizes,
            min_prob,
        }
    }

    fn compute_medium_layer(&self, x: &Tensor) -> Tensor {
        let mut h = x.view([-1, self.n_input_channels]);

        for layer in &self.layers {
            h = (layer.forward(&h)).relu();
        }

        h
    }

    fn compute_distribution(
        &self,
        h: &Tensor,
        beta: f64,
        relu_logits: bool,
    ) -> Box<dyn BaseDistribution> {
        let logits = self.logits_layer.forward(h);
        let logits = if relu_logits { logits.relu() } else { logits };

        if self.action_branch_sizes.len() == 1 {
            Box::new(SoftmaxDistribution::new(logits, beta, self.min_prob))
        } else {
            let mut branch_start = 0;
            let distributions = self
                .action_branch_sizes
                .iter()
                .map(|&branch_size| {
                    let branch_logits = logits.narrow(1, branch_start, branch_size);
                    branch_start += branch_size;
                    SoftmaxDistribution::new(branch_logits, beta, self.min_prob)
                })
                .collect();
            Box::new(MultiSoftmaxDistribution::new(distributions))
        }
    }
}

impl BasePolicy for FCSoftmaxPolicy {
    fn forward(&self, x: &Tensor) -> (Box<dyn BaseDistribution>, Option<Tensor>) {
        let h = self.compute_medium_layer(x);
        (self.compute_distribution(&h, 1.0, true), None)
    }

    fn device(&self) -> Device {
        self.vs.device()
    }
}

impl FCSoftmaxPolicyWithValue {
    pub fn new(
        vs: VarStore,
        n_input_channels: i64,
        n_actions: i64,
        n_hidden_layers: usize,
        n_hidden_channels: i64,
        min_prob: f64,
    ) -> Self {
        Self::new_multi(
            vs,
            n_input_channels,
            vec![n_actions],
            n_hidden_layers,
            n_hidden_channels,
            min_prob,
        )
    }

    pub fn new_multi(
        vs: VarStore,
        n_input_channels: i64,
        action_branch_sizes: Vec<i64>,
        n_hidden_layers: usize,
        n_hidden_channels: i64,
        min_prob: f64,
    ) -> Self {
        let root = (&vs).root();
        let value_layer = linear(
            &root,
            n_hidden_channels,
            1,
            LinearConfig {
                ws_init: he_init(n_hidden_channels),
                bs_init: Some(Init::Const(0.0)),
                bias: true,
            },
        );

        let base_policy: FCSoftmaxPolicy = FCSoftmaxPolicy::new_multi(
            vs,
            n_input_channels,
            action_branch_sizes,
            n_hidden_layers,
            n_hidden_channels,
            min_prob,
        );

        FCSoftmaxPolicyWithValue {
            base_policy,
            value_layer,
        }
    }
}

impl BasePolicy for FCSoftmaxPolicyWithValue {
    fn forward(&self, x: &Tensor) -> (Box<dyn BaseDistribution>, Option<Tensor>) {
        let h = self.base_policy.compute_medium_layer(x);
        let value = self.value_layer.forward(&h);
        (
            self.base_policy.compute_distribution(&h, 0.1, false),
            Some(value),
        )
    }

    fn device(&self) -> Device {
        self.base_policy.vs.device()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn, Device, Kind, Tensor};

    #[test]
    fn test_multi_softmax_policy_forward() {
        let vs = nn::VarStore::new(Device::Cpu);
        let policy = FCSoftmaxPolicyWithValue::new_multi(vs, 4, vec![3, 2], 2, 16, 0.0);
        let input = Tensor::randn(&[100, 4], (Kind::Float, Device::Cpu));

        let (dist, value) = policy.forward(&input);
        let value = value.unwrap();
        let action = dist.sample();
        let log_prob = dist.log_prob(&action);
        let entropy = dist.entropy();
        let most_probable = dist.most_probable();

        assert_eq!(value.size(), [100, 1]);
        assert_eq!(action.size(), [100, 2]);
        assert_eq!(log_prob.size(), [100]);
        assert_eq!(entropy.size(), [100]);
        assert_eq!(most_probable.size(), [100, 2]);

        for batch in 0..100 {
            let branch0 = action.int64_value(&[batch, 0]);
            let branch1 = action.int64_value(&[batch, 1]);
            assert!(0 <= branch0 && branch0 < 3);
            assert!(0 <= branch1 && branch1 < 2);
        }
    }
}
