use super::base_policy_network::BasePolicy;

use tch::{Tensor, nn};
use tch::nn::{Module, Linear, LinearConfig, Init};
use crate::prob_distributions::BaseDistribution;
use crate::prob_distributions::SoftmaxDistribution;
use crate::misc::weight_initializer::he_init;

pub struct FCSoftmaxPolicy {
    hidden_layers: Vec<Linear>,
    logits_layer: Linear,
    n_input_channels: i64,
    n_actions: i64,
    min_prob: f64,
}

pub struct FCSoftmaxPolicyWithValue {
    base_policy: FCSoftmaxPolicy,
    value_layer: Linear,
}


impl FCSoftmaxPolicy {
    pub fn new(vs: &nn::VarStore, n_input_channels: i64, n_actions: i64, 
           n_hidden_layers: usize, n_hidden_channels: Option<i64>, 
           min_prob: f64) -> Self {

        let root: nn::Path<'_> = vs.root(); 
        let mut hidden_layers: Vec<Linear> = Vec::new();
        let n_hidden_channels: i64 = n_hidden_channels.unwrap_or(256);

        if n_hidden_layers > 0 {
            hidden_layers.push(nn::linear(&root, n_input_channels, n_hidden_channels, LinearConfig {
                ws_init: he_init(n_input_channels),
                bs_init: Some(Init::Const(0.0)),
                bias: true,
            }));
            for _ in 0..n_hidden_layers - 1 {
                hidden_layers.push(nn::linear(&root, n_hidden_channels, n_hidden_channels, LinearConfig {
                    ws_init: he_init(n_hidden_channels),
                    bs_init: Some(Init::Const(0.0)),
                    bias: true,
                }));
            }
        }

        let logits_layer: Linear = nn::linear(&root, n_hidden_channels, n_actions, LinearConfig {
            ws_init: he_init(n_hidden_channels),
            bs_init: Some(Init::Const(0.0)),
            bias: true,
        });
        
        FCSoftmaxPolicy {
            hidden_layers,
            logits_layer,
            n_input_channels,
            n_actions,
            min_prob,
        }
    }

    fn compute_medium_layer(&self, x: &Tensor) -> Tensor {
        let mut h: Tensor = x.view([-1, self.n_input_channels]);
        
        for layer in &self.hidden_layers {
            h = (layer.forward(&h)).relu();
        }

        h
    }
}

impl BasePolicy for FCSoftmaxPolicy {
    fn forward(&self, x: &Tensor) -> (Box<dyn BaseDistribution>, Option<Tensor>) {
        let h: Tensor = self.compute_medium_layer(x);
        let logits: Tensor = self.logits_layer.forward(&h).relu();
        (Box::new(SoftmaxDistribution::new(logits, 1.0, self.min_prob)), None)
    }

    fn is_recurrent(&self) -> bool{
        false
    }

    fn reset_state(&mut self) {
        
    }

    fn is_cuda(&self) -> bool {
        false
    }
}

impl FCSoftmaxPolicyWithValue {
    pub fn new(vs: &nn::VarStore, n_input_channels: i64, n_actions: i64, 
        n_hidden_layers: usize, n_hidden_channels: Option<i64>, 
        min_prob: f64) -> Self {
        let base_policy: FCSoftmaxPolicy = FCSoftmaxPolicy::new(
            vs,
            n_input_channels,
            n_actions,
            n_hidden_layers,
            n_hidden_channels,
            min_prob,
        );

        let root: nn::Path<'_> = vs.root();
        let n_hidden_channels: i64 = n_hidden_channels.unwrap_or(256);
        let value_layer: Linear = nn::linear(
            &root,
            n_hidden_channels,
            1,
            LinearConfig {
                ws_init: he_init(n_hidden_channels),
                bs_init: Some(Init::Const(0.0)),
                bias: true,
            },
        );

        FCSoftmaxPolicyWithValue {
            base_policy,
            value_layer,
        }
    }
}

impl BasePolicy for FCSoftmaxPolicyWithValue {
    fn forward(&self, x: &Tensor) -> (Box<dyn BaseDistribution>, Option<Tensor>) {
        let h: Tensor = self.base_policy.compute_medium_layer(x);
        let logits: Tensor = self.base_policy.logits_layer.forward(&h).relu();
        let value: Tensor = self.value_layer.forward(&h);
        (
            Box::new(SoftmaxDistribution::new(logits, 1.0, self.base_policy.min_prob)),
            Some(value),
        )
    }

    fn is_recurrent(&self) -> bool {
        self.base_policy.is_recurrent()
    }

    fn reset_state(&mut self) {
        self.base_policy.reset_state();
    }

    fn is_cuda(&self) -> bool {
        self.base_policy.is_cuda()
    }
}
