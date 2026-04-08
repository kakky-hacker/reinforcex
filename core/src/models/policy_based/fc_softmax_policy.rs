use super::base_policy_network::BasePolicy;

use crate::misc::weight_initializer::he_init;
use crate::prob_distributions::BaseDistribution;
use crate::prob_distributions::SoftmaxDistribution;
use tch::nn::{linear, Init, Linear, LinearConfig, Module, VarStore};
use tch::{nn, Device, Tensor};

pub struct FCSoftmaxPolicy {
    vs: VarStore,
    layers: Vec<Linear>,
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
    pub fn new(
        vs: VarStore,
        n_input_channels: i64,
        n_actions: i64,
        n_hidden_layers: usize,
        n_hidden_channels: i64,
        min_prob: f64,
    ) -> Self {
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

        let logits_layer: Linear = linear(
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
            n_actions,
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
}

impl BasePolicy for FCSoftmaxPolicy {
    fn forward(&self, x: &Tensor) -> (Box<dyn BaseDistribution>, Option<Tensor>) {
        let h = self.compute_medium_layer(x);
        let logits = self.logits_layer.forward(&h).relu();
        (
            Box::new(SoftmaxDistribution::new(logits, 1.0, self.min_prob)),
            None,
        )
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

        let base_policy: FCSoftmaxPolicy = FCSoftmaxPolicy::new(
            vs,
            n_input_channels,
            n_actions,
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
        let logits = self.base_policy.logits_layer.forward(&h);
        let value = self.value_layer.forward(&h);
        (
            Box::new(SoftmaxDistribution::new(
                logits,
                0.1,
                self.base_policy.min_prob,
            )),
            Some(value),
        )
    }

    fn device(&self) -> Device {
        self.base_policy.vs.device()
    }
}
