use super::base_policy_network::BasePolicy;

use crate::misc::weight_initializer::he_init;
use crate::prob_distributions::BaseDistribution;
use crate::prob_distributions::SoftmaxDistribution;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Init, Linear, Module, VarBuilder, VarMap};

pub struct FCSoftmaxPolicy {
    layers: Vec<Linear>,
    logits_layer: Linear,
    n_input_channels: usize,
    n_actions: usize,
    min_prob: f64,
}

pub struct FCSoftmaxPolicyWithValue {
    base_policy: FCSoftmaxPolicy,
    value_layer: Linear,
}

impl FCSoftmaxPolicy {
    pub fn new(
        vb: VarBuilder,
        n_input_channels: usize,
        n_actions: usize,
        n_hidden_layers: usize,
        n_hidden_channels: Option<usize>,
        min_prob: f64,
    ) -> Self {
        let mut layers = Vec::new();
        let n_hidden_channels = n_hidden_channels.unwrap_or(256);

        layers.push(Linear::new(
            vb.get_with_hints(
                (n_hidden_channels, n_input_channels),
                "weight_input",
                he_init(n_input_channels),
            )
            .unwrap(),
            Some(
                vb.get_with_hints(1, "bias_input", Init::Const(0.0))
                    .unwrap(),
            ),
        ));
        for i in 0..n_hidden_layers {
            layers.push(Linear::new(
                vb.get_with_hints(
                    (n_hidden_channels, n_hidden_channels),
                    format!("weight_medium_{}", i).as_str(),
                    he_init(n_hidden_channels),
                )
                .unwrap(),
                Some(
                    vb.get_with_hints(1, format!("bias_medium_{}", i).as_str(), Init::Const(0.0))
                        .unwrap(),
                ),
            ));
        }

        let logits_layer: Linear = Linear::new(
            vb.get_with_hints(
                (n_actions, n_hidden_channels),
                "weight_output",
                he_init(n_hidden_channels),
            )
            .unwrap(),
            Some(
                vb.get_with_hints(1, "bias_output", Init::Const(0.0))
                    .unwrap(),
            ),
        );

        FCSoftmaxPolicy {
            layers,
            logits_layer,
            n_input_channels,
            n_actions,
            min_prob,
        }
    }

    fn compute_medium_layer(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.reshape(((), self.n_input_channels))?;
        for i in 0..self.layers.len() {
            h = self.layers[i].forward(&h)?;
            if i < self.layers.len() - 1 {
                h = h.relu()?;
            }
        }
        Ok(h)
    }
}

impl BasePolicy for FCSoftmaxPolicy {
    fn forward(&self, x: &Tensor) -> Result<(Box<dyn BaseDistribution>, Option<Tensor>)> {
        let h = self.compute_medium_layer(x)?;
        let logits = self.logits_layer.forward(&h)?.relu()?;
        Ok((
            Box::new(SoftmaxDistribution::new(logits, 1.0, self.min_prob)),
            None,
        ))
    }

    fn is_cuda(&self) -> bool {
        self.layers[0].weight().device().is_cuda()
    }

    fn get_device(&self) -> &Device {
        self.layers[0].weight().device()
    }
}

impl FCSoftmaxPolicyWithValue {
    pub fn new(
        vb: VarBuilder,
        n_input_channels: usize,
        n_actions: usize,
        n_hidden_layers: usize,
        n_hidden_channels: Option<usize>,
        min_prob: f64,
    ) -> Self {
        let n_hidden_channels = n_hidden_channels.unwrap_or(256);
        let value_layer = Linear::new(
            vb.get_with_hints(
                (n_hidden_channels, 1),
                "weight_value",
                he_init(n_hidden_channels),
            )
            .unwrap(),
            Some(
                vb.get_with_hints(1, "bias_value", Init::Const(0.0))
                    .unwrap(),
            ),
        );
        let base_policy: FCSoftmaxPolicy = FCSoftmaxPolicy::new(
            vb,
            n_input_channels,
            n_actions,
            n_hidden_layers,
            Some(n_hidden_channels),
            min_prob,
        );

        FCSoftmaxPolicyWithValue {
            base_policy,
            value_layer,
        }
    }
}

impl BasePolicy for FCSoftmaxPolicyWithValue {
    fn forward(&self, x: &Tensor) -> Result<(Box<dyn BaseDistribution>, Option<Tensor>)> {
        let h = self.base_policy.compute_medium_layer(x)?;
        let logits = self.base_policy.logits_layer.forward(&h)?.relu()?;
        let value = self.value_layer.forward(&h)?;
        Ok((
            Box::new(SoftmaxDistribution::new(
                logits,
                1.0,
                self.base_policy.min_prob,
            )),
            Some(value),
        ))
    }

    fn is_cuda(&self) -> bool {
        self.base_policy.is_cuda()
    }

    fn get_device(&self) -> &Device {
        self.base_policy.get_device()
    }
}
