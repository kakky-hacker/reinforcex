use super::base_policy_network::BasePolicy;

use crate::misc::weight_initializer::{he_init, xavier_init};
use crate::prob_distributions::BaseDistribution;
use crate::prob_distributions::GaussianDistribution;
use tch::nn::{linear, Init, Linear, LinearConfig, Module, VarStore};
use tch::{nn, Device, Tensor};

pub struct FCGaussianPolicy {
    vs: VarStore,
    layers: Vec<Linear>,
    mean_layer: Linear,
    var_layer: Linear,
    n_input_channels: i64,
    bound_mean: bool,
    min_action: Option<f64>,
    max_action: Option<f64>,
    min_var: f64,
}

pub struct FCGaussianPolicyWithValue {
    base_policy: FCGaussianPolicy,
    value_layer: Linear,
}

impl FCGaussianPolicy {
    pub fn new(
        vs: VarStore,
        n_input_channels: i64,
        action_size: i64,
        n_hidden_layers: usize,
        n_hidden_channels: i64,
        min_action: Option<f64>,
        max_action: Option<f64>,
        bound_mean: bool,
        var_type: &str,
        min_var: f64,
    ) -> Self {
        let root = (&vs).root();
        let mut layers: Vec<Linear> = Vec::new();

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

        let mean_layer = linear(
            &root,
            n_hidden_channels,
            action_size,
            LinearConfig {
                ws_init: xavier_init(n_hidden_channels, action_size),
                bs_init: Some(Init::Const(0.0)),
                bias: true,
            },
        );
        let var_size = if var_type == "spherical" {
            1
        } else {
            action_size
        };
        let var_layer = linear(
            &root,
            n_hidden_channels,
            var_size,
            LinearConfig {
                ws_init: he_init(n_hidden_channels),
                bs_init: Some(Init::Const(0.0)),
                bias: true,
            },
        );

        FCGaussianPolicy {
            vs,
            layers,
            mean_layer,
            var_layer,
            n_input_channels,
            bound_mean,
            min_action,
            max_action,
            min_var,
        }
    }

    fn compute_medium_layer(&self, x: &Tensor) -> Tensor {
        let mut h = x.view([-1, self.n_input_channels]);

        for layer in &self.layers {
            h = (layer.forward(&h)).relu();
        }

        h
    }

    fn compute_mean_and_var(&self, x: &Tensor) -> (Tensor, Tensor) {
        let mean = self.mean_layer.forward(&x);
        let mean = if self.bound_mean {
            self.bound_by_tanh(mean)
        } else {
            mean
        };

        let var = self.var_layer.forward(&x).softplus() + self.min_var;
        let var = var.expand(&mean.size(), false);
        (mean, var)
    }

    fn bound_by_tanh(&self, x: Tensor) -> Tensor {
        if self.min_action.is_none() || self.max_action.is_none() {
            return x;
        }
        let min_action = self.min_action.as_ref().unwrap();
        let max_action = self.max_action.as_ref().unwrap();
        let scale = (max_action - min_action) / 2.0;
        let x_mean = (max_action + min_action) / 2.0;
        x.tanh() * scale + x_mean
    }
}

impl BasePolicy for FCGaussianPolicy {
    fn forward(&self, x: &Tensor) -> (Box<dyn BaseDistribution>, Option<Tensor>) {
        let h = self.compute_medium_layer(x);
        let (mean, var) = self.compute_mean_and_var(&h);
        (Box::new(GaussianDistribution::new(mean, var)), None)
    }

    fn device(&self) -> Device {
        self.vs.device()
    }
}

impl FCGaussianPolicyWithValue {
    pub fn new(
        vs: VarStore,
        n_input_channels: i64,
        action_size: i64,
        n_hidden_layers: usize,
        n_hidden_channels: i64,
        min_action: Option<f64>,
        max_action: Option<f64>,
        bound_mean: bool,
        var_type: &str,
        min_var: f64,
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

        let base_policy = FCGaussianPolicy::new(
            vs,
            n_input_channels,
            action_size,
            n_hidden_layers,
            n_hidden_channels,
            min_action,
            max_action,
            bound_mean,
            var_type,
            min_var,
        );

        FCGaussianPolicyWithValue {
            base_policy,
            value_layer,
        }
    }

    fn compute_value(&self, x: &Tensor) -> Tensor {
        self.value_layer.forward(&x)
    }
}

impl BasePolicy for FCGaussianPolicyWithValue {
    fn forward(&self, x: &Tensor) -> (Box<dyn BaseDistribution>, Option<Tensor>) {
        let h = self.base_policy.compute_medium_layer(x);
        let (mean, var) = self.base_policy.compute_mean_and_var(&h);
        let value = self.compute_value(&h);
        (Box::new(GaussianDistribution::new(mean, var)), Some(value))
    }

    fn device(&self) -> Device {
        self.base_policy.vs.device()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn, Device, Tensor};

    #[test]
    fn test_initialization() {
        let vs = nn::VarStore::new(Device::Cpu);
        let n_input_channels = 4;
        let action_size = 2;
        let n_hidden_layers = 2;
        let n_hidden_channels = 64;
        let min_action = -1.0;
        let max_action = 1.0;
        let bound_mean = true;
        let var_type = "spherical";
        let min_var = 1e-3;

        let policy = FCGaussianPolicy::new(
            vs,
            n_input_channels,
            action_size,
            n_hidden_layers,
            n_hidden_channels,
            Some(min_action),
            Some(max_action),
            bound_mean,
            var_type,
            min_var,
        );

        assert_eq!(policy.n_input_channels, n_input_channels);
        assert_eq!(policy.layers.len(), n_hidden_layers + 1);
        assert_eq!(policy.bound_mean, bound_mean);
        assert!(policy.min_action.is_some());
        assert!(policy.max_action.is_some());
        assert_eq!(policy.min_var, min_var);
    }

    #[test]
    fn test_compute_mean_and_var() {
        let vs = nn::VarStore::new(Device::Cpu);
        let n_input_channels = 4;
        let action_size = 6;
        let policy = FCGaussianPolicy::new(
            vs,
            n_input_channels,
            action_size,
            2,
            64,
            None,
            None,
            false,
            "spherical",
            1e-3,
        );

        let input = Tensor::randn(&[3, n_input_channels], (tch::Kind::Float, Device::Cpu));
        let h = policy.compute_medium_layer(&input);
        let (mean, var) = policy.compute_mean_and_var(&h);

        assert_eq!(mean.size()[0], 3);
        assert_eq!(mean.size()[1], action_size);
        assert_eq!(var.size()[0], 3);
        assert_eq!(var.size()[1], action_size);
        assert!(var.min().double_value(&[]) >= 1e-3);
    }

    #[test]
    fn test_bound_by_tanh() {
        let vs = nn::VarStore::new(Device::Cpu);
        let n_input_channels = 4;
        let action_size = 2;
        let min_action = -1.0;
        let max_action = 1.0;

        let policy = FCGaussianPolicy::new(
            vs,
            n_input_channels,
            action_size,
            2,
            64,
            Some(min_action),
            Some(max_action),
            true,
            "spherical",
            1e-3,
        );

        let unbounded_mean = Tensor::from_slice(&[-2.0, 0.0, 2.0]);
        let bounded_mean = policy.bound_by_tanh(unbounded_mean);

        assert!(bounded_mean.min().double_value(&[]) >= min_action);
        assert!(bounded_mean.max().double_value(&[]) <= max_action);
    }

    #[test]
    fn test_forward() {
        let vs = nn::VarStore::new(Device::Cpu);
        let n_input_channels = 4;
        let action_size = 6;
        let policy = FCGaussianPolicy::new(
            vs,
            n_input_channels,
            action_size,
            2,
            64,
            None,
            None,
            false,
            "spherical",
            1e-3,
        );

        let input = Tensor::randn(&[3, n_input_channels], (tch::Kind::Float, Device::Cpu));
        let action_distribution = policy.forward(&input).0;

        let (mean, var) = action_distribution.params();

        assert!(mean.size() == vec![3, 6]);
        assert!(var.size() == vec![3, 6]);
    }
}
