use super::base_policy_network::BasePolicy;

use tch::{Tensor, nn};
use tch::nn::{Module, Linear, LinearConfig, Init};
use crate::prob_distributions::BaseDistribution;
use crate::prob_distributions::GaussianDistribution;
use crate::misc::weight_initializer::{xavier_init, he_init};

pub struct FCGaussianPolicy {
    hidden_layers: Vec<Linear>,
    mean_layer: Linear,
    var_layer: Linear,
    n_input_channels: i64,
    bound_mean: bool,
    min_action: Option<Tensor>,
    max_action: Option<Tensor>,
    min_var: f64,
}

pub struct FCGaussianPolicyWithValue {
    base_policy: FCGaussianPolicy,
    value_layer: Linear,
}


impl FCGaussianPolicy {
    pub fn new(vs: &nn::VarStore, n_input_channels: i64, action_size: i64, 
           n_hidden_layers: usize, n_hidden_channels: Option<i64>, 
           min_action: Option<Tensor>, max_action: Option<Tensor>, 
           bound_mean: bool, var_type: &str, min_var: f64) -> Self {

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

        let mean_layer: Linear = nn::linear(&root, n_hidden_channels, action_size, LinearConfig {
            ws_init: xavier_init(n_hidden_channels, action_size),
            bs_init: Some(Init::Const(0.0)),
            bias: true,
        });
        let var_size: i64 = if var_type == "spherical" { 1 } else { action_size };
        let var_layer: Linear = nn::linear(&root, n_hidden_channels, var_size, LinearConfig {
            ws_init: he_init(n_hidden_channels),
            bs_init: Some(Init::Const(0.0)),
            bias: true,
        });
        
        FCGaussianPolicy {
            hidden_layers,
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
        let mut h: Tensor = x.view([-1, self.n_input_channels]);
        
        for layer in &self.hidden_layers {
            h = (layer.forward(&h)).relu();
        }

        h
    }

    fn compute_mean_and_var(&self, x: &Tensor) -> (Tensor, Tensor) {
        let mean: Tensor = self.mean_layer.forward(&x);
        let mean: Tensor = if self.bound_mean {
            self.bound_by_tanh(mean)
        } else {
            mean
        };

        let var: Tensor = self.var_layer.forward(&x).softplus() + self.min_var;
        let var: Tensor = var.expand(&mean.size(), false);
        (mean, var)
    }

    fn bound_by_tanh(&self, x: Tensor) -> Tensor {
        if self.min_action.is_none() || self.max_action.is_none() {
            return x;
        }
        let min_action: &Tensor = self.min_action.as_ref().unwrap();
        let max_action: &Tensor = self.max_action.as_ref().unwrap();
        let scale: Tensor = (max_action - min_action) / 2.0;
        let x_mean: Tensor = (max_action + min_action) / 2.0;
        x.tanh() * scale + x_mean
    }
}

impl BasePolicy for FCGaussianPolicy {
    fn forward(&self, x: &Tensor) -> (Box<dyn BaseDistribution>, Option<Tensor>) {
        let h: Tensor = self.compute_medium_layer(x);
        let (mean, var) = self.compute_mean_and_var(&h);
        (Box::new(GaussianDistribution::new(mean, var)), None)
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

impl FCGaussianPolicyWithValue {
    pub fn new(
        vs: &nn::VarStore,
        n_input_channels: i64,
        action_size: i64,
        n_hidden_layers: usize,
        n_hidden_channels: Option<i64>,
        min_action: Option<Tensor>,
        max_action: Option<Tensor>,
        bound_mean: bool,
        var_type: &str,
        min_var: f64,
    ) -> Self {
        let base_policy: FCGaussianPolicy = FCGaussianPolicy::new(
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
        let h: Tensor = self.base_policy.compute_medium_layer(x);
        let (mean, var) = self.base_policy.compute_mean_and_var(&h);
        let value: Tensor = self.compute_value(&h);
        (
            Box::new(GaussianDistribution::new(mean, var)),
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


#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn, Tensor, Device};

    #[test]
    fn test_initialization() {
        let vs: nn::VarStore = nn::VarStore::new(Device::Cpu);
        let n_input_channels: i64 = 4;
        let action_size: i64 = 2;
        let n_hidden_layers: usize = 2;
        let n_hidden_channels: Option<i64> = Some(64);
        let min_action: Tensor = Tensor::from(-1.0).copy();
        let max_action: Tensor = Tensor::from(1.0).copy();
        let bound_mean: bool = true;
        let var_type: &str = "spherical";
        let min_var: f64 = 1e-3;

        let policy: FCGaussianPolicy = FCGaussianPolicy::new(
            &vs, n_input_channels, action_size, n_hidden_layers, n_hidden_channels,
            Some(min_action.shallow_clone()), Some(max_action.shallow_clone()), bound_mean, var_type, min_var
        );

        assert_eq!(policy.n_input_channels, n_input_channels);
        assert_eq!(policy.hidden_layers.len(), n_hidden_layers);
        assert_eq!(policy.bound_mean, bound_mean);
        assert!(policy.min_action.is_some());
        assert!(policy.max_action.is_some());
        assert_eq!(policy.min_var, min_var);
    }

    #[test]
    fn test_compute_mean_and_var() {
        let vs: nn::VarStore = nn::VarStore::new(Device::Cpu);
        let n_input_channels: i64 = 4;
        let action_size: i64 = 6;
        let policy: FCGaussianPolicy = FCGaussianPolicy::new(
            &vs, n_input_channels, action_size, 2, Some(64), None, None,
            false, "spherical", 1e-3,
        );

        let input: Tensor = Tensor::randn(&[3, n_input_channels], (tch::Kind::Float, Device::Cpu));
        let h: Tensor = policy.compute_medium_layer(&input);
        let (mean, var) = policy.compute_mean_and_var(&h);

        assert_eq!(mean.size()[0], 3);
        assert_eq!(mean.size()[1], action_size);
        assert_eq!(var.size()[0], 3);
        assert_eq!(var.size()[1], action_size);
        assert!(var.min().double_value(&[]) >= 1e-3);
    }

    #[test]
    fn test_bound_by_tanh() {
        let vs: nn::VarStore = nn::VarStore::new(Device::Cpu);
        let n_input_channels: i64 = 4;
        let action_size: i64 = 2;
        let min_action: Tensor = Tensor::from(-1.0);
        let max_action: Tensor = Tensor::from(1.0);

        let policy: FCGaussianPolicy = FCGaussianPolicy::new(
            &vs, n_input_channels, action_size, 2, Some(64), 
            Some(min_action.copy()), Some(max_action.copy()),
            true, "spherical", 1e-3,
        );

        let unbounded_mean: Tensor = Tensor::from_slice(&[-2.0, 0.0, 2.0]);
        let bounded_mean: Tensor = policy.bound_by_tanh(unbounded_mean);

        assert!(bounded_mean.min().double_value(&[]) >= min_action.double_value(&[]));
        assert!(bounded_mean.max().double_value(&[]) <= max_action.double_value(&[]));
    }

    #[test]
    fn test_forward(){
        let vs: nn::VarStore = nn::VarStore::new(Device::Cpu);
        let n_input_channels: i64 = 4;
        let action_size: i64 = 6;
        let policy: FCGaussianPolicy = FCGaussianPolicy::new(
            &vs, n_input_channels, action_size, 2, Some(64), None, None,
            false, "spherical", 1e-3,
        );

        let input: Tensor = Tensor::randn(&[3, n_input_channels], (tch::Kind::Float, Device::Cpu));
        let action_distribution:Box<dyn BaseDistribution>  = policy.forward(&input).0;

        let (mean, var) = action_distribution.params();
        
        assert!(mean.size() == vec![3, 6]);
        assert!(var.size() == vec![3, 6]);
    }
}
