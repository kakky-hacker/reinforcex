use super::base_policy_network::BasePolicy;

use crate::misc::weight_initializer::{he_init, xavier_init};
use crate::prob_distributions::BaseDistribution;
use crate::prob_distributions::GaussianDistribution;
use tch::nn::{linear, Init, Linear, LinearConfig, Module, VarStore};
use tch::{no_grad, Device, Tensor};

const PPO_ACTION_MEAN_INIT_STD: f64 = 0.01;

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
    max_variance: Option<f64>,
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
        self.compute_mean_and_var_with_max_variance(x, None)
    }

    fn compute_mean_and_var_with_max_variance(
        &self,
        x: &Tensor,
        max_variance: Option<f64>,
    ) -> (Tensor, Tensor) {
        let mean = self.mean_layer.forward(&x);
        let mean = if self.bound_mean {
            self.bound_by_tanh(mean)
        } else {
            mean
        };

        let raw_var = self.var_layer.forward(&x);
        let var = match max_variance {
            Some(max_var) => raw_var.sigmoid() * (max_var - self.min_var) + self.min_var,
            None => raw_var.softplus() + self.min_var,
        };
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
        let distribution = match (self.min_action, self.max_action) {
            (Some(min_action), Some(max_action)) => {
                GaussianDistribution::new_bounded(mean, var, min_action, max_action)
            }
            _ => GaussianDistribution::new(mean, var),
        };
        (Box::new(distribution), None)
    }

    fn device(&self) -> Device {
        self.vs.device()
    }

    fn save(&self, path: &str) {
        self.vs
            .save(path)
            .unwrap_or_else(|e| panic!("failed to save FCGaussianPolicy to {}: {}", path, e));
    }

    fn load(&mut self, path: &str) {
        self.vs
            .load(path)
            .unwrap_or_else(|e| panic!("failed to load FCGaussianPolicy from {}: {}", path, e));
    }
}

impl FCGaussianPolicyWithValue {
    /// Creates a policy with learnable variance and no upper variance bound.
    /// Use [`Self::with_max_variance`] to opt into a bounded variance parameterization.
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
        assert!(
            min_var.is_finite() && min_var >= 0.0,
            "min_var must be finite and nonnegative"
        );
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

        let mut base_policy = FCGaussianPolicy::new(
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
        no_grad(|| {
            let _ = base_policy
                .mean_layer
                .ws
                .normal_(0.0, PPO_ACTION_MEAN_INIT_STD);
            if let Some(bias) = base_policy.mean_layer.bs.as_mut() {
                let _ = bias.zero_();
            }
        });

        FCGaussianPolicyWithValue {
            base_policy,
            value_layer,
            max_variance: None,
        }
    }

    /// Bounds action variance between `min_var` and `max_variance` using a sigmoid.
    ///
    /// The upper bound must be finite and strictly greater than `min_var` so the
    /// variance head remains trainable. This changes the interpretation of its
    /// weights, so checkpoint loading must use the same bound as training.
    pub fn with_max_variance(mut self, max_variance: f64) -> Self {
        assert!(
            max_variance.is_finite() && max_variance > self.base_policy.min_var,
            "max_variance must be finite and greater than min_var"
        );
        self.max_variance = Some(max_variance);
        self
    }

    fn compute_value(&self, x: &Tensor) -> Tensor {
        self.value_layer.forward(&x)
    }
}

impl BasePolicy for FCGaussianPolicyWithValue {
    fn forward(&self, x: &Tensor) -> (Box<dyn BaseDistribution>, Option<Tensor>) {
        let h = self.base_policy.compute_medium_layer(x);
        let (mean, var) = self
            .base_policy
            .compute_mean_and_var_with_max_variance(&h, self.max_variance);
        let value = self.compute_value(&h);
        let distribution = match (self.base_policy.min_action, self.base_policy.max_action) {
            (Some(min_action), Some(max_action)) => {
                GaussianDistribution::new_bounded(mean, var, min_action, max_action)
            }
            _ => GaussianDistribution::new(mean, var),
        };
        (Box::new(distribution), Some(value))
    }

    fn device(&self) -> Device {
        self.base_policy.vs.device()
    }

    fn save(&self, path: &str) {
        self.base_policy.vs.save(path).unwrap_or_else(|e| {
            panic!(
                "failed to save FCGaussianPolicyWithValue to {}: {}",
                path, e
            )
        });
    }

    fn load(&mut self, path: &str) {
        self.base_policy.vs.load(path).unwrap_or_else(|e| {
            panic!(
                "failed to load FCGaussianPolicyWithValue from {}: {}",
                path, e
            )
        });
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
    fn test_policy_with_value_caps_action_variance() {
        let vs = nn::VarStore::new(Device::Cpu);
        let max_variance = 0.1;
        let mut policy = FCGaussianPolicyWithValue::new(
            vs,
            4,
            2,
            1,
            16,
            Some(-1.0),
            Some(1.0),
            true,
            "spherical",
            1e-3,
        )
        .with_max_variance(max_variance);
        let _ = tch::no_grad(|| {
            policy
                .base_policy
                .var_layer
                .bs
                .as_mut()
                .unwrap()
                .fill_(100.0)
        });

        let (distribution, _) =
            policy.forward(&Tensor::zeros([3, 4], (tch::Kind::Float, Device::Cpu)));
        let (_, variance) = distribution.params();
        assert!(variance.le(max_variance).all().int64_value(&[]) == 1);
        assert!(variance.ge(1e-3).all().int64_value(&[]) == 1);
    }

    fn policy_with_value(min_var: f64) -> FCGaussianPolicyWithValue {
        FCGaussianPolicyWithValue::new(
            nn::VarStore::new(Device::Cpu),
            4,
            2,
            1,
            16,
            Some(-1.0),
            Some(1.0),
            true,
            "spherical",
            min_var,
        )
    }

    fn assert_variance_entropy_gradient(policy: &FCGaussianPolicyWithValue) -> f64 {
        // Zero inputs and zero-initialized biases give a deterministic variance
        // head input, independently of the random weight initialization.
        let (distribution, _) =
            policy.forward(&Tensor::zeros([3, 4], (tch::Kind::Float, Device::Cpu)));
        let (_, variance) = distribution.params();
        let variance_value = variance.double_value(&[0, 0]);
        assert!(variance.isfinite().all().int64_value(&[]) == 1);
        distribution.entropy().mean(tch::Kind::Float).backward();
        let gradient = policy.base_policy.var_layer.bs.as_ref().unwrap().grad();
        assert!(gradient.defined());
        assert!(gradient.isfinite().all().int64_value(&[]) == 1);
        assert!(gradient.abs().min().double_value(&[]) > 0.0);
        variance_value
    }

    #[test]
    fn test_policy_with_value_default_variance_remains_trainable() {
        // In particular, the FFI default minimum of 0.1 and larger minima must
        // not collapse the variance interval and disconnect the variance head.
        for min_variance in [0.01, 0.1, 0.2] {
            let policy = policy_with_value(min_variance);
            let variance = assert_variance_entropy_gradient(&policy);
            assert!(variance > min_variance);
            assert!(variance > 0.1);
        }
    }

    #[test]
    fn test_policy_with_value_explicit_variance_cap_remains_trainable() {
        for min_variance in [0.01, 0.1, 0.2] {
            let max_variance = min_variance + 0.1;
            let policy = policy_with_value(min_variance).with_max_variance(max_variance);
            let variance = assert_variance_entropy_gradient(&policy);
            assert!(variance > min_variance);
            assert!(variance < max_variance);
        }
    }

    #[test]
    fn test_policy_with_value_rejects_invalid_variance_caps() {
        for max_variance in [0.0, 0.05, 0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(std::panic::catch_unwind(|| {
                policy_with_value(0.1).with_max_variance(max_variance)
            })
            .is_err());
        }
    }

    #[test]
    fn test_policy_with_value_rejects_invalid_variance_minima() {
        for min_variance in [-0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(std::panic::catch_unwind(|| policy_with_value(min_variance)).is_err());
        }
    }

    #[test]
    fn test_policy_with_value_starts_with_small_action_means() {
        let vs = nn::VarStore::new(Device::Cpu);
        let policy = FCGaussianPolicyWithValue::new(
            vs,
            17,
            6,
            1,
            256,
            Some(-1.0),
            Some(1.0),
            true,
            "diagonal",
            1e-2,
        );

        let weight_std = policy
            .base_policy
            .mean_layer
            .ws
            .std(false)
            .double_value(&[]);
        assert!(weight_std > 0.005);
        assert!(weight_std < 0.02);
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
