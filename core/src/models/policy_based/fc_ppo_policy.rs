//! PPO-specific actor/critic networks. Legacy policy and SAC models are unchanged.

use super::base_policy_network::BasePolicy;
use crate::prob_distributions::{BaseDistribution, GaussianDistribution, SoftmaxDistribution};
use std::collections::HashMap;
use std::path::Path;
use tch::nn::{self, Init, Linear, LinearConfig, Module, VarStore};
use tch::{Device, Tensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PpoActivation {
    Tanh,
    Relu,
}

impl PpoActivation {
    fn apply(self, value: Tensor) -> Tensor {
        match self {
            Self::Tanh => value.tanh(),
            Self::Relu => value.relu(),
        }
    }

    fn code(self) -> f64 {
        match self {
            Self::Tanh => 0.0,
            Self::Relu => 1.0,
        }
    }
}

enum ActionSpace {
    Discrete,
    Continuous {
        log_std: Tensor,
        min_action: f64,
        max_action: f64,
        min_variance: f64,
    },
}

/// Independent actor and value MLPs with orthogonal initialization.
///
/// Continuous means are linear/unbounded and variances are state independent,
/// with a separate learned log standard deviation for each action dimension.
/// Only `to_env_action` clips actions; likelihoods use the original Gaussian.
/// As in the legacy models, `hidden_layers` counts additional hidden layers
/// after the input-to-hidden layer, so the actual depth is `hidden_layers + 1`.
pub struct FCPpoPolicy {
    vs: VarStore,
    actor_layers: Vec<Linear>,
    value_layers: Vec<Linear>,
    policy_head: Linear,
    value_head: Linear,
    action_space: ActionSpace,
    activation: PpoActivation,
    obs_size: i64,
    configuration: Tensor,
}

fn initialized_linear(path: nn::Path<'_>, input: i64, output: i64, gain: f64) -> Linear {
    nn::linear(
        path,
        input,
        output,
        LinearConfig {
            ws_init: Init::Orthogonal { gain },
            bs_init: Some(Init::Const(0.0)),
            bias: true,
        },
    )
}

fn hidden_layers(path: nn::Path<'_>, input: i64, additional: usize, width: i64) -> Vec<Linear> {
    (0..=additional)
        .map(|index| {
            initialized_linear(
                &path / format!("hidden_{index}"),
                if index == 0 { input } else { width },
                width,
                2.0_f64.sqrt(),
            )
        })
        .collect()
}

impl FCPpoPolicy {
    pub fn new_discrete(
        vs: VarStore,
        obs_size: i64,
        action_size: i64,
        hidden_layers: usize,
        hidden_size: i64,
        activation: PpoActivation,
    ) -> Self {
        Self::new(
            vs,
            obs_size,
            action_size,
            hidden_layers,
            hidden_size,
            activation,
            None,
        )
    }

    pub fn new_continuous(
        vs: VarStore,
        obs_size: i64,
        action_size: i64,
        hidden_layers: usize,
        hidden_size: i64,
        activation: PpoActivation,
        min_action: f64,
        max_action: f64,
        min_variance: f64,
        initial_log_std: f64,
    ) -> Self {
        assert!(min_action.is_finite() && max_action.is_finite() && min_action < max_action);
        assert!(min_variance.is_finite() && min_variance > 0.0);
        assert!((min_variance as f32).is_finite() && min_variance as f32 > 0.0);
        assert!(initial_log_std.is_finite());
        assert!(
            initial_log_std >= 0.5 * min_variance.ln(),
            "initial standard deviation is below the variance floor"
        );
        let initial_variance = (2.0 * initial_log_std).exp() as f32;
        assert!(initial_variance.is_finite() && initial_variance > 0.0);
        Self::new(
            vs,
            obs_size,
            action_size,
            hidden_layers,
            hidden_size,
            activation,
            Some((min_action, max_action, min_variance, initial_log_std)),
        )
    }

    fn new(
        vs: VarStore,
        obs_size: i64,
        action_size: i64,
        additional_layers: usize,
        width: i64,
        activation: PpoActivation,
        continuous: Option<(f64, f64, f64, f64)>,
    ) -> Self {
        assert!(obs_size > 0 && action_size > 0 && width > 0);
        let root = vs.root();
        let actor_layers = hidden_layers(&root / "actor", obs_size, additional_layers, width);
        let value_layers = hidden_layers(&root / "critic", obs_size, additional_layers, width);
        let policy_head = initialized_linear(&root / "actor" / "output", width, action_size, 0.01);
        let value_head = initialized_linear(&root / "critic" / "output", width, 1, 1.0);
        let action_space = match continuous {
            Some((min_action, max_action, min_variance, initial_log_std)) => {
                ActionSpace::Continuous {
                    log_std: (&root / "actor").var(
                        "log_std",
                        &[action_size],
                        Init::Const(initial_log_std),
                    ),
                    min_action,
                    max_action,
                    min_variance,
                }
            }
            None => ActionSpace::Discrete,
        };
        let (kind, min_action, max_action, min_variance, initial_log_std) = match continuous {
            Some((low, high, minimum, initial)) => (1.0, low, high, minimum, initial),
            None => (0.0, 0.0, 0.0, 0.0, 0.0),
        };
        // Activation/bounds have no learned tensor shape, so persist an explicit
        // nontrainable descriptor and reject incompatible restoration before any
        // weights are changed. This is a new checkpoint format, not a legacy one.
        let configuration = root.add(
            "ppo_configuration",
            Tensor::from_slice(&[
                1.0,
                obs_size as f64,
                action_size as f64,
                additional_layers as f64,
                width as f64,
                activation.code(),
                kind,
                min_action,
                max_action,
                min_variance,
                initial_log_std,
            ])
            .to_device(vs.device()),
            false,
        );
        Self {
            vs,
            actor_layers,
            value_layers,
            policy_head,
            value_head,
            action_space,
            activation,
            obs_size,
            configuration,
        }
    }

    fn hidden(&self, input: &Tensor, layers: &[Linear]) -> Tensor {
        let mut value = input.view([-1, self.obs_size]);
        for layer in layers {
            value = self.activation.apply(layer.forward(&value));
        }
        value
    }
}

impl BasePolicy for FCPpoPolicy {
    fn forward(&self, input: &Tensor) -> (Box<dyn BaseDistribution>, Option<Tensor>) {
        let actor = self.hidden(input, &self.actor_layers);
        let critic = self.hidden(input, &self.value_layers);
        let output = self.policy_head.forward(&actor);
        let value = self.value_head.forward(&critic);
        let distribution: Box<dyn BaseDistribution> = match &self.action_space {
            ActionSpace::Discrete => Box::new(SoftmaxDistribution::new(output, 1.0, 0.0)),
            ActionSpace::Continuous {
                log_std,
                min_action,
                max_action,
                min_variance,
            } => {
                let variance = (log_std.clamp_min(0.5 * min_variance.ln()) * 2.0)
                    .exp()
                    .clamp_min(*min_variance)
                    .expand(output.size(), true);
                Box::new(GaussianDistribution::new_bounded(
                    output,
                    variance,
                    *min_action,
                    *max_action,
                ))
            }
        };
        (distribution, Some(value))
    }

    fn device(&self) -> Device {
        self.vs.device()
    }

    fn save(&self, path: &str) {
        self.vs
            .save(path)
            .unwrap_or_else(|error| panic!("failed to save FCPpoPolicy to {path}: {error}"));
    }

    fn load(&mut self, path: &str) {
        let tensors = match Path::new(path)
            .extension()
            .and_then(|extension| extension.to_str())
        {
            Some("safetensors") => Tensor::read_safetensors(path),
            Some("pt") | Some("bin") => Tensor::loadz_multi_with_device(path, self.device()),
            _ => Tensor::load_multi_with_device(path, self.device()),
        }
        .unwrap_or_else(|error| panic!("failed to inspect FCPpoPolicy checkpoint {path}: {error}"));
        let tensors: HashMap<_, _> = tensors.into_iter().collect();
        let descriptor = tensors
            .get("ppo_configuration")
            .expect("FCPpoPolicy checkpoint has no configuration");
        assert!(
            self.configuration
                .equal(&descriptor.to_device(self.device())),
            "FCPpoPolicy checkpoint configuration mismatch"
        );
        // Validate the complete tensor set before VarStore's sequential copy.
        for (name, destination) in self.vs.variables() {
            let source = tensors
                .get(&name)
                .unwrap_or_else(|| panic!("FCPpoPolicy checkpoint missing {name}"));
            assert_eq!(
                source.size(),
                destination.size(),
                "FCPpoPolicy checkpoint shape mismatch for {name}"
            );
            assert!(
                source.isfinite().all().int64_value(&[]) != 0,
                "FCPpoPolicy checkpoint tensor {name} is nonfinite"
            );
        }
        self.vs
            .load(path)
            .unwrap_or_else(|error| panic!("failed to load FCPpoPolicy from {path}: {error}"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn::OptimizerConfig, no_grad, Kind};

    fn continuous(activation: PpoActivation) -> FCPpoPolicy {
        FCPpoPolicy::new_continuous(
            VarStore::new(Device::Cpu),
            3,
            2,
            1,
            8,
            activation,
            -1.0,
            1.0,
            0.01,
            0.5_f64.ln(),
        )
    }

    fn input() -> Tensor {
        Tensor::from_slice(&[1.0_f32, -2.0, 0.5, -0.5, 1.0, 3.0]).view([2, 3])
    }

    fn copied_parameters(model: &FCPpoPolicy, prefix: &str) -> HashMap<String, Tensor> {
        model
            .vs
            .variables()
            .into_iter()
            .filter(|(name, _)| name.starts_with(prefix))
            .map(|(name, value)| (name, value.copy()))
            .collect()
    }

    fn assert_parameters_unchanged(model: &FCPpoPolicy, expected: &HashMap<String, Tensor>) {
        let actual = model.vs.variables();
        assert!(expected
            .iter()
            .all(|(name, value)| actual[name].equal(value)));
    }

    #[test]
    fn test_actor_and_value_gradients_are_separate() {
        for discrete in [false, true] {
            let model = if discrete {
                FCPpoPolicy::new_discrete(
                    VarStore::new(Device::Cpu),
                    3,
                    2,
                    1,
                    8,
                    PpoActivation::Tanh,
                )
            } else {
                continuous(PpoActivation::Tanh)
            };
            let mut optimizer = nn::Sgd::default().build(&model.vs, 0.01).unwrap();
            let actor_before = copied_parameters(&model, "actor.");
            let (_, value) = model.forward(&input());
            let value_before = value.unwrap().detach().copy();
            let (_, value) = model.forward(&input());
            optimizer.backward_step(
                &(value.unwrap() - (&value_before + 1.0))
                    .square()
                    .mean(Kind::Float),
            );
            assert_parameters_unchanged(&model, &actor_before);
            let (_, value_after) = model.forward(&input());
            assert!(!value_before.equal(&value_after.unwrap()));

            let critic_before = copied_parameters(&model, "critic.");
            let (distribution, _) = model.forward(&input());
            let actions = if discrete {
                Tensor::zeros([2, 1], (Kind::Int64, Device::Cpu))
            } else {
                Tensor::from_slice(&[0.2_f32, 0.8, -0.1, 0.4]).view([2, 2])
            };
            optimizer.backward_step(&(-distribution.log_prob(&actions).mean(Kind::Float)));
            assert_parameters_unchanged(&model, &critic_before);
            let actor_after = model.vs.variables();
            assert!(actor_before
                .iter()
                .any(|(name, value)| !actor_after[name].equal(value)));
        }
    }

    #[test]
    fn test_log_std_is_per_action_and_independent_of_observation_scale() {
        let mut model = continuous(PpoActivation::Relu);
        let ActionSpace::Continuous { log_std, .. } = &mut model.action_space else {
            unreachable!()
        };
        no_grad(|| log_std.copy_(&Tensor::from_slice(&[0.2_f32.ln(), 0.7_f32.ln()])));
        assert_eq!(log_std.size(), [2]);
        let (distribution, _) = model.forward(&input());
        let (_, variance) = distribution.params();
        assert!((variance.double_value(&[0, 0]) - 0.04).abs() < 1e-6);
        assert!((variance.double_value(&[0, 1]) - 0.49).abs() < 1e-6);
        let (scaled, _) = model.forward(&(input() * 1000.0));
        assert!(variance.equal(scaled.params().1));
        let mean = distribution.params().0.detach();
        let offsets = Tensor::from_slice(&[0.1_f32, 2.0]).view([1, 2]);
        (-distribution.log_prob(&(mean + offsets)).mean(Kind::Float)).backward();
        let ActionSpace::Continuous { log_std, .. } = &model.action_space else {
            unreachable!()
        };
        let gradient = log_std.grad();
        assert!(gradient.isfinite().all().int64_value(&[]) != 0);
        assert!(gradient.abs().min().double_value(&[]) > 0.0);
        assert_ne!(gradient.double_value(&[0]), gradient.double_value(&[1]));
    }

    #[test]
    fn test_continuous_mean_is_unbounded_and_only_environment_action_is_clipped() {
        let mut model = continuous(PpoActivation::Tanh);
        no_grad(|| {
            let _ = model.policy_head.ws.zero_();
            model
                .policy_head
                .bs
                .as_mut()
                .unwrap()
                .copy_(&Tensor::from_slice(&[-3.0_f32, 4.0]));
            let ActionSpace::Continuous { log_std, .. } = &mut model.action_space else {
                unreachable!()
            };
            let _ = log_std.fill_(-100.0);
        });
        let (distribution, _) = model.forward(&input());
        let (mean, variance) = distribution.params();
        assert_eq!(mean.double_value(&[0, 0]), -3.0);
        assert_eq!(mean.double_value(&[0, 1]), 4.0);
        assert!(variance.ge(0.01).all().int64_value(&[]) != 0);
        let environment = distribution.to_env_action(mean);
        assert_eq!(environment.double_value(&[0, 0]), -1.0);
        assert_eq!(environment.double_value(&[0, 1]), 1.0);
        assert!(!distribution
            .log_prob(mean)
            .equal(&distribution.log_prob(&environment)));
    }

    #[test]
    fn test_orthogonal_gains_and_zero_biases() {
        let model = continuous(PpoActivation::Tanh);
        let check = |linear: &Linear, gain: f64| {
            let size = linear.ws.size();
            let gram = if size[0] <= size[1] {
                linear.ws.matmul(&linear.ws.transpose(0, 1))
            } else {
                linear.ws.transpose(0, 1).matmul(&linear.ws)
            };
            let expected =
                Tensor::eye(size[0].min(size[1]), (Kind::Float, Device::Cpu)) * gain.powi(2);
            assert!(gram.allclose(&expected, 1e-5, 1e-6, false));
            assert_eq!(
                linear.bs.as_ref().unwrap().abs().max().double_value(&[]),
                0.0
            );
        };
        for layer in model.actor_layers.iter().chain(&model.value_layers) {
            check(layer, 2.0_f64.sqrt());
        }
        check(&model.policy_head, 0.01);
        check(&model.value_head, 1.0);
    }

    #[test]
    fn test_checkpoint_round_trip_and_configuration_mismatch() {
        for discrete in [false, true] {
            let make = || {
                if discrete {
                    FCPpoPolicy::new_discrete(
                        VarStore::new(Device::Cpu),
                        3,
                        2,
                        1,
                        8,
                        PpoActivation::Tanh,
                    )
                } else {
                    continuous(PpoActivation::Tanh)
                }
            };
            let model = make();
            let mut optimizer = nn::Sgd::default().build(&model.vs, 0.1).unwrap();
            let (_, value) = model.forward(&input());
            optimizer.backward_step(&(value.unwrap() - 1.0).square().mean(Kind::Float));
            let path =
                std::env::temp_dir().join(format!("reinforcex-ppo-model-{}.ot", ulid::Ulid::new()));
            let path_string = path.to_str().unwrap();
            model.save(path_string);
            let mut restored = make();
            restored.load(path_string);
            let (original_distribution, original_value) = model.forward(&input());
            let (restored_distribution, restored_value) = restored.forward(&input());
            assert!(original_distribution
                .params()
                .0
                .equal(restored_distribution.params().0));
            assert!(original_distribution
                .params()
                .1
                .equal(restored_distribution.params().1));
            assert!(original_value.unwrap().equal(&restored_value.unwrap()));

            let mut mismatched = if discrete {
                FCPpoPolicy::new_discrete(
                    VarStore::new(Device::Cpu),
                    3,
                    2,
                    1,
                    8,
                    PpoActivation::Relu,
                )
            } else {
                continuous(PpoActivation::Relu)
            };
            let before = copied_parameters(&mismatched, "");
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || mismatched.load(path_string)
            ))
            .is_err());
            assert_parameters_unchanged(&mismatched, &before);
            std::fs::remove_file(path).unwrap();
        }
    }
}
