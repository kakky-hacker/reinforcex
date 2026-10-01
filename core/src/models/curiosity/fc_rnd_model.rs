use super::base_curiosity_model::BasecuriosityModel;
use crate::misc::weight_initializer::{he_init, xavier_init};
use std::fs;
use std::path::Path as StdPath;
use tch::nn::{linear, Init, Linear, LinearConfig, Module, Path, VarStore};
use crate::misc::autograd::no_grad;
use tch::{Device, Kind, Tensor};

pub struct FCRNDModel {
    predictor_vs: VarStore,
    target_vs: VarStore,
    predictor_layers: Vec<Linear>,
    target_layers: Vec<Linear>,
    n_input_channels: i64,
    feature_size: i64,
}

impl FCRNDModel {
    pub fn new(
        predictor_vs: VarStore,
        mut target_vs: VarStore,
        n_input_channels: i64,
        feature_size: i64,
        n_hidden_layers: usize,
        n_hidden_channels: i64,
    ) -> Self {
        assert!(n_input_channels > 0);
        assert!(feature_size > 0);
        assert!(n_hidden_channels > 0);
        assert_eq!(predictor_vs.device(), target_vs.device());

        let predictor_layers = {
            let root = predictor_vs.root();
            Self::build_layers(
                &root,
                n_input_channels,
                feature_size,
                n_hidden_layers,
                n_hidden_channels,
            )
        };
        let target_layers = {
            let root = target_vs.root();
            Self::build_layers(
                &root,
                n_input_channels,
                feature_size,
                n_hidden_layers,
                n_hidden_channels,
            )
        };
        target_vs.freeze();

        FCRNDModel {
            predictor_vs,
            target_vs,
            predictor_layers,
            target_layers,
            n_input_channels,
            feature_size,
        }
    }

    fn build_layers(
        root: &Path,
        n_input_channels: i64,
        feature_size: i64,
        n_hidden_layers: usize,
        n_hidden_channels: i64,
    ) -> Vec<Linear> {
        let mut layers = Vec::new();

        layers.push(linear(
            root,
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
                root,
                n_hidden_channels,
                n_hidden_channels,
                LinearConfig {
                    ws_init: he_init(n_hidden_channels),
                    bs_init: Some(Init::Const(0.0)),
                    bias: true,
                },
            ));
        }
        layers.push(linear(
            root,
            n_hidden_channels,
            feature_size,
            LinearConfig {
                ws_init: xavier_init(n_hidden_channels, feature_size),
                bs_init: Some(Init::Const(0.0)),
                bias: true,
            },
        ));

        layers
    }

    fn forward_layers(&self, layers: &[Linear], x: &Tensor) -> Tensor {
        let mut h = x.view([-1, self.n_input_channels]);

        for i in 0..layers.len() {
            h = layers[i].forward(&h);
            if i < layers.len() - 1 {
                h = h.relu();
            }
        }

        h.view([-1, self.feature_size])
    }

    fn predictor_forward(&self, x: &Tensor) -> Tensor {
        self.forward_layers(&self.predictor_layers, x)
    }

    fn target_forward(&self, x: &Tensor) -> Tensor {
        self.forward_layers(&self.target_layers, x)
    }

    pub fn predictor_var_store(&self) -> &VarStore {
        &self.predictor_vs
    }

    fn checkpoint_path(path: &str, filename: &str) -> String {
        StdPath::new(path)
            .join(filename)
            .to_string_lossy()
            .into_owned()
    }
}

impl BasecuriosityModel for FCRNDModel {
    fn forward(&self, x: &Tensor) -> Tensor {
        let predictor_feature = self.predictor_forward(x);
        let target_feature = no_grad(|| self.target_forward(x)).detach();
        (predictor_feature - target_feature)
            .square()
            .mean_dim(&[1i64][..], false, Kind::Float)
    }

    fn device(&self) -> Device {
        debug_assert_eq!(self.predictor_vs.device(), self.target_vs.device());
        self.predictor_vs.device()
    }

    fn save(&self, path: &str) {
        fs::create_dir_all(path).expect("failed to create RND checkpoint directory");
        self.predictor_vs
            .save(Self::checkpoint_path(path, "rnd_predictor.ot"))
            .expect("failed to save RND predictor model");
        self.target_vs
            .save(Self::checkpoint_path(path, "rnd_target.ot"))
            .expect("failed to save RND target model");
    }

    fn load(&mut self, path: &str) {
        self.predictor_vs
            .load(Self::checkpoint_path(path, "rnd_predictor.ot"))
            .expect("failed to load RND predictor model");
        self.target_vs
            .load(Self::checkpoint_path(path, "rnd_target.ot"))
            .expect("failed to load RND target model");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

    #[test]
    fn test_fc_rnd_model_forward() {
        let predictor_vs = nn::VarStore::new(Device::Cpu);
        let target_vs = nn::VarStore::new(Device::Cpu);
        let model = FCRNDModel::new(predictor_vs, target_vs, 4, 8, 1, 16);

        let input = Tensor::randn([3, 4], (Kind::Float, Device::Cpu));
        let reward = model.forward(&input);

        assert_eq!(reward.size(), vec![3]);
        assert!(reward.isfinite().all().int64_value(&[]) == 1);
    }

    #[test]
    fn test_fc_rnd_model_save_and_load() {
        let predictor_vs = nn::VarStore::new(Device::Cpu);
        let target_vs = nn::VarStore::new(Device::Cpu);
        let mut model = FCRNDModel::new(predictor_vs, target_vs, 4, 8, 1, 16);
        let path = std::env::temp_dir().join(format!("reinforcex-rnd-{}", ulid::Ulid::new()));
        let path = path.to_string_lossy().into_owned();

        model.save(&path);
        model.load(&path);

        let _ = std::fs::remove_dir_all(path);
    }

    #[test]
    fn test_only_predictor_learns_and_target_stays_frozen() {
        let model = FCRNDModel::new(
            nn::VarStore::new(Device::Cpu),
            nn::VarStore::new(Device::Cpu),
            4,
            8,
            1,
            16,
        );
        // Use an active, deterministic ReLU fixture. A global manual_seed is
        // insufficient when unrelated tests initialize models concurrently.
        no_grad(|| {
            for (name, mut value) in model.predictor_vs.variables() {
                let _ = value.fill_(if name.starts_with("weight") {
                    0.05
                } else {
                    0.01
                });
            }
            for (name, mut value) in model.target_vs.variables() {
                let _ = value.fill_(if name.starts_with("weight") {
                    0.03
                } else {
                    0.02
                });
            }
        });
        let before = model
            .target_vs
            .variables()
            .into_iter()
            .map(|(name, value)| (name, value.copy()))
            .collect::<std::collections::HashMap<_, _>>();
        let input = Tensor::from_slice(&[1.0_f32, 0.5, -0.5, 2.0]).view([1, 4]);
        let initial_error = model.forward(&input).double_value(&[0]);
        let mut optimizer = nn::Adam::default()
            .build(model.predictor_var_store(), 1e-3)
            .unwrap();
        for _ in 0..100 {
            let loss = model.forward(&input).mean(Kind::Float);
            optimizer.backward_step(&loss);
        }
        let learned_error = model.forward(&input).double_value(&[0]);
        assert!(
            learned_error < initial_error * 0.1,
            "predictor did not learn: initial={initial_error}, learned={learned_error}"
        );
        for (name, value) in model.target_vs.variables() {
            assert!(!value.requires_grad());
            assert!(!value.grad().defined());
            assert!(
                value.equal(&before[&name]),
                "target parameter {name} changed"
            );
        }
    }
}
