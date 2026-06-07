use super::base_curiousity_model::BaseCuriousityModel;
use crate::misc::weight_initializer::{he_init, xavier_init};
use tch::nn::{
    linear, Adam, Init, Linear, LinearConfig, Module, Optimizer, OptimizerConfig, Path, VarStore,
};
use tch::{no_grad, Device, Kind, Tensor};

pub struct FCRNDModel {
    predictor_vs: VarStore,
    target_vs: VarStore,
    predictor_layers: Vec<Linear>,
    target_layers: Vec<Linear>,
    optimizer: Optimizer,
    n_input_channels: i64,
    feature_size: i64,
}

impl FCRNDModel {
    pub fn new(
        predictor_vs: VarStore,
        target_vs: VarStore,
        n_input_channels: i64,
        feature_size: i64,
        n_hidden_layers: usize,
        n_hidden_channels: i64,
        learning_rate: f64,
    ) -> Self {
        assert!(n_input_channels > 0);
        assert!(feature_size > 0);
        assert!(n_hidden_channels > 0);
        assert!(learning_rate > 0.0);
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
        let optimizer = Adam::default()
            .build(&predictor_vs, learning_rate)
            .expect("failed to build RND predictor optimizer");

        FCRNDModel {
            predictor_vs,
            target_vs,
            predictor_layers,
            target_layers,
            optimizer,
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
}

impl BaseCuriousityModel for FCRNDModel {
    fn forward(&self, x: &Tensor) -> Tensor {
        let predictor_feature = self.predictor_forward(x);
        let target_feature = no_grad(|| self.target_forward(x)).detach();
        (predictor_feature - target_feature)
            .square()
            .mean_dim(&[1i64][..], false, Kind::Float)
    }

    fn update(&mut self, x: &Tensor) -> Tensor {
        let loss = self.forward(x).mean(Kind::Float);
        let detached_loss = loss.detach();
        self.optimizer.zero_grad();
        loss.backward();
        self.optimizer.step();
        detached_loss
    }

    fn device(&self) -> Device {
        debug_assert_eq!(self.predictor_vs.device(), self.target_vs.device());
        self.predictor_vs.device()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn, Device, Kind, Tensor};

    #[test]
    fn test_fc_rnd_model_forward() {
        let predictor_vs = nn::VarStore::new(Device::Cpu);
        let target_vs = nn::VarStore::new(Device::Cpu);
        let model = FCRNDModel::new(predictor_vs, target_vs, 4, 8, 1, 16, 1e-3);

        let input = Tensor::randn([3, 4], (Kind::Float, Device::Cpu));
        let reward = model.forward(&input);

        assert_eq!(reward.size(), vec![3]);
        assert!(reward.isfinite().all().int64_value(&[]) == 1);
    }

    #[test]
    fn test_fc_rnd_model_update() {
        let predictor_vs = nn::VarStore::new(Device::Cpu);
        let target_vs = nn::VarStore::new(Device::Cpu);
        let mut model = FCRNDModel::new(predictor_vs, target_vs, 4, 8, 1, 16, 1e-3);

        let input = Tensor::randn([8, 4], (Kind::Float, Device::Cpu));
        let loss = model.update(&input);

        assert_eq!(loss.size(), Vec::<i64>::new());
        assert!(loss.isfinite().int64_value(&[]) == 1);
    }
}
